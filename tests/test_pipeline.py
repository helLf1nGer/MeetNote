"""GPU handover between diarization and local transcription.

Skipped unless the ML dependencies are installed, since importing the pipeline
pulls in torch and pyannote.audio.
"""

import pytest

pytest.importorskip('torch', reason='transcription extras not installed')
pytest.importorskip('pyannote.audio', reason='diarization extras not installed')

import torch  # noqa: E402

import pipeline as pipeline_module  # noqa: E402


class FakePyannotePipeline:
    """Records the devices it is moved to, in order."""

    def __init__(self, events, name='pipeline'):
        self._events = events
        self._name = name

    def to(self, device):
        self._events.append(f"{self._name}.to({device.type})")
        return self


@pytest.fixture
def harness(monkeypatch):
    """
    Stubs out every expensive stage of _transcribe_and_diarize and records the
    order in which they run, so the VRAM handover can be asserted on a machine
    with no GPU.
    """
    events = []
    fake_pipeline = FakePyannotePipeline(events)

    monkeypatch.setattr(
        pipeline_module.Pipeline, 'from_pretrained',
        staticmethod(lambda *a, **k: fake_pipeline),
    )
    monkeypatch.setattr(
        pipeline_module, 'diarize_audio',
        lambda *a, **k: (events.append('diarize'), ([{'start': 0.0, 'end': 1.0,
                                                     'speaker': 'SPEAKER_00'}], 'cuda'))[1],
    )
    monkeypatch.setattr(
        pipeline_module, 'create_local_model',
        lambda config: (events.append('load_whisper'), (object(), 'cuda'))[1],
    )
    monkeypatch.setattr(
        pipeline_module, 'transcribe_audio',
        lambda model, path: (events.append('transcribe'),
                             [{'start': 0.0, 'end': 1.0, 'text': 'hello'}])[1],
    )
    monkeypatch.setattr(
        pipeline_module, 'transcribe_audio_with_groq',
        lambda path: (events.append('transcribe_groq'),
                      [{'start': 0.0, 'end': 1.0, 'text': 'hello'}])[1],
    )

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: events.append('empty_cache'))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats',
                        lambda *a, **k: events.append('reset_peak'))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a, **k: 1536 * 1024 * 1024)

    return events


def _run(method='local'):
    return pipeline_module._transcribe_and_diarize(
        processed_file='meeting.wav',
        config={'use_cuda': True},
        report=lambda percent, message='': None,
        transcription_method=method,
        diarization_model='speaker-diarization-3.1',
        num_speakers=2,
        processing_location='local',
        hugging_face_token='token',
    )


class TestVramHandover:
    """
    Regression tests.

    diarize_audio moves the pyannote pipeline onto CUDA and nothing moved it
    back, so pyannote's segmentation and speaker-embedding models stayed
    resident while Whisper loaded onto the same device. empty_cache alone could
    not fix it: it only frees blocks nothing references.
    """

    def test_pipeline_leaves_the_gpu_before_whisper_loads(self, harness):
        _run('local')
        assert 'pipeline.to(cpu)' in harness
        assert harness.index('pipeline.to(cpu)') < harness.index('load_whisper')

    def test_the_cache_is_emptied_before_whisper_loads(self, harness):
        _run('local')
        assert harness.index('empty_cache') < harness.index('load_whisper')

    def test_the_move_happens_after_diarization_not_before(self, harness):
        _run('local')
        assert harness.index('diarize') < harness.index('pipeline.to(cpu)')

    def test_groq_path_releases_only_once_diarization_has_finished(self, harness):
        # The Groq path runs diarization in a worker thread; moving the pipeline
        # while that thread is still using it would fail.
        _run('groq')
        assert harness.index('diarize') < harness.index('pipeline.to(cpu)')
        assert 'load_whisper' not in harness

    def test_results_are_still_returned(self, harness):
        transcription, diarization, device = _run('local')
        assert transcription[0]['text'] == 'hello'
        assert diarization[0]['speaker'] == 'SPEAKER_00'
        assert device == 'cuda'

    def test_nothing_is_moved_when_cuda_is_disabled(self, monkeypatch, harness):
        pipeline_module._transcribe_and_diarize(
            processed_file='meeting.wav',
            config={'use_cuda': False},
            report=lambda percent, message='': None,
            transcription_method='local',
            diarization_model='speaker-diarization-3.1',
            num_speakers=2,
            processing_location='local',
            hugging_face_token='token',
        )
        assert 'pipeline.to(cpu)' not in harness
        assert 'empty_cache' not in harness

    @pytest.mark.parametrize('failing', ['diarize_audio', 'transcribe_audio_with_groq'])
    def test_groq_path_releases_the_gpu_even_when_a_stage_fails(
        self, monkeypatch, harness, failing
    ):
        """
        Regression test.

        The release used to sit after the `with` block as a plain statement, so
        a raising Future.result() propagated straight past it - and the
        traceback keeps this frame alive, so the pipeline stayed on the device.
        """
        def explode(*args, **kwargs):
            raise RuntimeError('stage failed')

        monkeypatch.setattr(pipeline_module, failing, explode)
        with pytest.raises(RuntimeError, match='stage failed'):
            _run('groq')
        assert 'pipeline.to(cpu)' in harness

    def test_peak_stats_are_reset_before_the_run(self, harness):
        # max_memory_allocated is process-lifetime, so without a reset the
        # second run in a long-lived GUI reports the first run's peak.
        _run('local')
        assert harness[0] == 'reset_peak'

    def test_peak_stats_are_not_reset_when_cuda_is_disabled(self, harness):
        pipeline_module._transcribe_and_diarize(
            processed_file='meeting.wav',
            config={'use_cuda': False},
            report=lambda percent, message='': None,
            transcription_method='local',
            diarization_model='speaker-diarization-3.1',
            num_speakers=2,
            processing_location='local',
            hugging_face_token='token',
        )
        assert 'reset_peak' not in harness

    def test_a_failed_move_does_not_abort_the_run(self, monkeypatch, harness):
        # Losing the headroom is survivable; losing the transcription is not.
        def explode(device):
            raise RuntimeError('device busy')

        monkeypatch.setattr(pipeline_module.Pipeline, 'from_pretrained', staticmethod(
            lambda *a, **k: type('Broken', (), {'to': staticmethod(explode)})()
        ))
        transcription, _, _ = _run('local')
        assert transcription[0]['text'] == 'hello'


class FakeBatchedPipeline(FakePyannotePipeline):
    """Stands in for SpeakerDiarization, which carries both batch-size knobs."""

    def __init__(self, events, name='pipeline'):
        super().__init__(events, name)
        self.segmentation_batch_size = 1
        self.embedding_batch_size = 1


class TestBatchSizes:
    """
    pyannote.audio 3.3.1 defaults both batch sizes to 1, so the segmentation and
    embedding models are fed one chunk at a time.
    """

    def test_defaults_raise_both_above_one(self):
        pipeline = FakeBatchedPipeline([])
        pipeline_module._apply_batch_sizes(pipeline, {})
        assert pipeline.segmentation_batch_size == pipeline_module.DEFAULT_BATCH_SIZE
        assert pipeline.embedding_batch_size == pipeline_module.DEFAULT_BATCH_SIZE
        # The whole point is not being pyannote's 1.
        assert pipeline_module.DEFAULT_BATCH_SIZE > 1

    def test_the_default_matches_the_config_default(self):
        # Two sources for the same number; a drift would make the fallback path
        # behave differently from the normal one for no visible reason.
        from utils.config_manager import ConfigManager

        diarization = ConfigManager()._get_default_config()['diarization']
        assert diarization['segmentation_batch_size'] == pipeline_module.DEFAULT_BATCH_SIZE
        assert diarization['embedding_batch_size'] == pipeline_module.DEFAULT_BATCH_SIZE

    def test_config_overrides_them_independently(self):
        pipeline = FakeBatchedPipeline([])
        pipeline_module._apply_batch_sizes(pipeline, {'diarization': {
            'segmentation_batch_size': 8, 'embedding_batch_size': 16,
        }})
        assert pipeline.segmentation_batch_size == 8
        assert pipeline.embedding_batch_size == 16

    @pytest.mark.parametrize('bad', [0, -4, 'many', None])
    def test_an_unusable_value_falls_back_to_the_default(self, bad):
        pipeline = FakeBatchedPipeline([])
        pipeline_module._apply_batch_sizes(pipeline, {'diarization': {
            'segmentation_batch_size': bad,
        }})
        assert pipeline.segmentation_batch_size == pipeline_module.DEFAULT_BATCH_SIZE

    def test_a_pipeline_without_the_knobs_is_left_alone(self):
        # The model dropdown also offers pipelines that are not SpeakerDiarization.
        pipeline = FakePyannotePipeline([])
        pipeline_module._apply_batch_sizes(pipeline, {})
        assert not hasattr(pipeline, 'segmentation_batch_size')

    def test_a_rejected_assignment_does_not_abort_the_run(self):
        class Stubborn(FakeBatchedPipeline):
            @property
            def segmentation_batch_size(self):
                return 1

        pipeline = Stubborn.__new__(Stubborn)
        pipeline.embedding_batch_size = 1
        # Throughput is worth having, but not at the cost of the run.
        pipeline_module._apply_batch_sizes(pipeline, {})
        assert pipeline.embedding_batch_size == pipeline_module.DEFAULT_BATCH_SIZE

    def test_they_are_applied_before_diarization_starts(self, monkeypatch, harness):
        pipeline = FakeBatchedPipeline(harness)
        monkeypatch.setattr(pipeline_module.Pipeline, 'from_pretrained',
                            staticmethod(lambda *a, **k: pipeline))
        _run('local')
        assert pipeline.segmentation_batch_size == pipeline_module.DEFAULT_BATCH_SIZE
        assert pipeline.embedding_batch_size == pipeline_module.DEFAULT_BATCH_SIZE


class TestCacheIdentity:
    """
    Regression tests for the key run_pipeline builds.

    The expensive stages are cached, so anything that changes what the
    diarization should contain has to be in the key or a stale result is reused.
    """

    @staticmethod
    def _key(num_speakers, config):
        from utils.data_manager import cache_key

        return cache_key(
            'meeting.mp4',
            speakers=pipeline_module.speaker_cache_identity(num_speakers, config),
        )

    def test_changing_the_auto_range_invalidates_the_entry(self):
        narrow = self._key(0, {'diarization': {'min_speakers': 1, 'max_speakers': 4}})
        wide = self._key(0, {'diarization': {'min_speakers': 1, 'max_speakers': 9}})
        assert narrow != wide

    def test_changing_the_minimum_invalidates_the_entry(self):
        assert (self._key(0, {'diarization': {'min_speakers': 1, 'max_speakers': 10}})
                != self._key(0, {'diarization': {'min_speakers': 3, 'max_speakers': 10}}))

    def test_the_range_is_irrelevant_when_a_count_is_forced(self):
        # pyannote ignores min/max once num_speakers is given, so the cached
        # result really is still valid.
        assert (self._key(2, {'diarization': {'max_speakers': 4}})
                == self._key(2, {'diarization': {'max_speakers': 9}}))

    def test_auto_and_forced_are_distinct_identities(self):
        assert self._key(0, {}) != self._key(2, {})

    def test_run_pipeline_puts_the_resolved_range_in_the_key(
        self, monkeypatch, tmp_path, isolated_config
    ):
        """
        Covers the call site, not just the helper: the bug was run_pipeline
        passing the raw spinbox value, so a helper-only test would still pass.

        isolated_config is required, not optional - run_pipeline calls
        save_config(), which would otherwise write this test's settings over the
        developer's real Config/config.json.
        """
        media = tmp_path / 'meeting.wav'
        media.write_bytes(b'audio')
        keys = []

        monkeypatch.setattr(pipeline_module, 'process_file', lambda path: str(media))
        monkeypatch.setattr(pipeline_module, 'cache_key',
                            lambda path, **settings: (keys.append(settings), 'k')[1])
        # Stop right after the key is built; the rest needs a GPU and a token.
        monkeypatch.setattr(pipeline_module.DataManager, 'results_exist',
                            lambda self, key: (_ for _ in ()).throw(RuntimeError('stop')))

        isolated_config.config['diarization'] = {'min_speakers': 2, 'max_speakers': 7}

        for speakers in (0, 2):
            with pytest.raises(RuntimeError, match='stop'):
                pipeline_module.run_pipeline({
                    'file_path': str(media), 'num_speakers': speakers,
                })

        assert keys[0]['speakers'] == 'auto:2-7'
        # A forced count keeps the bare number older cache entries were built with.
        assert keys[1]['speakers'] == '2'


def test_peak_vram_is_logged(harness, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger=pipeline_module.logger.name):
        _run('local')
    # Worded for what it actually covers: torch allocations up to the end of
    # transcription. Not CTranslate2 (Whisper), and not the combiners, which
    # load after this point.
    assert any('Peak torch VRAM through transcription' in record.message
               for record in caplog.records)
