"""Speaker-count resolution and waveform preparation.

Skipped unless the ML dependencies are installed, since importing the diarizer
pulls in torch, torchaudio and pyannote.audio.
"""

import sys

import pytest

pytest.importorskip('torch', reason='diarization extras not installed')
pytest.importorskip('torchaudio', reason='diarization extras not installed')
pytest.importorskip('pyannote.audio', reason='diarization extras not installed')

import torch  # noqa: E402

from diarization import diarizer  # noqa: E402


class TestResolveSpeakerArgs:
    """
    Forcing the count too low is the expensive mistake: pyannote's
    AgglomerativeClustering merges distinct speakers and the turn boundaries are
    gone before the combiner runs. Forcing it too high only over-splits, which a
    reader can repair.
    """

    def test_a_positive_count_is_forced(self):
        assert diarizer.resolve_speaker_args(3, {}) == {'num_speakers': 3}

    @pytest.mark.parametrize('value', [0, -1])
    def test_zero_or_less_asks_pyannote_to_decide(self, value):
        args = diarizer.resolve_speaker_args(value, {})
        assert args == {'min_speakers': 1, 'max_speakers': 10}
        assert 'num_speakers' not in args

    def test_auto_uses_the_configured_range(self):
        config = {'diarization': {'min_speakers': 2, 'max_speakers': 5}}
        assert diarizer.resolve_speaker_args(0, config) == {
            'min_speakers': 2, 'max_speakers': 5,
        }

    def test_a_forced_count_ignores_the_range(self):
        config = {'diarization': {'min_speakers': 2, 'max_speakers': 5}}
        assert diarizer.resolve_speaker_args(4, config) == {'num_speakers': 4}

    def test_none_is_treated_as_auto(self):
        assert diarizer.resolve_speaker_args(None, {}) == {
            'min_speakers': 1, 'max_speakers': 10,
        }

    @pytest.mark.parametrize('value,expected', [
        ('2', {'num_speakers': 2}),
        ('0', {'min_speakers': 1, 'max_speakers': 10}),
        (2.0, {'num_speakers': 2}),
    ])
    def test_a_numeric_string_count_is_coerced(self, value, expected):
        """
        Regression test.

        A hand-edited `"default_num_speakers": "0"` reached here verbatim and
        `'0' > 0` raised TypeError, so a config typo took the run down instead of
        falling back.
        """
        assert diarizer.resolve_speaker_args(value, {}) == expected

    @pytest.mark.parametrize('junk', ['junk', '', 'two', [], {}, object()])
    def test_an_uncoercible_count_falls_back_to_auto(self, junk):
        # Auto is the safe reading: it is also what clearing the field means.
        assert diarizer.resolve_speaker_args(junk, {}) == {
            'min_speakers': 1, 'max_speakers': 10,
        }

    @pytest.mark.parametrize('bad', [
        {'min_speakers': 8, 'max_speakers': 3},   # inverted
        {'min_speakers': 0, 'max_speakers': 10},  # pyannote requires >= 1
        {'min_speakers': 'two', 'max_speakers': 5},
        {'min_speakers': None, 'max_speakers': None},
    ])
    def test_an_invalid_range_falls_back_instead_of_raising(self, bad):
        # pyannote raises ValueError on min > max, several minutes into a run.
        args = diarizer.resolve_speaker_args(0, {'diarization': bad})
        assert args == {'min_speakers': 1, 'max_speakers': 10}

    def test_the_result_is_accepted_by_pyannote(self):
        from pyannote.audio.pipelines.utils.diarization import SpeakerDiarizationMixin

        for n_speakers in (0, 2):
            args = diarizer.resolve_speaker_args(n_speakers, {})
            # Raises if the range is invalid, which is what we are guarding.
            SpeakerDiarizationMixin.set_num_speakers(**args)


class TestSpeakerCacheIdentity:
    """
    Regression tests.

    The cache key carried the raw spinbox value, so auto mode was identified by
    the 0 that stands for the range rather than by the range itself: widening
    diarization.max_speakers reused the diarization computed with the old one.
    """

    def test_auto_identity_names_the_resolved_range(self):
        assert diarizer.speaker_cache_identity(0, {}) == 'auto:1-10'

    def test_widening_the_range_changes_the_identity(self):
        narrow = diarizer.speaker_cache_identity(
            0, {'diarization': {'min_speakers': 1, 'max_speakers': 4}})
        wide = diarizer.speaker_cache_identity(
            0, {'diarization': {'min_speakers': 1, 'max_speakers': 9}})
        assert narrow != wide

    def test_the_same_range_keeps_the_same_identity(self):
        config = {'diarization': {'min_speakers': 2, 'max_speakers': 5}}
        assert (diarizer.speaker_cache_identity(0, config)
                == diarizer.speaker_cache_identity(0, dict(config)))

    def test_a_forced_count_keeps_its_bare_number(self):
        # Entries cached before auto mode existed must stay valid, so the string
        # has to be exactly what str(num_speakers) produced before.
        assert diarizer.speaker_cache_identity(2, {}) == '2'

    def test_the_forced_cache_key_is_unchanged_by_this_indirection(self):
        from utils.data_manager import cache_key

        before = cache_key('meeting.mp4', speakers=3)
        after = cache_key('meeting.mp4', speakers=diarizer.speaker_cache_identity(3, {}))
        assert before == after

    def test_a_forced_count_is_never_confused_with_auto(self):
        forced = diarizer.speaker_cache_identity(2, {})
        auto = diarizer.speaker_cache_identity(
            0, {'diarization': {'min_speakers': 2, 'max_speakers': 2}})
        assert forced != auto

    @pytest.mark.parametrize('value', [0, -1, None, '0', 'junk'])
    def test_every_auto_spelling_lands_on_the_same_identity(self, value):
        assert diarizer.speaker_cache_identity(value, {}) == 'auto:1-10'


class TestCloudSpeakerCount:
    """
    The Vertex payload has a `num_speakers` field and no range field, and the
    serving code is not in this repository, so auto cannot be expressed. Sending
    the 0 verbatim was the one certainly-wrong option.
    """

    def test_a_forced_count_passes_through(self):
        assert diarizer.resolve_cloud_speaker_count(3, {}) == 3

    def test_auto_falls_back_to_the_configured_default(self):
        assert diarizer.resolve_cloud_speaker_count(
            0, {'diarization': {'default_num_speakers': 4}}) == 4

    def test_auto_warns_that_cloud_cannot_do_it(self, caplog):
        with caplog.at_level('WARNING'):
            diarizer.resolve_cloud_speaker_count(0, {})
        assert 'auto' in caplog.text.lower()

    @pytest.mark.parametrize('value', [0, -1, None, '0', 'junk'])
    def test_zero_is_never_sent_to_the_endpoint(self, value):
        # Zero either errors server-side or asks for no speakers at all.
        assert diarizer.resolve_cloud_speaker_count(value, {}) > 0

    def test_a_default_that_is_itself_auto_still_yields_a_number(self):
        # Nothing left to fall back to, but a number must still be sent.
        assert diarizer.resolve_cloud_speaker_count(
            0, {'diarization': {'default_num_speakers': 0}}) == 2

    def test_the_payload_carries_the_resolved_count(self, monkeypatch, tmp_path):
        """
        Covers the payload construction itself, not just the helper: the bug was
        that diarize_cloud interpolated n_speakers directly.
        """
        audio = tmp_path / 'meeting.wav'
        audio.write_bytes(b'RIFF....WAVEfmt ')
        sent = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {'predictions': [{'segments': [
                    {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'},
                ]}]}

        def fake_post(url, json=None, headers=None):
            sent.update(json['instances'][0])
            return FakeResponse()

        monkeypatch.setattr(diarizer.requests, 'post', fake_post)
        monkeypatch.setattr(diarizer, 'get_cloud_provider', lambda: {
            'provider': 'gcp', 'project_id': 'p', 'region': 'r', 'endpoint_id': 'e',
        })
        monkeypatch.setitem(diarizer.config_manager.config, 'diarization',
                            {'default_num_speakers': 3})

        # google.auth is imported inside the function; stub the module entries.
        import types
        auth = types.ModuleType('google.auth')
        auth.default = lambda: (types.SimpleNamespace(
            refresh=lambda request: None, token='t'), 'project')
        transport = types.ModuleType('google.auth.transport.requests')
        transport.Request = lambda: None
        monkeypatch.setitem(sys.modules, 'google.auth', auth)
        monkeypatch.setitem(sys.modules, 'google.auth.transport.requests', transport)

        results, device = diarizer.diarize_cloud(None, str(audio), 0)

        assert device == 'gcp'
        assert results[0]['speaker'] == 'SPEAKER_00'
        # 0 would have gone out verbatim before.
        assert sent['num_speakers'] == 3


class TestPrepareWaveform:
    """
    pyannote resamples to 16 kHz mono itself, but only once it has the audio, so
    handing it raw 44.1 kHz stereo moves ~5.5x more data onto the GPU than it
    uses - about 1.27 GB for a 60-minute meeting.
    """

    def test_stereo_44k_becomes_mono_16k(self):
        waveform = torch.rand(2, 2 * 44100)
        prepared, rate = diarizer.prepare_waveform(waveform, 44100)
        assert rate == 16000
        assert prepared.shape == (1, 2 * 16000)

    def test_it_shrinks_the_tensor(self):
        waveform = torch.rand(2, 60 * 44100)
        prepared, _ = diarizer.prepare_waveform(waveform, 44100)
        assert prepared.numel() < waveform.numel() / 5

    def test_mono_16k_is_left_alone(self):
        waveform = torch.rand(1, 16000)
        prepared, rate = diarizer.prepare_waveform(waveform, 16000)
        assert rate == 16000
        assert torch.equal(prepared, waveform)

    def test_channels_are_averaged_not_dropped(self):
        left = torch.full((1, 16000), 1.0)
        right = torch.full((1, 16000), 0.0)
        prepared, _ = diarizer.prepare_waveform(torch.cat([left, right]), 16000)
        assert prepared.shape == (1, 16000)
        assert torch.allclose(prepared, torch.full((1, 16000), 0.5))

    def test_dtype_stays_float32(self):
        prepared, _ = diarizer.prepare_waveform(
            torch.randint(-1000, 1000, (2, 44100), dtype=torch.int16), 44100
        )
        assert prepared.dtype == torch.float32

    def test_a_1d_waveform_gains_a_channel_axis(self):
        # pyannote rejects anything that is not (channel, time).
        prepared, _ = diarizer.prepare_waveform(torch.rand(44100), 44100)
        assert prepared.dim() == 2
        assert prepared.shape[0] == 1

    def test_the_result_passes_pyannotes_own_validation(self):
        from pyannote.audio.core.io import Audio

        prepared, rate = diarizer.prepare_waveform(torch.rand(2, 2 * 44100), 44100)
        validated = Audio.validate_file({'waveform': prepared, 'sample_rate': rate})
        assert validated['sample_rate'] == 16000
        assert validated['waveform'].shape[0] == 1

    def test_a_six_channel_source_still_downmixes(self):
        prepared, rate = diarizer.prepare_waveform(torch.rand(6, 48000), 48000)
        assert prepared.shape == (1, 16000)
        assert rate == 16000

    @pytest.mark.parametrize('shape', [(1, 0), (2, 0), (0,), (0, 0)])
    def test_empty_audio_raises_something_actionable(self, shape):
        """
        Regression test.

        A zero-frame file reached resample and failed with "cannot reshape tensor
        of 0 elements into shape [-1, 0]", which tells the user nothing about the
        file being empty.
        """
        with pytest.raises(ValueError, match='no samples'):
            diarizer.prepare_waveform(torch.zeros(*shape), 44100)

    def test_the_error_names_the_file(self):
        with pytest.raises(ValueError, match=r'silent\.wav'):
            diarizer.prepare_waveform(torch.zeros(1, 0), 44100,
                                      file_path='C:/audio/silent.wav')

    def test_the_error_is_tolerable_without_a_path(self):
        # prepare_waveform is also called directly, without file context.
        with pytest.raises(ValueError, match='no samples'):
            diarizer.prepare_waveform(torch.zeros(1, 0), 44100)

    def test_a_single_sample_is_not_treated_as_empty(self):
        # The guard is about zero frames, not about being short.
        prepared, rate = diarizer.prepare_waveform(torch.rand(1, 1), 16000)
        assert prepared.shape == (1, 1)
        assert rate == 16000
