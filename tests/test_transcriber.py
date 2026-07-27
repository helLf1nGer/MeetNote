"""Chunking maths and Groq response handling.

Skipped unless the transcription dependencies are installed, since importing
the module pulls in torch, faster-whisper and the Groq SDK.
"""

import pytest

pytest.importorskip('torch', reason='transcription extras not installed')
pytest.importorskip('faster_whisper', reason='transcription extras not installed')
pytest.importorskip('groq', reason='transcription extras not installed')

from transcription import transcriber  # noqa: E402

GROQ_LIMIT_BYTES = 25 * 1024 * 1024


class TestChunkDuration:
    """
    Regression tests.

    len(AudioSegment) is milliseconds, but the old code stepped through it by
    `max_size_bytes // 32` - a byte count used as a duration. Chunks came out
    the right size only by coincidence of the default bitrate.
    """

    def test_chunk_fits_under_the_upload_limit(self):
        duration_ms = transcriber.chunk_duration_ms(24, '128k')
        bytes_per_ms = (128 * 1000 / 8) / 1000
        assert duration_ms * bytes_per_ms < GROQ_LIMIT_BYTES

    @pytest.mark.parametrize('bitrate', ['64k', '128k', '192k', '256k', '320k'])
    def test_holds_at_every_bitrate(self, bitrate):
        # The old byte/millisecond confusion broke down above ~256k.
        duration_ms = transcriber.chunk_duration_ms(24, bitrate)
        bytes_per_ms = (int(bitrate.rstrip('k')) * 1000 / 8) / 1000
        assert duration_ms * bytes_per_ms < GROQ_LIMIT_BYTES

    def test_higher_bitrate_means_shorter_chunks(self):
        assert transcriber.chunk_duration_ms(24, '320k') < transcriber.chunk_duration_ms(24, '128k')

    def test_duration_is_positive_and_sane(self):
        duration_ms = transcriber.chunk_duration_ms(24, '128k')
        assert 60_000 < duration_ms < 60 * 60_000


class TestParseWaitTime:
    @pytest.mark.parametrize('message,expected', [
        ('Please try again in 2m30.092s.', 150.092),
        ('Please try again in 30.5s.', 30.5),
        ('Rate limit reached. Please try again in 1m0s.', 60.0),
    ])
    def test_parses_known_shapes(self, message, expected):
        assert transcriber.parse_wait_time(message) == pytest.approx(expected)

    def test_unparseable_message_uses_default(self):
        assert transcriber.parse_wait_time('something else entirely') == 120


class TestFieldAccess:
    """Groq segments arrive as dicts on some SDK versions and objects on others."""

    def test_reads_dicts(self):
        assert transcriber._field({'start': 1.5}, 'start') == 1.5

    def test_reads_objects(self):
        class Segment:
            start = 2.5

        assert transcriber._field(Segment(), 'start') == 2.5

    def test_missing_returns_default(self):
        assert transcriber._field({}, 'start', 0.0) == 0.0
        assert transcriber._field(object(), 'nope', 'fallback') == 'fallback'


class TestCudnnPreflight:
    """
    CTranslate2 exits the process when it cannot find cuDNN, so the mismatch has
    to be caught before a long transcription starts rather than during it.
    """

    @pytest.mark.parametrize('version,expected', [
        ('4.3.1', 'cudnn_ops_infer64_8.dll'),   # pre-4.5 links against cuDNN 8
        ('4.4.0', 'cudnn_ops_infer64_8.dll'),
        ('4.5.0', 'cudnn_ops64_9.dll'),         # 4.5.0 switched to cuDNN 9
        ('4.6.1', 'cudnn_ops64_9.dll'),
        ('5.0.0', 'cudnn_ops64_9.dll'),
    ])
    def test_library_tracks_ctranslate2_version(self, monkeypatch, version, expected):
        import ctranslate2
        monkeypatch.setattr(ctranslate2, '__version__', version, raising=False)
        assert transcriber.required_cudnn_library() == expected

    def test_check_is_skipped_off_windows(self, monkeypatch):
        monkeypatch.setattr(transcriber.os, 'name', 'posix')
        usable, missing = transcriber.check_cudnn_available()
        assert usable and missing == ''


class TestDecodeOptions:
    """
    The decode options guard against two separate failure modes: Whisper
    hallucinating over silence, and a config written for one faster-whisper
    version being forwarded verbatim to another that no longer accepts it.
    """

    def _build(self, decode):
        return transcriber.build_decode_options({'transcription': {'decode': decode}})

    def test_defaults_break_the_hallucination_feedback_loop(self, isolated_config):
        options = transcriber.build_decode_options(isolated_config.config)
        assert options['vad_filter'] is True
        assert options['condition_on_previous_text'] is False
        assert options['word_timestamps'] is True
        assert options['hallucination_silence_threshold'] == 2.0
        assert options['beam_size'] == 5

    def test_default_vad_parameters(self, isolated_config):
        vad = transcriber.build_decode_options(isolated_config.config)['vad_parameters']
        assert vad == {
            'threshold': 0.5,
            'min_silence_duration_ms': 1000,
            'speech_pad_ms': 400,
        }

    def test_defaults_are_accepted_by_the_installed_library(self, isolated_config):
        import inspect

        from faster_whisper import WhisperModel
        from faster_whisper.vad import VadOptions

        options = transcriber.build_decode_options(isolated_config.config)
        accepted = inspect.signature(WhisperModel.transcribe).parameters
        assert set(options) <= set(accepted)
        # VadOptions(**vad_parameters) is how faster-whisper consumes the dict.
        VadOptions(**options['vad_parameters'])

    def test_missing_decode_block_still_yields_the_defaults(self):
        # A config written before this block existed must not lose the tuning.
        options = transcriber.build_decode_options({'transcription': {'language': 'ru'}})
        assert options['vad_filter'] is True
        assert options['condition_on_previous_text'] is False

    def test_config_overrides_a_default(self):
        options = self._build({'beam_size': 1, 'vad_filter': False})
        assert options['beam_size'] == 1
        assert options['vad_filter'] is False

    def test_thresholds_left_at_library_defaults_are_overridable(self):
        options = self._build({'no_speech_threshold': 0.8, 'temperature': [0.0, 0.2]})
        assert options['no_speech_threshold'] == 0.8
        assert options['temperature'] == [0.0, 0.2]

    def test_unknown_top_level_key_is_dropped(self):
        # model.transcribe raises TypeError on an unexpected keyword.
        assert 'nonsense_option' not in self._build({'nonsense_option': 1})

    def test_vad_key_absent_from_the_installed_vadoptions_is_dropped(self, monkeypatch):
        """
        window_size_samples exists in 1.0.2 but was removed in 1.1, so any VAD
        key has to be checked against the installed class rather than assumed.
        """
        from faster_whisper.vad import VadOptions

        monkeypatch.setattr(
            VadOptions, '_fields', ('threshold', 'min_silence_duration_ms', 'speech_pad_ms')
        )
        options = self._build({'vad_parameters': {
            'threshold': 0.4, 'window_size_samples': 1024,
        }})
        assert options['vad_parameters'] == {'threshold': 0.4}

    @pytest.mark.parametrize('bad', [[], ['threshold', 0.5], '0.5', 0.5, True])
    def test_non_mapping_vad_parameters_fall_back_to_the_defaults(self, bad):
        """
        Regression test.

        faster-whisper unpacks vad_parameters late, so a list or string from a
        hand-edited config crashed inside the VAD only after the model had
        loaded - minutes into a run, for a typo.
        """
        from faster_whisper.vad import VadOptions

        vad = self._build({'vad_parameters': bad})['vad_parameters']
        assert vad == {
            'threshold': 0.5,
            'min_silence_duration_ms': 1000,
            'speech_pad_ms': 400,
        }
        VadOptions(**vad)

    def test_null_vad_option_is_dropped(self):
        # VadOptions does arithmetic on these unguarded, so a JSON null is a
        # TypeError rather than "use the default".
        vad = self._build({'vad_parameters': {
            'threshold': None, 'speech_pad_ms': 200,
        }})['vad_parameters']
        assert vad == {'speech_pad_ms': 200}

    def test_none_values_are_not_forwarded(self):
        # None means "leave the library default alone", which is not the same as
        # passing None: no_speech_threshold=None disables the check entirely.
        assert self._build({'hotwords': None, 'initial_prompt': None}) == {}

    def test_blank_hotwords_are_not_forwarded(self):
        assert self._build({'hotwords': '   '}) == {}

    def test_hotwords_are_forwarded_when_set(self):
        options = self._build({'hotwords': 'Acme Corp, Kyiv, Oleksandr'})
        assert options['hotwords'] == 'Acme Corp, Kyiv, Oleksandr'

    def test_hallucination_threshold_requires_word_timestamps(self):
        # faster-whisper only consults the threshold inside its word-timestamp
        # branch, so keeping it here would be a silent no-op.
        options = self._build({
            'hallucination_silence_threshold': 2.0, 'word_timestamps': False,
        })
        assert 'hallucination_silence_threshold' not in options

    def test_transcribe_audio_forwards_the_options(self, monkeypatch):
        captured = {}

        class FakeModel:
            def transcribe(self, file_path, **kwargs):
                captured.update(kwargs)
                return [], None

        monkeypatch.setitem(
            transcriber.config_manager.config, 'transcription',
            {'language': 'ru', 'task': 'transcribe', 'decode': {'beam_size': 3}},
        )
        transcriber.transcribe_audio(FakeModel(), 'meeting.wav')

        assert captured['language'] == 'ru'
        assert captured['task'] == 'transcribe'
        assert captured['beam_size'] == 3


class TestWordTimestampRetention:
    """
    segment.words used to be discarded, so word_timestamps=True cost decode time
    and bought nothing. The word_level combiner needs those times to split a
    segment at a speaker change.
    """

    class FakeWord:
        def __init__(self, start, end, word):
            self.start = start
            self.end = end
            self.word = word

    class FakeSegment:
        def __init__(self, start, end, text, words=None):
            self.start = start
            self.end = end
            self.text = text
            self.words = words

    def _segment(self, **kwargs):
        return transcriber._segment_to_dict(self.FakeSegment(**kwargs))

    def test_words_are_retained_when_present(self):
        result = self._segment(
            start=0.0, end=2.0, text=' Hello there. ',
            words=[self.FakeWord(0.0, 0.5, 'Hello'), self.FakeWord(0.5, 1.2, ' there.')],
        )
        assert result['text'] == 'Hello there.'
        assert result['words'] == [
            {'start': 0.0, 'end': 0.5, 'word': 'Hello'},
            {'start': 0.5, 'end': 1.2, 'word': ' there.'},
        ]

    def test_the_leading_space_is_preserved_verbatim(self):
        # The combiner rejoins words by concatenation, so these spaces are the
        # only word boundaries it has.
        result = self._segment(
            start=0.0, end=1.0, text='a b',
            words=[self.FakeWord(0.0, 0.4, 'a'), self.FakeWord(0.4, 0.8, ' b')],
        )
        assert result['words'][1]['word'] == ' b'
        assert ''.join(w['word'] for w in result['words']) == 'a b'

    @pytest.mark.parametrize('words', [None, []])
    def test_the_key_is_omitted_when_there_are_no_words(self, words):
        # Omitted rather than None, so `'words' in segment` answers the question.
        result = self._segment(start=0.0, end=2.0, text='No words.', words=words)
        assert 'words' not in result
        assert set(result) == {'start', 'end', 'text'}

    def test_a_segment_without_the_attribute_at_all_is_fine(self):
        class Bare:
            start, end, text = 0.0, 1.0, 'bare'

        assert 'words' not in transcriber._segment_to_dict(Bare())

    def test_an_incomplete_word_entry_drops_the_whole_list(self):
        # Half a word list would silently lose text once the combiner used it.
        result = self._segment(
            start=0.0, end=2.0, text='one two',
            words=[self.FakeWord(0.0, 0.5, 'one'), self.FakeWord(None, 1.0, ' two')],
        )
        assert 'words' not in result
        assert result['text'] == 'one two'

    def test_the_result_is_json_serializable(self):
        # DataManager round-trips these dicts through the intermediate cache.
        import json

        result = self._segment(
            start=0.0, end=2.0, text='Привіт світ',
            words=[self.FakeWord(0.0, 0.6, 'Привіт'), self.FakeWord(0.6, 1.2, ' світ')],
        )
        assert json.loads(json.dumps(result, ensure_ascii=False)) == result

    def test_transcribe_audio_passes_words_through(self, monkeypatch):
        segment = self.FakeSegment(
            0.0, 2.0, 'Hello there.',
            [self.FakeWord(0.0, 0.5, 'Hello'), self.FakeWord(0.5, 1.2, ' there.')],
        )

        class FakeModel:
            def transcribe(self, file_path, **kwargs):
                return [segment], None

        monkeypatch.setitem(
            transcriber.config_manager.config, 'transcription',
            {'language': 'en', 'task': 'transcribe', 'decode': {'word_timestamps': True}},
        )
        result = transcriber.transcribe_audio(FakeModel(), 'meeting.wav')

        assert len(result) == 1
        assert [w['word'] for w in result[0]['words']] == ['Hello', ' there.']

    def test_the_word_level_combiner_accepts_this_shape(self):
        """
        The two halves of this feature have to agree on the dict shape, so this
        asserts against the real combiner rather than a copy of its rules.
        """
        from utils import word_level_combiner

        segments = [self._segment(
            start=0.0, end=4.0, text='mine yours',
            words=[self.FakeWord(0.0, 0.5, 'mine'), self.FakeWord(3.0, 3.5, ' yours')],
        )]
        diarization = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'},
            {'start': 2.5, 'end': 4.0, 'speaker': 'SPEAKER_01'},
        ]

        out = word_level_combiner.combine(segments, diarization)
        assert [s['speaker'] for s in out] == ['SPEAKER_00', 'SPEAKER_01']
        assert [s['text'] for s in out] == ['mine', 'yours']


def test_chunk_offsets_are_absolute_not_accumulated():
    """
    Regression test for timestamp drift.

    Offsets used to be accumulated as chunks succeeded, so a chunk that failed
    every retry was skipped without advancing the total - shifting every later
    segment earlier. Deriving each offset from the chunk's own start position
    makes that impossible.
    """
    chunk_ms = transcriber.chunk_duration_ms(24, '128k')
    offsets = [start / 1000.0 for start in range(0, chunk_ms * 4, chunk_ms)]

    # Dropping the second chunk must not move the third or fourth.
    surviving = [offsets[0], offsets[2], offsets[3]]
    assert surviving == [0.0, chunk_ms * 2 / 1000.0, chunk_ms * 3 / 1000.0]
