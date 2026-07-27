"""
Contract tests for the speaker-assignment combiners.

These encode behaviour that was verified empirically against the real models.
Each case is one that a combiner in this repo actually got wrong.

The semantic combiners load a sentence-transformer, so they are skipped unless
it is installed.
"""

import importlib

import pytest

# Combiners with no ML dependency - always tested.
PLAIN = ['simple', 'weighted']
# Assigns speakers per word. It has no ML dependency either, but it is listed
# separately because it is not a candidate default until it is wired up.
# The fixtures below carry no 'words' key, so these cases exercise its
# delegation to weighted - which is exactly the path a Groq transcript takes.
WORD_LEVEL = ['word_level']
# Every combiner that can be tested without downloading a model.
NO_ML = PLAIN + WORD_LEVEL
# Combiners that load a SentenceTransformer.
SEMANTIC = ['semantic', 'semantic_adaptive', 'semantic_flow', 'adaptive', 'adaptive_rule']

# Combiners that survive a malformed segment or turn. Every other combiner in
# this repo indexes ``segment['start']`` unguarded and raises KeyError on a
# segment carrying no timestamps - see TestMalformedInput for which ones, why it
# is not being fixed here, and what it costs.
ROBUST = ['weighted', 'word_level']

MODULES = {
    'simple': 'utils.simple_combiner',
    'weighted': 'utils.weighted_combiner',
    'word_level': 'utils.word_level_combiner',
    'semantic': 'utils.semantic_combiner',
    'semantic_adaptive': 'utils.semantic_combiner_adaptive',
    'semantic_flow': 'utils.semantic_flow_combiner',
    'adaptive': 'utils.adaptive_combiner',
    'adaptive_rule': 'utils.adaptive_rule_combiner',
}


def load(name):
    if name in SEMANTIC:
        pytest.importorskip('sentence_transformers', reason='semantic extras not installed')
    return importlib.import_module(MODULES[name])


def speakers(module, transcription, diarization):
    out = module.combine([dict(t) for t in transcription], [dict(d) for d in diarization])
    return out, [s.get('speaker') for s in out]


class TestNoTextIsEverLost:
    """
    The strongest contract: a combiner may mislabel a speaker, but it must
    never drop transcribed speech. `simple` used to delete any segment with no
    overlapping speaker turn.
    """

    @pytest.mark.parametrize('name', NO_ML + SEMANTIC)
    def test_segment_without_overlap_survives(self, name):
        module = load(name)
        transcription = [
            {'start': 0.0, 'end': 4.0, 'text': 'First speaker talking about the plan.'},
            {'start': 50.0, 'end': 54.0, 'text': 'Speech where diarization found nobody.'},
        ]
        diarization = [{'start': 0.0, 'end': 4.2, 'speaker': 'SPEAKER_00'}]

        out, _ = speakers(module, transcription, diarization)
        combined = ' '.join(s['text'] for s in out)
        assert 'diarization found nobody' in combined

    @pytest.mark.parametrize('name', NO_ML + SEMANTIC)
    def test_empty_diarization_does_not_lose_text_or_crash(self, name):
        module = load(name)
        transcription = [{'start': 0.0, 'end': 4.0, 'text': 'Talking with no diarization at all.'}]

        out, _ = speakers(module, transcription, [])
        assert len(out) == 1
        assert 'no diarization' in out[0]['text']


class TestSpeakerAssignment:
    """Assignment must follow maximum temporal overlap."""

    @pytest.mark.parametrize('name', NO_ML + ['semantic', 'semantic_flow'])
    def test_segment_straddling_a_speaker_change(self, name):
        """
        The first overlapping turn contributes 0.5s, the second 4.5s.
        semantic_adaptive picked the first because it scored every candidate
        against an empty string, making the score constant.
        """
        module = load(name)
        transcription = [{'start': 5.0, 'end': 10.0, 'text': 'and that is why we should raise now.'}]
        diarization = [
            {'start': 4.0, 'end': 5.5, 'speaker': 'SPEAKER_00'},   # 0.5s
            {'start': 5.5, 'end': 11.0, 'speaker': 'SPEAKER_01'},  # 4.5s
        ]

        _, got = speakers(module, transcription, diarization)
        assert got == ['SPEAKER_01']

    @pytest.mark.parametrize('name', NO_ML + ['semantic', 'semantic_flow'])
    def test_turn_contained_within_a_segment(self, name):
        """A turn wholly inside a segment must still be found."""
        module = load(name)
        transcription = [{'start': 0.0, 'end': 10.0, 'text': 'The roadmap for next quarter.'}]
        diarization = [{'start': 3.0, 'end': 5.0, 'speaker': 'SPEAKER_01'}]

        _, got = speakers(module, transcription, diarization)
        assert got == ['SPEAKER_01']

    @pytest.mark.parametrize('name', NO_ML + ['semantic_adaptive'])
    def test_short_backchannel_does_not_steal_the_segment(self, name):
        """
        A 2s "mhm" fully covered by the segment must not outrank a 5s overlap
        with the speaker actually talking. Combiners that score by
        overlap/diarization_duration reward short turns and get this wrong.
        """
        module = load(name)
        transcription = [{'start': 10.0, 'end': 15.0, 'text': 'so the plan is to expand.'}]
        diarization = [
            {'start': 0.0, 'end': 30.0, 'speaker': 'SPEAKER_00'},   # 5.0s overlap
            {'start': 10.5, 'end': 12.5, 'speaker': 'SPEAKER_01'},  # 2.0s, 100% covered
        ]

        _, got = speakers(module, transcription, diarization)
        assert got == ['SPEAKER_00']

    @pytest.mark.parametrize('name', NO_ML + ['semantic', 'semantic_adaptive'])
    def test_fast_on_topic_reply_keeps_its_own_speaker(self, name):
        """
        semantic_flow relabelled a genuine speaker change as the previous
        speaker whenever the reply was quick and on-topic - which describes
        most turn-taking in a meeting.
        """
        module = load(name)
        transcription = [
            {'start': 0.0, 'end': 4.0, 'text': 'The revenue numbers look strong this quarter.'},
            {'start': 4.5, 'end': 8.0, 'text': 'Yes, the revenue numbers are strong this quarter.'},
        ]
        diarization = [
            {'start': 0.0, 'end': 4.2, 'speaker': 'SPEAKER_00'},
            {'start': 4.3, 'end': 8.2, 'speaker': 'SPEAKER_01'},
        ]

        _, got = speakers(module, transcription, diarization)
        assert got == ['SPEAKER_00', 'SPEAKER_01']


class TestOutputShape:
    @pytest.mark.parametrize('name', NO_ML + SEMANTIC)
    def test_every_segment_has_the_expected_keys(self, name):
        module = load(name)
        transcription = [
            {'start': 0.0, 'end': 4.0, 'text': 'First utterance in the meeting.'},
            {'start': 4.5, 'end': 8.0, 'text': 'Second utterance in the meeting.'},
        ]
        diarization = [
            {'start': 0.0, 'end': 4.2, 'speaker': 'SPEAKER_00'},
            {'start': 4.3, 'end': 8.2, 'speaker': 'SPEAKER_01'},
        ]

        out, _ = speakers(module, transcription, diarization)
        assert out, 'combiner returned nothing'
        for segment in out:
            assert {'speaker', 'text', 'start', 'end'} <= set(segment)
            # _print_transcript formats these with %.2f, so they must be numeric.
            assert isinstance(segment['start'], (int, float))
            assert isinstance(segment['end'], (int, float))


class TestMalformedInput:
    """
    Regression tests for two crashes on the default code path.

    Both were found by the word_level combiner delegating to `weighted` and are
    fixed in `weighted` only. The parametrize lists are deliberately narrow:

    - A segment with no timestamps raised KeyError in `weighted.segment_score`.
      `simple` and every semantic combiner still do this - they index
      `segment['start']` directly. Widening these cases to them would just
      record known breakage as a failing suite, so they are scoped to ROBUST and
      named here instead. Reachable in production: the LLM-based combiners and
      older cached transcripts emit segments without timestamps.
    - A turn with `speaker: None` satisfied `if best_dia`, so its None speaker
      compared equal to the initial `current_speaker` and the extend branch
      dereferenced `current_segment` before one existed - TypeError. Only
      `semantic` shares the crash; `simple`, `semantic_adaptive` and
      `semantic_flow` survive it but label the segment `None` rather than
      'Unknown', which then reaches the PDF as the literal text "None". Hence
      the speaker assertion is ROBUST-only while the no-crash assertion is
      not - see test_a_none_speaker_turn_does_not_crash_anyone.
    """

    @pytest.mark.parametrize('name', ROBUST)
    def test_segment_without_timestamps_survives_as_unknown(self, name):
        module = load(name)
        transcription = [{'text': 'Speech with no timestamps at all.'}]
        diarization = [{'start': 0.0, 'end': 4.0, 'speaker': 'SPEAKER_00'}]

        out, got = speakers(module, transcription, diarization)

        assert 'no timestamps at all' in ' '.join(s['text'] for s in out)
        assert got == ['Unknown']
        # Downstream formats these with %.2f and writes them into SRT cues.
        for segment in out:
            assert isinstance(segment['start'], (int, float))
            assert isinstance(segment['end'], (int, float))

    @pytest.mark.parametrize('name', ROBUST)
    def test_a_timestampless_segment_among_timed_ones_is_not_dropped(self, name):
        module = load(name)
        transcription = [
            {'start': 0.0, 'end': 4.0, 'text': 'Properly timed speech.'},
            {'text': 'Untimed speech that must still appear.'},
            {'start': 8.0, 'end': 12.0, 'text': 'More timed speech.'},
        ]
        diarization = [{'start': 0.0, 'end': 12.0, 'speaker': 'SPEAKER_00'}]

        out, _ = speakers(module, transcription, diarization)
        combined = ' '.join(s['text'] for s in out)

        assert 'Untimed speech that must still appear.' in combined
        assert 'Properly timed speech.' in combined
        assert 'More timed speech.' in combined
        assert [s['start'] for s in out] == sorted(s['start'] for s in out)

    @pytest.mark.parametrize('name', ROBUST)
    def test_unusable_timestamps_do_not_crash(self, name):
        module = load(name)
        transcription = [
            {'start': None, 'end': None, 'text': 'nulls'},
            {'start': 'x', 'end': 'y', 'text': 'strings'},
            {'start': float('nan'), 'end': 2.0, 'text': 'nan'},
        ]
        diarization = [{'start': 0.0, 'end': 12.0, 'speaker': 'SPEAKER_00'}]

        out, _ = speakers(module, transcription, diarization)
        combined = ' '.join(s['text'] for s in out)
        for text in ('nulls', 'strings', 'nan'):
            assert text in combined

    @pytest.mark.parametrize('name', ROBUST)
    def test_a_none_speaker_turn_yields_unknown_not_none(self, name):
        """
        'None' as a speaker label reaches the PDF as the literal text "None".
        """
        module = load(name)
        transcription = [{'start': 0.0, 'end': 4.0, 'text': 'Speech to attribute.'}]
        diarization = [{'start': 0.0, 'end': 1.0, 'speaker': None}]

        out, got = speakers(module, transcription, diarization)
        assert got == ['Unknown']
        assert 'Speech to attribute.' in out[0]['text']

    @pytest.mark.parametrize('name', [n for n in NO_ML + SEMANTIC if n != 'semantic'])
    def test_a_none_speaker_turn_does_not_crash_anyone(self, name):
        """
        Wider than the assertion above: crashing is worse than mislabelling, and
        every combiner except `semantic` already clears this bar. `semantic` is
        excluded rather than fixed - it is out of scope for this change.
        """
        module = load(name)
        transcription = [{'start': 0.0, 'end': 4.0, 'text': 'Speech to attribute.'}]
        diarization = [{'start': 0.0, 'end': 1.0, 'speaker': None}]

        out, _ = speakers(module, transcription, diarization)
        assert 'Speech to attribute.' in ' '.join(str(s['text']) for s in out)

    @pytest.mark.parametrize('name', ROBUST)
    def test_a_usable_turn_alongside_broken_ones_is_still_used(self, name):
        module = load(name)
        transcription = [{'start': 1.0, 'end': 3.0, 'text': 'Attribute me correctly.'}]
        diarization = [
            {'start': 4.0, 'end': 2.0, 'speaker': 'SPEAKER_98'},   # reversed
            {'start': 0.0, 'end': 4.0, 'speaker': 'SPEAKER_00'},   # usable
            {'start': None, 'end': 4.0, 'speaker': 'SPEAKER_97'},  # no start
            {'start': 0.0, 'end': 4.0, 'speaker': ''},             # no speaker
        ]

        _, got = speakers(module, transcription, diarization)
        assert got == ['SPEAKER_00']

    @pytest.mark.parametrize('name', ROBUST)
    def test_a_non_dict_turn_does_not_crash(self, name):
        module = load(name)
        # combine() is called directly: the `speakers` helper copies each turn
        # with dict(), which is itself what fails on a None entry.
        out = module.combine(
            [{'start': 0.0, 'end': 4.0, 'text': 'Still has to come out.'}],
            [None, 'SPEAKER_00', {'start': 0.0, 'end': 4.0, 'speaker': 'SPEAKER_00'}],
        )

        assert [s['speaker'] for s in out] == ['SPEAKER_00']
        assert 'Still has to come out.' in out[0]['text']


class TestWeightedMergeBehaviourIsUnchanged:
    """
    The crash fix rewrote `combine`'s loop, so the merge behaviour it is built
    around is pinned here separately.
    """

    def test_consecutive_same_speaker_segments_merge(self):
        module = load('weighted')
        transcription = [
            {'start': 0.0, 'end': 4.0, 'text': 'First part.'},
            {'start': 4.2, 'end': 8.0, 'text': 'Second part.'},
        ]
        diarization = [{'start': 0.0, 'end': 9.0, 'speaker': 'SPEAKER_00'}]

        out, got = speakers(module, transcription, diarization)
        assert got == ['SPEAKER_00']
        assert out[0]['text'] == 'First part. Second part.'
        assert (out[0]['start'], out[0]['end']) == (0.0, 8.0)

    def test_a_gap_beyond_the_window_starts_a_new_segment(self):
        module = load('weighted')
        transcription = [
            {'start': 0.0, 'end': 4.0, 'text': 'First part.'},
            {'start': 5.5, 'end': 8.0, 'text': 'Second part.'},
        ]
        diarization = [{'start': 0.0, 'end': 9.0, 'speaker': 'SPEAKER_00'}]

        out, got = speakers(module, transcription, diarization)
        assert got == ['SPEAKER_00', 'SPEAKER_00']
        assert [s['text'] for s in out] == ['First part.', 'Second part.']

    def test_two_unlabelled_segments_stay_separate(self):
        """
        Unknown is not a speaker: two unattributed segments are not known to be
        the same person, so they must not be merged into one paragraph.
        """
        module = load('weighted')
        transcription = [
            {'start': 0.0, 'end': 1.0, 'text': 'First orphan.'},
            {'start': 1.1, 'end': 2.0, 'text': 'Second orphan.'},
        ]

        out, got = speakers(module, transcription, [])
        assert got == ['Unknown', 'Unknown']
        assert [s['text'] for s in out] == ['First orphan.', 'Second orphan.']


def test_default_combiner_passes_every_contract():
    """The configured default must be one that survives all of the above."""
    from utils.config_manager import ConfigManager

    assert ConfigManager()._get_default_config()['combiner']['method'] in PLAIN
