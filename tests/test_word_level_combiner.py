"""
Word-level speaker assignment.

Pure dict-in/dict-out, so no ML dependency and no importorskip: this module is
deliberately the only combiner that can be tested without a model.

The behavioural cases mirror tests/test_combiners.py - a word-level combiner has
to satisfy every contract the segment-level ones do - plus the splitting that is
the reason it exists.
"""

import re
from pathlib import Path

import pytest

from utils import weighted_combiner, word_level_combiner


def words(*specs):
    """Build a faster-whisper style word list. Strings keep their leading space."""
    return [{'start': start, 'end': end, 'word': text} for start, end, text in specs]


def combine(transcription, diarization):
    out = word_level_combiner.combine(
        [dict(t) for t in transcription], [dict(d) for d in diarization]
    )
    return out, [segment['speaker'] for segment in out]


class TestMidSegmentSplit:
    """The whole point: one Whisper segment, two speakers, split at the change."""

    def test_a_segment_spanning_two_turns_becomes_two_segments(self):
        # SPEAKER_00 holds the floor to 6.0s, SPEAKER_01 from 6.0s. A
        # segment-level combiner has to mislabel one half or the other.
        transcription = [{
            'start': 4.0, 'end': 8.0,
            'text': 'so we should ship it. No, not yet.',
            'words': words(
                (4.0, 4.4, 'so'), (4.4, 4.8, ' we'), (4.8, 5.3, ' should'),
                (5.3, 5.8, ' ship'), (5.8, 6.0, ' it.'),
                (6.2, 6.6, ' No,'), (6.6, 7.0, ' not'), (7.0, 7.4, ' yet.'),
            ),
        }]
        diarization = [
            {'start': 3.5, 'end': 6.0, 'speaker': 'SPEAKER_00'},
            {'start': 6.0, 'end': 8.5, 'speaker': 'SPEAKER_01'},
        ]

        out, got = combine(transcription, diarization)

        assert got == ['SPEAKER_00', 'SPEAKER_01']
        assert out[0]['text'] == 'so we should ship it.'
        assert out[1]['text'] == 'No, not yet.'
        # The boundary lands on the word gap, not on the segment edges.
        assert out[0]['start'] == 4.0
        assert out[0]['end'] == 6.0
        assert out[1]['start'] == 6.2
        assert out[1]['end'] == 7.4

    def test_no_text_is_lost_in_a_split(self):
        transcription = [{
            'start': 0.0, 'end': 3.0,
            'text': 'one two three four',
            'words': words((0.0, 0.5, 'one'), (0.5, 1.0, ' two'),
                           (1.5, 2.0, ' three'), (2.0, 2.5, ' four')),
        }]
        diarization = [
            {'start': 0.0, 'end': 1.2, 'speaker': 'SPEAKER_00'},
            {'start': 1.2, 'end': 3.0, 'speaker': 'SPEAKER_01'},
        ]

        out, _ = combine(transcription, diarization)
        assert ' '.join(s['text'] for s in out).split() == ['one', 'two', 'three', 'four']

    def test_three_way_alternation_produces_three_segments(self):
        transcription = [{
            'start': 0.0, 'end': 6.0,
            'text': 'aaa bbb ccc',
            'words': words((0.0, 1.0, 'aaa'), (2.0, 3.0, ' bbb'), (4.0, 5.0, ' ccc')),
        }]
        diarization = [
            {'start': 0.0, 'end': 1.5, 'speaker': 'SPEAKER_00'},
            {'start': 1.8, 'end': 3.5, 'speaker': 'SPEAKER_01'},
            {'start': 3.8, 'end': 6.0, 'speaker': 'SPEAKER_02'},
        ]

        out, got = combine(transcription, diarization)
        assert got == ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']
        assert [s['text'] for s in out] == ['aaa', 'bbb', 'ccc']

    def test_a_single_speaker_segment_is_not_split(self):
        transcription = [{
            'start': 0.0, 'end': 3.0,
            'text': 'all of this is mine',
            'words': words((0.0, 0.5, 'all'), (0.5, 1.0, ' of'), (1.0, 1.5, ' this'),
                           (1.5, 2.0, ' is'), (2.0, 2.5, ' mine')),
        }]
        diarization = [{'start': 0.0, 'end': 3.0, 'speaker': 'SPEAKER_00'}]

        out, got = combine(transcription, diarization)
        assert got == ['SPEAKER_00']
        assert out[0]['text'] == 'all of this is mine'


class TestLeadingSpaceJoin:
    """faster-whisper word strings carry their own leading space."""

    def test_words_rejoin_without_doubled_spaces(self):
        transcription = [{
            'start': 0.0, 'end': 2.0,
            'text': "Hello, it's Kyiv here.",
            'words': words((0.0, 0.4, 'Hello,'), (0.4, 0.8, " it's"),
                           (0.8, 1.4, ' Kyiv'), (1.4, 1.8, ' here.')),
        }]
        diarization = [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'}]

        out, _ = combine(transcription, diarization)
        assert out[0]['text'] == "Hello, it's Kyiv here."
        assert '  ' not in out[0]['text']

    def test_the_leading_space_of_a_split_segment_is_stripped(self):
        transcription = [{
            'start': 0.0, 'end': 2.0,
            'text': 'mine yours',
            'words': words((0.0, 0.5, 'mine'), (1.0, 1.5, ' yours')),
        }]
        diarization = [
            {'start': 0.0, 'end': 0.7, 'speaker': 'SPEAKER_00'},
            {'start': 0.9, 'end': 2.0, 'speaker': 'SPEAKER_01'},
        ]

        out, _ = combine(transcription, diarization)
        assert out[1]['text'] == 'yours'
        assert not out[1]['text'].startswith(' ')

    def test_cyrillic_survives_the_round_trip(self):
        transcription = [{
            'start': 0.0, 'end': 2.0,
            'text': 'Привіт, як справи?',
            'words': words((0.0, 0.6, 'Привіт,'), (0.6, 1.0, ' як'), (1.0, 1.5, ' справи?')),
        }]
        diarization = [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'}]

        out, _ = combine(transcription, diarization)
        assert out[0]['text'] == 'Привіт, як справи?'


class TestFallbackToWeighted:
    """
    Word data is optional: the Groq path returns none and word_timestamps can be
    switched off. Delegating beats degrading - never crash, never lose text.
    """

    def test_no_words_at_all_delegates(self, caplog):
        transcription = [
            {'start': 0.0, 'end': 4.0, 'text': 'First utterance in the meeting.'},
            {'start': 4.5, 'end': 8.0, 'text': 'Second utterance in the meeting.'},
        ]
        diarization = [
            {'start': 0.0, 'end': 4.2, 'speaker': 'SPEAKER_00'},
            {'start': 4.3, 'end': 8.2, 'speaker': 'SPEAKER_01'},
        ]

        with caplog.at_level('INFO'):
            out, got = combine(transcription, diarization)

        assert out == weighted_combiner.combine(
            [dict(t) for t in transcription], [dict(d) for d in diarization]
        )
        assert got == ['SPEAKER_00', 'SPEAKER_01']
        assert 'delegating to weighted' in caplog.text

    def test_empty_word_list_delegates(self):
        transcription = [{'start': 0.0, 'end': 4.0, 'text': 'No words here.', 'words': []}]
        diarization = [{'start': 0.0, 'end': 4.0, 'speaker': 'SPEAKER_00'}]

        out, got = combine(transcription, diarization)
        assert got == ['SPEAKER_00']
        assert out[0]['text'] == 'No words here.'

    def test_malformed_word_entries_delegate_rather_than_dropping_text(self):
        # Half a word list is worse than none: the text has to survive whole.
        transcription = [{
            'start': 0.0, 'end': 4.0, 'text': 'Text that must survive.',
            'words': [{'start': 0.0, 'end': 0.5, 'word': 'Text'}, {'word': ' that'}],
        }]
        diarization = [{'start': 0.0, 'end': 4.0, 'speaker': 'SPEAKER_00'}]

        out, got = combine(transcription, diarization)
        assert got == ['SPEAKER_00']
        assert out[0]['text'] == 'Text that must survive.'

    def test_a_truncated_word_list_delegates_rather_than_dropping_text(self):
        """
        Regression test.

        Every entry was structurally valid, so the malformed-entry guard passed
        it, and the emptiness guard passed too because the words were not blank -
        but the list covered only the first word, so the rest of the utterance
        was silently discarded. Structural validity is not the same as accounting
        for the segment's text.
        """
        transcription = [{
            'start': 0.0, 'end': 2.0, 'text': 'hello world',
            'words': words((0.0, 0.5, 'hello')),
        }]
        diarization = [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'}]

        out, got = combine(transcription, diarization)
        assert got == ['SPEAKER_00']
        assert out[0]['text'] == 'hello world'

    def test_a_word_list_missing_the_middle_delegates(self):
        transcription = [{
            'start': 0.0, 'end': 3.0, 'text': 'one two three',
            'words': words((0.0, 0.5, 'one'), (2.0, 2.5, ' three')),
        }]
        diarization = [{'start': 0.0, 'end': 3.0, 'speaker': 'SPEAKER_00'}]

        out, _ = combine(transcription, diarization)
        assert out[0]['text'] == 'one two three'

    @pytest.mark.parametrize('text,pieces', [
        ('Hello, world!', ['Hello,', ' world!']),
        ('Привіт, як справи?', ['Привіт,', ' як', ' справи?']),
        ("it's fine", ["it's", ' fine']),
        ('one  two', ['one', '  two']),
        ('Yes - really.', ['Yes', ' -', ' really.']),
    ])
    def test_spacing_and_punctuation_differences_keep_the_word_data(self, text, pieces):
        """
        The completeness check must not be so strict that it rejects correct
        word lists: Whisper's segment text and its concatenated words routinely
        differ in whitespace and punctuation placement.
        """
        segment = {
            'start': 0.0, 'end': 2.0, 'text': text,
            'words': [{'start': i * 0.5, 'end': i * 0.5 + 0.4, 'word': p}
                      for i, p in enumerate(pieces)],
        }
        assert word_level_combiner._usable_words(segment, text) is not None

    def test_missing_segment_timestamps_still_produce_output(self):
        """
        Both this module and weighted_combiner guard these now; the text has to
        survive whichever path it takes. See tests/test_combiners.py
        TestMalformedInput for the shared contract.
        """
        transcription = [{'text': 'No timestamps whatsoever.'}]
        diarization = [{'start': 0.0, 'end': 4.0, 'speaker': 'SPEAKER_00'}]

        out, _ = combine(transcription, diarization)
        assert 'No timestamps whatsoever.' in ' '.join(s['text'] for s in out)
        # Downstream code formats these with %.2f, so they must exist.
        assert all(isinstance(s['start'], (int, float)) for s in out)

    def test_a_timestampless_segment_is_placed_after_the_previous_one(self):
        transcription = [
            {'start': 0.0, 'end': 5.0, 'text': 'timed',
             'words': words((0.0, 2.0, 'timed'))},
            {'text': 'untimed but must survive'},
        ]
        diarization = [{'start': 0.0, 'end': 10.0, 'speaker': 'SPEAKER_00'}]

        out, _ = combine(transcription, diarization)
        combined = ' '.join(s['text'] for s in out)
        assert 'untimed but must survive' in combined
        assert [s['start'] for s in out] == sorted(s['start'] for s in out)

    def test_a_non_dict_transcript_entry_is_ignored(self):
        transcription = [
            {'start': 0.0, 'end': 2.0, 'text': 'real segment',
             'words': words((0.0, 1.0, 'real'), (1.0, 2.0, ' segment'))},
        ]
        diarization = [{'start': 0.0, 'end': 3.0, 'speaker': 'SPEAKER_00'}]

        out = word_level_combiner.combine(transcription + ['not a dict'], diarization)
        assert [s['text'] for s in out] == ['real segment']

    def test_empty_diarization_delegates_and_keeps_the_text(self):
        transcription = [{
            'start': 0.0, 'end': 4.0, 'text': 'Talking with no diarization at all.',
            'words': words((0.0, 0.5, 'Talking'), (0.5, 1.0, ' with')),
        }]

        out, _ = combine(transcription, [])
        assert len(out) == 1
        assert 'no diarization' in out[0]['text']

    def test_unusable_diarization_turns_delegate(self):
        """
        Reversed and speaker-less turns carry no information, so they are dropped
        and the segment comes out unattributed rather than labelled 'None'.
        """
        transcription = [{
            'start': 0.0, 'end': 4.0, 'text': 'Still has to come out.',
            'words': words((0.0, 0.5, 'Still'), (0.5, 1.0, ' has')),
        }]
        diarization = [
            {'start': 4.0, 'end': 2.0, 'speaker': 'SPEAKER_00'},
            {'start': 0.0, 'end': 1.0, 'speaker': None},
        ]

        out, got = combine(transcription, diarization)
        assert 'Still has to come out.' in ' '.join(s['text'] for s in out)
        assert got == ['Unknown']

    def test_unusable_turns_are_filtered_out_of_assignment(self):
        # A good turn alongside broken ones must still be used.
        transcription = [{
            'start': 0.0, 'end': 4.0, 'text': 'assign me',
            'words': words((0.0, 1.0, 'assign'), (1.0, 2.0, ' me')),
        }]
        diarization = [
            {'start': 4.0, 'end': 2.0, 'speaker': 'SPEAKER_99'},  # reversed
            {'start': 0.0, 'end': 4.0, 'speaker': 'SPEAKER_00'},  # usable
            {'start': 'x', 'end': 1.0, 'speaker': 'SPEAKER_98'},  # unparseable
        ]

        _, got = combine(transcription, diarization)
        assert got == ['SPEAKER_00']

    def test_empty_transcription_returns_empty(self):
        assert word_level_combiner.combine([], []) == []
        assert word_level_combiner.combine(
            [], [{'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'}]
        ) == []


class TestRepoRegressionContracts:
    """The six behaviours tests/test_combiners.py encodes, at word level."""

    def test_max_overlap_not_first_overlap(self):
        """The first turn contributes 0.5s, the second 4.5s."""
        transcription = [{
            'start': 5.0, 'end': 10.0, 'text': 'and that is why we should raise now.',
            'words': words((5.6, 6.0, 'and'), (6.0, 6.5, ' that'), (6.5, 7.0, ' is'),
                           (7.0, 7.5, ' why'), (7.5, 8.0, ' we'), (8.0, 8.6, ' should'),
                           (8.6, 9.2, ' raise'), (9.2, 9.6, ' now.')),
        }]
        diarization = [
            {'start': 4.0, 'end': 5.5, 'speaker': 'SPEAKER_00'},   # 0.5s
            {'start': 5.5, 'end': 11.0, 'speaker': 'SPEAKER_01'},  # 4.5s
        ]

        _, got = combine(transcription, diarization)
        assert got == ['SPEAKER_01']

    def test_turn_contained_within_a_segment_is_found(self):
        transcription = [{
            'start': 0.0, 'end': 10.0, 'text': 'The roadmap for next quarter.',
            'words': words((3.0, 3.5, 'The'), (3.5, 4.0, ' roadmap'), (4.0, 4.4, ' for'),
                           (4.4, 4.7, ' next'), (4.7, 5.0, ' quarter.')),
        }]
        diarization = [{'start': 3.0, 'end': 5.0, 'speaker': 'SPEAKER_01'}]

        _, got = combine(transcription, diarization)
        assert got == ['SPEAKER_01']

    def test_fast_on_topic_reply_keeps_its_own_speaker(self):
        transcription = [
            {'start': 0.0, 'end': 4.0, 'text': 'The revenue numbers look strong this quarter.',
             'words': words((0.0, 0.5, 'The'), (0.5, 1.2, ' revenue'), (1.2, 1.9, ' numbers'),
                            (1.9, 2.4, ' look'), (2.4, 3.0, ' strong'), (3.0, 3.4, ' this'),
                            (3.4, 4.0, ' quarter.'))},
            {'start': 4.5, 'end': 8.0, 'text': 'Yes, the revenue numbers are strong this quarter.',
             'words': words((4.5, 4.9, 'Yes,'), (4.9, 5.2, ' the'), (5.2, 5.9, ' revenue'),
                            (5.9, 6.5, ' numbers'), (6.5, 6.9, ' are'), (6.9, 7.4, ' strong'),
                            (7.4, 7.7, ' this'), (7.7, 8.0, ' quarter.'))},
        ]
        diarization = [
            {'start': 0.0, 'end': 4.2, 'speaker': 'SPEAKER_00'},
            {'start': 4.3, 'end': 8.2, 'speaker': 'SPEAKER_01'},
        ]

        _, got = combine(transcription, diarization)
        assert got == ['SPEAKER_00', 'SPEAKER_01']

    def test_zero_overlap_text_survives_via_the_nearest_turn(self):
        transcription = [
            {'start': 0.0, 'end': 4.0, 'text': 'First speaker talking',
             'words': words((0.0, 1.0, 'First'), (1.0, 2.0, ' speaker'), (2.0, 3.0, ' talking'))},
            {'start': 50.0, 'end': 54.0, 'text': 'Speech where nobody.',
             'words': words((50.0, 51.0, 'Speech'), (51.0, 52.0, ' where'),
                            (52.0, 53.0, ' nobody.'))},
        ]
        diarization = [{'start': 0.0, 'end': 4.2, 'speaker': 'SPEAKER_00'}]

        out, _ = combine(transcription, diarization)
        combined = ' '.join(s['text'] for s in out)
        assert 'nobody.' in combined

    def test_nearest_turn_is_chosen_by_boundary_distance(self):
        # The word sits between two turns, closer to the later one.
        transcription = [{
            'start': 9.0, 'end': 9.5, 'text': 'orphan',
            'words': words((9.0, 9.5, 'orphan')),
        }]
        diarization = [
            {'start': 0.0, 'end': 5.0, 'speaker': 'SPEAKER_00'},   # 4.0s away
            {'start': 10.0, 'end': 20.0, 'speaker': 'SPEAKER_01'},  # 0.5s away
        ]

        _, got = combine(transcription, diarization)
        assert got == ['SPEAKER_01']

    def test_a_short_backchannel_only_takes_the_words_it_covers(self):
        """
        The segment-level version of this test asserts the 2s backchannel does
        not steal the whole segment. Per word the correct answer is finer: only
        the words spoken during the backchannel go to it, and the surrounding
        words stay with the speaker who holds the floor.
        """
        # The segment text is the concatenation of its own words: Whisper does
        # not emit a text that disagrees with them, and a list that fails to
        # account for the text is now treated as truncated and delegated.
        transcription = [{
            'start': 10.0, 'end': 15.0, 'text': 'so the mhm is to expand.',
            'words': words((10.0, 10.3, 'so'), (10.3, 10.5, ' the'),      # SPEAKER_00
                           (11.0, 11.5, ' mhm'),                          # inside the backchannel
                           (13.0, 13.4, ' is'), (13.4, 13.7, ' to'),      # SPEAKER_00 again
                           (13.7, 14.4, ' expand.')),
        }]
        diarization = [
            {'start': 0.0, 'end': 30.0, 'speaker': 'SPEAKER_00'},
            {'start': 10.5, 'end': 12.5, 'speaker': 'SPEAKER_01'},  # 2s, fully covered
        ]

        out, got = combine(transcription, diarization)

        assert got == ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_00']
        assert out[0]['text'] == 'so the'
        assert out[1]['text'] == 'mhm'
        assert out[2]['text'] == 'is to expand.'

    def test_an_exact_tie_goes_to_the_tighter_turn(self):
        """
        The word is nested in both turns, so coverage cannot separate them. The
        shorter turn localises it better - this is what attributes a backchannel
        to the person who actually said it.
        """
        transcription = [{
            'start': 10.0, 'end': 11.0, 'text': 'word',
            'words': words((10.6, 11.0, 'word')),
        }]
        diarization = [
            {'start': 10.5, 'end': 12.5, 'speaker': 'SPEAKER_01'},
            {'start': 0.0, 'end': 30.0, 'speaker': 'SPEAKER_00'},
        ]

        _, got = combine(transcription, diarization)
        assert got == ['SPEAKER_01']

    def test_a_short_turn_does_not_outrank_a_larger_overlap(self):
        """
        The short-turn bias the segment-level combiners had came from scoring by
        overlap/turn_duration. Raw overlap is compared first here, so a brief
        turn only ever wins when the coverage is genuinely identical.
        """
        # One long word, so the segment text matches its word list exactly.
        transcription = [{
            'start': 10.0, 'end': 14.0, 'text': 'aaaa',
            'words': words((10.0, 14.0, 'aaaa')),
        }]
        diarization = [
            {'start': 0.0, 'end': 30.0, 'speaker': 'SPEAKER_00'},   # 4.0s overlap
            {'start': 10.5, 'end': 12.5, 'speaker': 'SPEAKER_01'},  # 2.0s, fully covered
        ]

        _, got = combine(transcription, diarization)
        assert got == ['SPEAKER_00']

    def test_tie_break_does_not_depend_on_diarization_order(self):
        transcription = [{
            'start': 10.0, 'end': 11.0, 'text': 'word',
            'words': words((10.6, 11.0, 'word')),
        }]
        turns = [
            {'start': 0.0, 'end': 30.0, 'speaker': 'SPEAKER_00'},
            {'start': 10.5, 'end': 12.5, 'speaker': 'SPEAKER_01'},
        ]

        _, forward = combine(transcription, turns)
        _, reversed_order = combine(transcription, list(reversed(turns)))
        assert forward == reversed_order


class TestSegmentsWithoutWordsInAWordyTranscript:
    """A mixed transcript must not lose the segments that lack word data."""

    def test_the_wordless_segment_keeps_its_text_and_gets_a_speaker(self):
        transcription = [
            {'start': 0.0, 'end': 2.0, 'text': 'with words',
             'words': words((0.0, 0.8, 'with'), (0.8, 1.6, ' words'))},
            {'start': 10.0, 'end': 14.0, 'text': 'no words on this one'},
        ]
        diarization = [
            {'start': 0.0, 'end': 3.0, 'speaker': 'SPEAKER_00'},
            {'start': 9.5, 'end': 15.0, 'speaker': 'SPEAKER_01'},
        ]

        out, got = combine(transcription, diarization)
        assert got == ['SPEAKER_00', 'SPEAKER_01']
        assert out[1]['text'] == 'no words on this one'

    def test_output_stays_sorted_when_the_wordless_segment_comes_first(self):
        transcription = [
            {'start': 10.0, 'end': 14.0, 'text': 'later, no words'},
            {'start': 0.0, 'end': 2.0, 'text': 'earlier with words',
             'words': words((0.0, 0.8, 'earlier'), (0.8, 1.6, ' with'), (1.6, 2.0, ' words'))},
        ]
        diarization = [
            {'start': 0.0, 'end': 3.0, 'speaker': 'SPEAKER_00'},
            {'start': 9.5, 'end': 15.0, 'speaker': 'SPEAKER_01'},
        ]

        out, _ = combine(transcription, diarization)
        assert [s['start'] for s in out] == sorted(s['start'] for s in out)
        assert out[0]['text'] == 'earlier with words'


class TestMergeWindow:
    def test_same_speaker_across_a_short_gap_is_merged(self):
        gap = word_level_combiner.MERGE_GAP_SECONDS / 2
        transcription = [
            {'start': 0.0, 'end': 1.0, 'text': 'first part',
             'words': words((0.0, 0.4, 'first'), (0.4, 1.0, ' part'))},
            {'start': 1.0 + gap, 'end': 2.0 + gap, 'text': 'second part',
             'words': words((1.0 + gap, 1.4 + gap, 'second'), (1.4 + gap, 2.0 + gap, ' part'))},
        ]
        diarization = [{'start': 0.0, 'end': 5.0, 'speaker': 'SPEAKER_00'}]

        out, got = combine(transcription, diarization)
        assert got == ['SPEAKER_00']
        assert out[0]['text'] == 'first part second part'
        assert out[0]['start'] == 0.0
        assert out[0]['end'] == 2.0 + gap

    def test_same_speaker_across_a_long_gap_stays_separate(self):
        gap = word_level_combiner.MERGE_GAP_SECONDS + 0.25
        transcription = [
            {'start': 0.0, 'end': 1.0, 'text': 'first part',
             'words': words((0.0, 0.4, 'first'), (0.4, 1.0, ' part'))},
            {'start': 1.0 + gap, 'end': 2.0 + gap, 'text': 'second part',
             'words': words((1.0 + gap, 1.4 + gap, 'second'), (1.4 + gap, 2.0 + gap, ' part'))},
        ]
        diarization = [{'start': 0.0, 'end': 5.0, 'speaker': 'SPEAKER_00'}]

        out, got = combine(transcription, diarization)
        assert got == ['SPEAKER_00', 'SPEAKER_00']
        assert [s['text'] for s in out] == ['first part', 'second part']

    def test_different_speakers_are_never_merged(self):
        transcription = [
            {'start': 0.0, 'end': 1.0, 'text': 'mine',
             'words': words((0.0, 1.0, 'mine'))},
            {'start': 1.1, 'end': 2.0, 'text': 'yours',
             'words': words((1.1, 2.0, ' yours'))},
        ]
        diarization = [
            {'start': 0.0, 'end': 1.05, 'speaker': 'SPEAKER_00'},
            {'start': 1.05, 'end': 3.0, 'speaker': 'SPEAKER_01'},
        ]

        out, got = combine(transcription, diarization)
        assert got == ['SPEAKER_00', 'SPEAKER_01']
        assert [s['text'] for s in out] == ['mine', 'yours']

    def test_the_merge_window_is_below_a_typical_handover_pause(self):
        # Above ~1s a pause usually is a real turn change; merging across it
        # would undo the splitting this combiner exists for.
        assert 0 < word_level_combiner.MERGE_GAP_SECONDS < 1.0


class TestOutputShape:
    """The contract tests/test_combiners.py TestOutputShape enforces."""

    def _out(self):
        transcription = [
            {'start': 0.0, 'end': 4.0, 'text': 'First utterance',
             'words': words((0.0, 1.0, 'First'), (1.0, 2.0, ' utterance'))},
            {'start': 4.5, 'end': 8.0, 'text': 'Second utterance',
             'words': words((4.5, 5.5, 'Second'), (5.5, 6.5, ' utterance'))},
        ]
        diarization = [
            {'start': 0.0, 'end': 4.2, 'speaker': 'SPEAKER_00'},
            {'start': 4.3, 'end': 8.2, 'speaker': 'SPEAKER_01'},
        ]
        out, _ = combine(transcription, diarization)
        return out

    def test_exactly_the_four_contract_keys(self):
        out = self._out()
        assert out, 'combiner returned nothing'
        for segment in out:
            # Exactly, not merely a superset: a leaked 'words' key would reach
            # the JSON sidecar and the PDF renderer.
            assert set(segment) == {'speaker', 'text', 'start', 'end'}

    def test_timestamps_are_numeric(self):
        # _print_transcript formats these with %.2f.
        for segment in self._out():
            assert isinstance(segment['start'], (int, float))
            assert isinstance(segment['end'], (int, float))
            assert not isinstance(segment['start'], bool)

    def test_output_is_sorted_and_each_segment_advances(self):
        out = self._out()
        assert [s['start'] for s in out] == sorted(s['start'] for s in out)
        for segment in out:
            assert segment['end'] >= segment['start']

    def test_no_empty_text_segments(self):
        transcription = [{
            'start': 0.0, 'end': 2.0, 'text': 'real words here',
            'words': words((0.0, 0.5, 'real'), (0.5, 0.6, ' '), (0.6, 1.0, ' words'),
                           (1.0, 1.5, ' here')),
        }]
        diarization = [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'}]

        out, _ = combine(transcription, diarization)
        assert all(segment['text'].strip() for segment in out)

    def test_the_input_is_not_mutated(self):
        transcription = [{
            'start': 0.0, 'end': 2.0, 'text': 'do not',
            'words': words((0.0, 0.5, 'do'), (0.5, 1.0, ' not')),
        }]
        diarization = [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'}]
        expected = [dict(transcription[0])]

        word_level_combiner.combine(transcription, diarization)
        assert transcription == expected


class TestAssignHelper:
    """Direct cover for the assignment rule, independent of grouping."""

    TURNS = [
        {'start': 0.0, 'end': 5.0, 'speaker': 'SPEAKER_00'},
        {'start': 5.0, 'end': 10.0, 'speaker': 'SPEAKER_01'},
    ]

    def test_overlap_wins(self):
        assert word_level_combiner._assign(4.0, 4.5, self.TURNS) == 'SPEAKER_00'
        assert word_level_combiner._assign(6.0, 6.5, self.TURNS) == 'SPEAKER_01'

    def test_a_word_straddling_the_boundary_goes_to_the_larger_share(self):
        assert word_level_combiner._assign(4.9, 5.4, self.TURNS) == 'SPEAKER_01'
        assert word_level_combiner._assign(4.6, 5.1, self.TURNS) == 'SPEAKER_00'

    def test_fragmented_turns_win_only_after_being_merged(self):
        """
        Defragmentation is _merge_turns' job, not _assign's.

        pyannote splits one person's speech at breaths, and those pieces have to
        count as the continuous turn they really are. _assign deliberately does
        not sum them itself: summing per speaker lets two unrelated fragments
        outvote a turn that individually overlaps the word more. Merging first
        and then comparing single turns gets the fragmentation right without that
        side effect.
        """
        fragmented = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'SPEAKER_00'},
            {'start': 0.0, 'end': 1.4, 'speaker': 'SPEAKER_01'},
        ]

        # Raw, the widest single turn wins.
        assert word_level_combiner._assign(0.0, 2.0, fragmented) == 'SPEAKER_01'
        # Merged - which is what combine() feeds it - the real turn wins.
        merged = word_level_combiner._merge_turns(fragmented)
        assert word_level_combiner._assign(0.0, 2.0, merged) == 'SPEAKER_00'

    def test_micro_turns_do_not_gang_up_on_a_continuous_turn(self):
        """
        Reviewer repro for the ganging-up defect, with non-abutting fragments.

        Two fragments of B (0.7s each) used to out-total A's single continuous
        1.0s overlap purely by being two objects. Per-turn comparison ends that.

        The review's literal figures (A=[0,10], B=[0,.5]+[.5,1], word [0,1]) do
        not exercise this: B's pieces abut, so the mandated merge pre-pass turns
        them into exactly [0,1], the overlap ties at 1.0, and the outcome is then
        decided by the tie-break rather than by max overlap - see
        test_a_containment_tie_cannot_also_resolve_to_the_wider_turn.
        """
        turns = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'},    # 1.0s overlap
            {'start': 1.2, 'end': 1.9, 'speaker': 'B'},    # 0.7s
            {'start': 2.1, 'end': 2.8, 'speaker': 'B'},    # 0.7s -> 1.4s aggregated
        ]

        merged = word_level_combiner._merge_turns(turns)
        assert word_level_combiner._assign(0.0, 3.0, merged) == 'A'

        # And through the real entry point.
        transcription = [{
            'start': 0.0, 'end': 3.0, 'text': 'word',
            'words': words((0.0, 3.0, 'word')),
        }]
        _, got = combine(transcription, turns)
        assert got == ['A']

    def test_a_containment_tie_cannot_also_resolve_to_the_wider_turn(self):
        """
        Documents a genuine conflict, so the trade-off is not silently re-broken.

        Two shapes are geometrically identical - a word nested in two turns with
        equal overlap - but want opposite answers:

          this repro          word [0,1],       A=[0,10]  vs B=[0,1]     -> wants A
          backchannel (req 6) word [10.6,11.0], S00=[0,30] vs S01=[10.5,12.5]
                                                                        -> wants S01

        The first wants the wider containing turn, the second the tighter one. No
        rule over (overlap, width, containment, centre distance, coverage) gives
        both. The backchannel case wins: it is an explicit requirement with a
        real acoustic story - only the words spoken during a backchannel belong
        to it - and it describes actual meeting audio, whereas a diarization turn
        exactly coextensive with one word does not occur in practice.
        """
        turns = word_level_combiner._merge_turns([
            {'start': 0.0, 'end': 10.0, 'speaker': 'A'},
            {'start': 0.0, 'end': 0.5, 'speaker': 'B'},
            {'start': 0.5, 'end': 1.0, 'speaker': 'B'},
        ])
        # B merged to exactly [0,1]; the tie goes to the tighter turn.
        assert word_level_combiner._assign(0.0, 1.0, turns) == 'B'

        # The requirement that forces that choice still holds.
        backchannel = [
            {'start': 0.0, 'end': 30.0, 'speaker': 'SPEAKER_00'},
            {'start': 10.5, 'end': 12.5, 'speaker': 'SPEAKER_01'},
        ]
        assert word_level_combiner._assign(10.6, 11.0, backchannel) == 'SPEAKER_01'

    def test_a_wider_turn_still_wins_on_strictly_greater_overlap(self):
        # The fix must not turn into a general preference for narrow turns.
        turns = [
            {'start': 0.0, 'end': 10.0, 'speaker': 'A'},
            {'start': 0.0, 'end': 0.6, 'speaker': 'B'},
        ]
        assert word_level_combiner._assign(0.0, 1.0, turns) == 'A'

    def test_no_turns_yields_unknown(self):
        assert word_level_combiner._assign(1.0, 2.0, []) == 'Unknown'

    def test_a_zero_duration_word_is_placed_by_containment(self):
        """
        Reviewer repro. A point word has no duration to overlap with, so it fell
        through to nearest-boundary logic, where both turns scored 0.0 distance
        and the answer came down to list order. Containment is the right rule,
        and among containing turns the tightest one localises it best.
        """
        turns = [
            {'start': 0.0, 'end': 20.0, 'speaker': 'FLOOR'},
            {'start': 9.0, 'end': 11.0, 'speaker': 'BACKCHANNEL'},
        ]

        assert word_level_combiner._assign(10.0, 10.0, turns) == 'BACKCHANNEL'
        # Order of the turns must not change the answer.
        assert word_level_combiner._assign(10.0, 10.0, list(reversed(turns))) == 'BACKCHANNEL'

    def test_a_zero_duration_word_outside_every_turn_uses_the_nearest(self):
        assert word_level_combiner._assign(50.0, 50.0, self.TURNS) == 'SPEAKER_01'

    def test_a_zero_duration_word_on_a_boundary_still_gets_a_speaker(self):
        # Exactly on a turn edge is not strictly inside it, so this resolves by
        # nearest boundary - it must not raise or return None.
        assert word_level_combiner._assign(5.0, 5.0, self.TURNS) in ('SPEAKER_00', 'SPEAKER_01')

    def test_zero_length_word_still_gets_a_speaker(self):
        # Whisper occasionally emits start == end for a clipped token.
        assert word_level_combiner._assign(3.0, 3.0, self.TURNS) == 'SPEAKER_00'

    @pytest.mark.parametrize('bad', [None, 'nope', float('nan'), float('inf'), True])
    def test_unusable_timestamps_are_rejected(self, bad):
        assert word_level_combiner._numeric(bad) is None


class TestMergeTurns:
    """Defragmenting diarization before any word is assigned."""

    def test_abutting_same_speaker_turns_become_one(self):
        merged = word_level_combiner._merge_turns([
            {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'SPEAKER_00'},
        ])
        assert merged == [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'}]

    def test_a_tiny_gap_is_bridged(self):
        gap = word_level_combiner.MERGE_TURN_GAP_SECONDS / 2
        merged = word_level_combiner._merge_turns([
            {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'},
            {'start': 1.0 + gap, 'end': 2.0, 'speaker': 'SPEAKER_00'},
        ])
        assert len(merged) == 1

    def test_a_real_pause_is_not_bridged(self):
        # Someone else could have spoken in the gap; merging would hide it.
        merged = word_level_combiner._merge_turns([
            {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'},
            {'start': 3.0, 'end': 4.0, 'speaker': 'SPEAKER_00'},
        ])
        assert len(merged) == 2

    def test_overlapping_same_speaker_turns_merge(self):
        merged = word_level_combiner._merge_turns([
            {'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'},
            {'start': 1.0, 'end': 3.0, 'speaker': 'SPEAKER_00'},
        ])
        assert merged == [{'start': 0.0, 'end': 3.0, 'speaker': 'SPEAKER_00'}]

    def test_different_speakers_are_never_merged(self):
        turns = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'SPEAKER_01'},
        ]
        assert len(word_level_combiner._merge_turns(turns)) == 2

    def test_fragments_merge_across_a_speaker_listed_in_between(self):
        """
        Grouping is per speaker, not between list neighbours. The crosstalk turn
        here sits inside the first fragment rather than in the gap between the
        two, so it does not make them discontinuous.
        """
        merged = word_level_combiner._merge_turns([
            {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'},
            {'start': 0.2, 'end': 0.4, 'speaker': 'SPEAKER_01'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'SPEAKER_00'},
        ])
        assert {'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'} in merged
        assert len(merged) == 2

    def test_a_gap_filled_by_another_speaker_is_not_bridged(self):
        """
        Regression test.

        Merging per speaker without checking the gap handed a brief interjection
        to whoever spoke either side of it: A[0,1] + A[1.09,2] became A[0,2],
        swallowing B entirely. Bridging across another speaker is the exact
        mis-attribution this combiner exists to prevent.
        """
        turns = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'},
            {'start': 1.01, 'end': 1.08, 'speaker': 'B'},
            {'start': 1.09, 'end': 2.0, 'speaker': 'A'},
        ]

        merged = word_level_combiner._merge_turns(turns)
        assert len(merged) == 3
        assert {'start': 0.0, 'end': 2.0, 'speaker': 'A'} not in merged
        # B is the only speaker with any overlap in that window.
        assert word_level_combiner._assign(1.0, 1.09, merged) == 'B'

    def test_an_empty_gap_of_the_same_width_still_merges(self):
        # The gap width is unchanged from the case above; only the occupancy is.
        merged = word_level_combiner._merge_turns([
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'},
            {'start': 1.09, 'end': 2.0, 'speaker': 'A'},
        ])
        assert merged == [{'start': 0.0, 'end': 2.0, 'speaker': 'A'}]

    def test_a_gap_exactly_at_the_boundary_merges(self):
        """
        Regression test.

        The threshold is documented as inclusive, but 1.1 - 1.0 evaluates to
        0.10000000000000009 in binary floating point, so an exact `<= 0.1`
        comparison rejected turns precisely one window apart.
        """
        merged = word_level_combiner._merge_turns([
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'},
            {'start': 1.0 + word_level_combiner.MERGE_TURN_GAP_SECONDS,
             'end': 2.0, 'speaker': 'A'},
        ])
        assert len(merged) == 1

    def test_the_tolerance_cannot_bridge_a_meaningful_gap(self):
        # It exists only to absorb float representation error.
        assert word_level_combiner.GAP_TOLERANCE_SECONDS < 0.001
        merged = word_level_combiner._merge_turns([
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'},
            {'start': 1.5, 'end': 2.0, 'speaker': 'A'},
        ])
        assert len(merged) == 2

    def test_the_result_is_sorted_by_start(self):
        merged = word_level_combiner._merge_turns([
            {'start': 5.0, 'end': 6.0, 'speaker': 'SPEAKER_01'},
            {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'},
        ])
        assert [t['start'] for t in merged] == [0.0, 5.0]

    def test_the_window_is_far_below_a_conversational_pause(self):
        assert 0 < word_level_combiner.MERGE_TURN_GAP_SECONDS <= 0.25

    def test_no_turns_is_fine(self):
        assert word_level_combiner._merge_turns([]) == []


class TestOutOfOrderWordTimestamps:
    """
    Whisper's word alignment is not guaranteed monotonic, so a segment must be
    built from min(starts)/max(ends) rather than the first and last word.
    """

    def test_a_segment_covers_its_own_words(self):
        """Reviewer repro: words A[5,6] then A[1,2] claimed only [5,6]."""
        transcription = [{
            'start': 1.0, 'end': 6.0, 'text': 'later earlier',
            'words': words((5.0, 6.0, 'later'), (1.0, 2.0, ' earlier')),
        }]
        diarization = [{'start': 0.0, 'end': 10.0, 'speaker': 'SPEAKER_00'}]

        out, got = combine(transcription, diarization)

        assert got == ['SPEAKER_00']
        assert out[0]['start'] == 1.0
        assert out[0]['end'] == 6.0
        # Decoder order is authoritative for the text, so the join is unchanged.
        assert out[0]['text'] == 'later earlier'

    def test_the_span_contains_every_word_it_grouped(self):
        transcription = [{
            'start': 0.0, 'end': 9.0, 'text': 'c a b',
            'words': words((6.0, 7.0, 'c'), (1.0, 2.0, ' a'), (3.5, 4.5, ' b')),
        }]
        diarization = [{'start': 0.0, 'end': 10.0, 'speaker': 'SPEAKER_00'}]

        out, _ = combine(transcription, diarization)
        segment = out[0]
        for start, end in ((6.0, 7.0), (1.0, 2.0), (3.5, 4.5)):
            assert segment['start'] <= start and segment['end'] >= end

    def test_out_of_order_words_across_a_speaker_change(self):
        transcription = [{
            'start': 0.0, 'end': 10.0, 'text': 'mine too yours',
            'words': words((8.0, 9.0, 'mine'), (1.0, 2.0, ' too'), (4.0, 5.0, ' yours')),
        }]
        diarization = [
            {'start': 0.0, 'end': 3.0, 'speaker': 'SPEAKER_00'},
            {'start': 3.5, 'end': 10.0, 'speaker': 'SPEAKER_01'},
        ]

        out, _ = combine(transcription, diarization)
        for segment in out:
            assert segment['end'] >= segment['start']
        assert [s['start'] for s in out] == sorted(s['start'] for s in out)
        # Decoder order is preserved *within* a segment, but segments are sorted
        # by time, so a word timestamped earlier than the one printed before it
        # legitimately moves. No word may be lost.
        assert sorted(' '.join(s['text'] for s in out).split()) == ['mine', 'too', 'yours']


def test_the_per_word_fixtures_actually_reach_the_per_word_path():
    """
    Guards the tests themselves.

    A fixture whose word list does not account for its text is now treated as
    truncated and delegated to weighted, so such a fixture would keep passing
    while testing none of the per-word logic. That is how two of these tests were
    silently reduced to fallback coverage. Every fixture is therefore checked to
    be self-consistent, except the few that are deliberately not.
    """
    deliberately_inconsistent = {
        # Truncation regression tests - the mismatch IS the input under test.
        'hello world',
        'one two three',
        # Delegation is forced by the diarization side in these, so the word
        # list is irrelevant to what they assert.
        'Talking with no diarization at all.',
        'Still has to come out.',
    }

    source = Path(__file__).read_text(encoding='utf-8')
    fixtures = re.findall(
        r"\{\s*'start':[^{}]*?'text':\s*'([^']*)',\s*\n?\s*'words':\s*words\("
        r"((?:[^()]|\([^()]*\))*)\)",
        source,
    )
    assert len(fixtures) > 20, 'fixture scan found suspiciously few segments'

    unexpected = []
    for text, word_source in fixtures:
        pieces = re.findall(r"\(\s*[\d.]+\s*,\s*[\d.]+\s*,\s*'([^']*)'\s*\)", word_source)
        if not pieces:
            continue
        joined = ''.join(pieces).strip()
        consistent = (
            word_level_combiner._comparable(joined)
            == word_level_combiner._comparable(text)
        )
        if not consistent and text not in deliberately_inconsistent:
            unexpected.append((text, joined))

    assert not unexpected, (
        'these fixtures silently fall back instead of testing per-word logic: '
        f'{unexpected}'
    )


def test_dispatch_knows_the_method():
    """result_combiner must route 'word_level' here."""
    from utils import result_combiner

    assert result_combiner.word_level_combiner is word_level_combiner
