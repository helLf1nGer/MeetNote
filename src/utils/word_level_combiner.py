"""
Speaker assignment at word granularity.

Every other combiner in this repo labels a whole Whisper segment with one
speaker. Segments run 5-30 seconds and frequently contain a speaker change, so
that decision is wrong for part of any segment where the floor changed - and no
amount of scoring at segment level can fix it, because the segment itself is the
wrong unit.

Decoding now runs with ``word_timestamps=True``, so each segment carries per-word
times. Assigning every *word* to the diarization turn it overlaps most and then
regrouping consecutive same-speaker words lets a segment be split at the point
the speaker actually changed. This is the approach WhisperX takes; it needs no
model and no extra dependency.

Word data is not always there - the Groq path returns none, and a user can turn
``word_timestamps`` off - so this module delegates to ``weighted_combiner``
rather than degrading when it has nothing to work with.
"""

import logging
import math

from . import weighted_combiner

logger = logging.getLogger(__name__)

# Adjacent output segments from the same speaker with a shorter gap than this are
# merged back together. Whisper cuts segments at pauses, not at speaker changes,
# so one continuous turn usually arrives as several segments; without this the
# output would be fragmented into a line per breath. Kept below the ~1s pause
# that normally marks a real handover, so a genuine turn change is not swallowed.
MERGE_GAP_SECONDS = 0.75

# Same-speaker diarization turns closer together than this are treated as one
# continuous turn before any word is assigned. pyannote splits an utterance at
# breaths, and those fragments otherwise distort the per-turn overlap comparison
# in _assign. Deliberately tiny: this is for stitching what is really one turn
# back together, not for bridging a pause during which somebody else could have
# spoken.
MERGE_TURN_GAP_SECONDS = 0.1

# Slack for comparing gaps against the thresholds above. Timestamps arrive as
# binary floats, so a gap that is 0.1s in decimal computes as
# 0.10000000000000009 and would fall outside an inclusive 0.1 boundary. A
# microsecond is far below diarization's real resolution, so this only fixes the
# representation error and cannot merge anything meaningfully further apart.
GAP_TOLERANCE_SECONDS = 1e-6


def combine(transcription, diarization):
    """
    Assign speakers per word and regroup into speaker-homogeneous segments.

    Returns the combiner contract: dicts of exactly ``speaker``, ``text``,
    ``start`` and ``end``, sorted by ``start``. The internal ``words`` key is
    never leaked into the output.
    """
    if not transcription:
        return []

    turns = _merge_turns(_usable_turns(diarization))
    segments = _prepare(transcription)

    if not segments:
        logger.warning("[Combiner] word_level: no usable segments in the transcript.")
        return []

    # Two ways this combiner has nothing to add. In both, the weighted combiner
    # is the established behaviour, so hand over rather than invent a degraded
    # path - losing text is the one thing a combiner must never do.
    #
    # Delegation passes the normalised segments and turns rather than the
    # caller's originals. weighted_combiner now guards both of these itself, so
    # this is defence in depth, not a requirement: it is kept because _prepare
    # and _usable_turns have already run - this module needs numeric times and
    # real speakers for its own per-word assignment - so handing over the
    # normalised copies costs nothing and keeps one definition of "usable".
    if not turns:
        logger.info("[Combiner] word_level: no usable diarization turns; delegating to weighted.")
        return weighted_combiner.combine(_for_weighted(segments), turns)

    if not any(segment['words'] for segment in segments):
        logger.info("[Combiner] word_level: transcript carries no word timestamps "
                    "(Groq path, or word_timestamps disabled); delegating to weighted.")
        return weighted_combiner.combine(_for_weighted(segments), turns)

    counts = {}
    split_segments = 0
    groups = []

    for segment in segments:
        if segment['words']:
            produced = _group_words(segment['words'], turns, counts)
            if len(produced) > 1:
                split_segments += 1
            groups.extend(produced)
        else:
            # A segment with no word data inside a transcript that otherwise has
            # it - a chunk transcribed by a different path, say. Falling back to
            # whole-segment maximum overlap keeps its text in the output.
            speaker = _assign(segment['start'], segment['end'], turns, counts)
            groups.append({
                'speaker': speaker,
                'text': segment['text'],
                'start': segment['start'],
                'end': segment['end'],
            })

    groups.sort(key=lambda group: group['start'])
    combined = _merge_adjacent(groups)

    logger.info("[Combiner] word_level: %d word(s) by overlap, %d by nearest turn; "
                "%d of %d Whisper segment(s) split at a speaker change; "
                "%d output segment(s).",
                counts.get('overlap', 0), counts.get('nearest', 0),
                split_segments, len(segments), len(combined))

    return combined


def _group_words(words, turns, counts):
    """
    Assign each word a speaker and merge consecutive words that agree.

    Words are consumed in decoder order, which is authoritative for the text, but
    the span is taken as min(starts)/max(ends) rather than first/last. Whisper's
    word timestamps are not guaranteed monotonic - alignment can hand back a word
    that starts before the one printed ahead of it - and keeping the first word's
    start would produce a segment that does not contain its own words.
    """
    groups = []

    for word in words:
        speaker = _assign(word['start'], word['end'], turns, counts)

        if groups and groups[-1]['speaker'] == speaker:
            # faster-whisper's word strings carry their own leading space, so
            # they concatenate directly - inserting separators here would double
            # the spacing and break punctuation.
            groups[-1]['text'] += word['word']
            groups[-1]['start'] = min(groups[-1]['start'], word['start'])
            groups[-1]['end'] = max(groups[-1]['end'], word['end'])
        else:
            groups.append({
                'speaker': speaker,
                'text': word['word'],
                'start': word['start'],
                'end': word['end'],
            })

    finished = []
    for group in groups:
        group['text'] = group['text'].strip()
        # A run of whitespace-only word strings carries no speech, so dropping
        # it loses nothing - and an empty line in the transcript reads as a bug.
        if group['text']:
            finished.append(group)

    return finished


def _merge_adjacent(groups):
    """Join neighbouring segments from the same speaker across a short gap."""
    merged = []

    for group in groups:
        if merged:
            previous = merged[-1]
            if (previous['speaker'] == group['speaker']
                    and group['start'] - previous['end'] < MERGE_GAP_SECONDS):
                previous['end'] = max(previous['end'], group['end'])
                previous['text'] = f"{previous['text']} {group['text']}".strip()
                continue

        merged.append(dict(group))

    return merged


def _assign(start, end, turns, counts=None):
    """
    Speaker of the diarization turn this interval belongs to.

    Assignment is per *turn*, by maximum overlap, against turns that
    ``_merge_turns`` has already defragmented. Summing overlap per speaker
    instead looks equivalent and is not: two 0.5s fragments of one speaker
    collectively outvote a continuous turn that individually overlaps the word
    more, so a speaker whose turns pyannote happened to chop up wins purely on
    fragment count. Merging first, then comparing single turns, keeps the
    defragmentation benefit without letting fragments gang up.

    Ties are broken towards the *tighter* turn. A tie means the word is nested
    in two overlapping turns, which is what pyannote emits for a backchannel over
    someone else's sentence; the narrower turn localises the word better, and
    that is how an "mhm" gets attributed to the person who said it. This is not
    the short-turn bias the segment-level combiners had - that came from scoring
    overlap/turn_duration, which let a 2s turn outrank a far larger overlap. Here
    raw overlap is compared first, so a short turn only wins on an exact tie.

    A zero-duration word is a point and can overlap nothing, so it is resolved by
    containment instead - again preferring the tightest containing turn.
    """
    if not turns:
        # Unreachable from combine(), which delegates when there are no turns.
        # Kept so the helper is safe to call on its own.
        _count(counts, 'unknown')
        return 'Unknown'

    if end > start:
        candidates = []
        for turn in turns:
            overlap = min(end, turn['end']) - max(start, turn['start'])
            if overlap > 0:
                candidates.append((overlap, turn))

        if candidates:
            _count(counts, 'overlap')
            return _tightest(candidates, key=lambda item: -item[0])
    else:
        # Whisper emits start == end for a clipped token. Such a word has no
        # duration to overlap with, so containment decides it.
        containing = [
            (0.0, turn) for turn in turns if turn['start'] < start < turn['end']
        ]
        if containing:
            _count(counts, 'contained')
            return _tightest(containing)

    # Nothing overlaps or contains it: diarization missed this stretch of speech.
    # The nearest turn is a better guess than discarding the words, and either
    # way the text has to reach the output.
    midpoint = (start + end) / 2
    nearest = [
        (max(turn['start'] - midpoint, midpoint - turn['end'], 0.0), turn)
        for turn in turns
    ]
    _count(counts, 'nearest')
    return _tightest(nearest, key=lambda item: item[0])


def _tightest(candidates, key=None):
    """
    Speaker of the best candidate ``(score, turn)`` pair.

    ``key`` orders by the caller's score first (ascending); ties then go to the
    shorter turn, the earlier one, and finally the speaker label. The last two
    are arbitrary but total, which is the point: without them the answer would
    depend on the order pyannote happened to emit its turns in.
    """
    def ordering(item):
        turn = item[1]
        primary = key(item) if key else 0.0
        return (primary, turn['end'] - turn['start'], turn['start'], str(turn['speaker']))

    return min(candidates, key=ordering)[1]['speaker']


def _merge_turns(turns):
    """
    Join same-speaker diarization turns separated by no real silence.

    pyannote routinely splits one continuous utterance into several turns at
    breaths and short pauses. Left fragmented, those pieces distort per-turn
    overlap comparisons - see ``_assign``. Grouping is per speaker rather than
    between list neighbours, since the fragments of one utterance are not
    necessarily adjacent in the list.

    A gap is only closed when nobody else is speaking in it. Bridging across
    another speaker's turn would hand that speaker's words to the person either
    side of them - the exact mis-attribution this combiner exists to prevent.
    """
    by_speaker = {}
    for turn in turns:
        by_speaker.setdefault(turn['speaker'], []).append(turn)

    merged = []
    joined = 0

    for speaker, speaker_turns in by_speaker.items():
        others = [turn for turn in turns if turn['speaker'] != speaker]
        current = None
        for turn in sorted(speaker_turns, key=lambda turn: turn['start']):
            if current is not None:
                gap = turn['start'] - current['end']
                # A tolerance rather than a bare <=: the gap is floating-point
                # subtraction, so turns exactly MERGE_TURN_GAP_SECONDS apart
                # come out as 0.10000000000000009 and would miss the boundary.
                bridgeable = gap <= MERGE_TURN_GAP_SECONDS + GAP_TOLERANCE_SECONDS
                if bridgeable and not _occupied(current['end'], turn['start'], others):
                    current['end'] = max(current['end'], turn['end'])
                    joined += 1
                    continue
            current = {'start': turn['start'], 'end': turn['end'], 'speaker': speaker}
            merged.append(current)

    if joined:
        logger.info("[Combiner] word_level: merged %d fragmented diarization turn(s) "
                    "into continuous ones before assignment.", joined)

    merged.sort(key=lambda turn: turn['start'])
    return merged


def _occupied(start, end, others):
    """Whether any of ``others`` speaks inside the interval ``(start, end)``."""
    if end <= start:
        return False
    return any(turn['start'] < end and turn['end'] > start for turn in others)


def _count(counts, key):
    if counts is not None:
        counts[key] = counts.get(key, 0) + 1


def _usable_turns(diarization):
    """Diarization turns with usable times and a speaker, earliest first."""
    turns = []

    for turn in diarization or []:
        if not isinstance(turn, dict):
            continue
        start = _numeric(turn.get('start'))
        end = _numeric(turn.get('end'))
        speaker = turn.get('speaker')
        if start is None or end is None or end <= start or not speaker:
            continue
        turns.append({'start': start, 'end': end, 'speaker': speaker})

    turns.sort(key=lambda turn: turn['start'])
    return turns


def _prepare(transcription):
    """
    Normalise the transcript into segments with usable numeric times.

    A segment whose timestamps are missing or unusable is placed at the end of
    the previous one rather than discarded. That is a guess about *where* it was
    spoken, but the alternative is dropping transcribed speech, and downstream
    code (the SRT writer, ``_print_transcript``) needs numeric times to exist at
    all.

    This is required regardless of what weighted_combiner does: assigning words
    to turns and sorting the output are arithmetic on these values.
    """
    segments = []
    timeline = 0.0
    repaired = 0

    for segment in transcription:
        if not isinstance(segment, dict):
            logger.warning("[Combiner] word_level: ignoring a non-dict transcript entry: %r",
                           type(segment).__name__)
            continue

        start = _numeric(segment.get('start'))
        end = _numeric(segment.get('end'))
        if start is None:
            start = timeline
            repaired += 1
        if end is None:
            end = start
            repaired += 1

        # Guards a reversed segment, which would otherwise sort and merge
        # unpredictably.
        end = max(end, start)
        timeline = max(timeline, end)

        text = segment.get('text')
        text = '' if text is None else str(text).strip()
        segments.append({
            'start': start,
            'end': end,
            'text': text,
            'words': _usable_words(segment, text),
        })

    if repaired:
        logger.warning("[Combiner] word_level: %d missing segment timestamp(s) were "
                       "filled from the preceding segment so no text is dropped.", repaired)

    segments.sort(key=lambda segment: segment['start'])
    return segments


def _for_weighted(segments):
    """The plain ``start``/``end``/``text`` dicts weighted_combiner expects."""
    return [
        {'start': segment['start'], 'end': segment['end'], 'text': segment['text']}
        for segment in segments
    ]


def _usable_words(segment, text):
    """
    The segment's word list, or None to fall back to whole-segment assignment.

    All or nothing per segment: one malformed word would otherwise silently drop
    its text, and a segment handled at segment level still keeps every word of
    its speech.
    """
    raw = segment.get('words')
    if not isinstance(raw, (list, tuple)) or not raw:
        return None

    words = []
    for word in raw:
        if not isinstance(word, dict):
            return None
        start = _numeric(word.get('start'))
        end = _numeric(word.get('end'))
        # faster-whisper calls the field 'word'; 'text' is accepted because
        # cached transcripts from other tooling use that name.
        piece = word.get('word', word.get('text'))
        if start is None or end is None or piece is None:
            return None
        # Word order comes from the decoder and is authoritative for the text,
        # so the list is deliberately not re-sorted by time.
        words.append({'start': start, 'end': max(end, start), 'word': str(piece)})

    joined = ''.join(word['word'] for word in words).strip()

    if text and not joined:
        # Word strings that are all whitespace would replace real text with an
        # empty segment.
        return None

    # The word list has to account for the segment's whole text, not merely be
    # well-formed. A truncated list - every entry valid, but covering only the
    # first few words - passed every structural check and then silently dropped
    # the rest of the utterance, which is the one outcome a combiner must never
    # produce. Comparing on alphanumerics only, because the two spellings
    # legitimately differ in whitespace and punctuation placement.
    if text and _comparable(joined) != _comparable(text):
        logger.warning("[Combiner] word_level: word list for the segment at %.2fs does "
                       "not reconstruct its text (%d vs %d characters); falling back to "
                       "whole-segment assignment so no speech is dropped.",
                       words[0]['start'], len(_comparable(joined)), len(_comparable(text)))
        return None

    return words


def _comparable(text):
    """
    Reduce text to what two spellings of the same speech must share.

    Whisper's segment text and the concatenation of its words differ in spacing
    and sometimes punctuation, so only letters and digits are compared - enough
    to catch a truncated or mismatched word list, loose enough not to reject a
    correct one over a comma.
    """
    return ''.join(character for character in text.lower() if character.isalnum())


def _numeric(value):
    """Coerce a timestamp to float, or None when it is not a usable number."""
    # bool is an int subclass, and True would silently become 1.0 seconds.
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number
