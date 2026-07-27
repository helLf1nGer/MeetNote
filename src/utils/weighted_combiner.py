import logging
import math

logger = logging.getLogger(__name__)

# Segments whose speaker matches and whose gap is shorter than this are joined
# into one output segment, so a continuous turn is not fragmented into a line per
# Whisper segment.
MERGE_GAP_SECONDS = 1.0


def segment_score(transcript_segment: dict, diarization_segment: dict, weights: dict = None) -> float:
    if weights is None:
        weights = {
            'overlap': 0.5,
            'coverage': 0.3,
            'center_distance': 0.2
        }

    # Both dicts are read defensively: this is the default combiner, and the
    # segments reaching it are not always Whisper's own. The LLM-based combiners
    # and cached transcripts can hand over a segment with no timestamps at all,
    # which used to raise KeyError here and take down the whole run.
    transcript_start = _numeric(transcript_segment.get('start'))
    transcript_end = _numeric(transcript_segment.get('end'))
    diarization_start = _numeric(diarization_segment.get('start'))
    diarization_end = _numeric(diarization_segment.get('end'))

    if None in (transcript_start, transcript_end, diarization_start, diarization_end):
        # Unscoreable, which is the same situation as no overlap: the caller
        # emits the segment as Unknown rather than dropping its text.
        return 0.0

    transcript_duration = transcript_end - transcript_start
    diarization_duration = diarization_end - diarization_start
    if transcript_duration <= 0 or diarization_duration <= 0:
        return 0.0

    overlap_start = max(transcript_start, diarization_start)
    overlap_end = min(transcript_end, diarization_end)
    overlap_duration = max(0, overlap_end - overlap_start)

    overlap_ratio = overlap_duration / transcript_duration
    coverage_ratio = overlap_duration / diarization_duration

    transcript_center = (transcript_start + transcript_end) / 2
    diarization_center = (diarization_start + diarization_end) / 2
    center_distance = abs(transcript_center - diarization_center)
    max_duration = max(transcript_duration, diarization_duration)
    normalized_distance = center_distance / max_duration

    score = (weights['overlap'] * overlap_ratio +
             weights['coverage'] * coverage_ratio -
             weights['center_distance'] * normalized_distance)

    return max(score, 0.0)


def usable_turns(diarization):
    """
    Diarization turns that can actually be scored against.

    A turn with no speaker used to be worse than useless: it still won the
    ``if best_dia`` test, so its ``None`` speaker compared equal to the initial
    ``current_speaker`` and the extend branch dereferenced ``current_segment``
    before any segment existed - a TypeError on the default code path.
    """
    turns = []
    rejected = 0

    for turn in diarization or []:
        if not isinstance(turn, dict):
            rejected += 1
            continue
        start = _numeric(turn.get('start'))
        end = _numeric(turn.get('end'))
        if start is None or end is None or end <= start or not turn.get('speaker'):
            rejected += 1
            continue
        turns.append(turn)

    if rejected:
        logger.warning("[Combiner] %d diarization turn(s) had no speaker or no usable "
                       "time range and were ignored.", rejected)

    return turns


def combine(transcription, diarization):
    combined_results = []
    current_speaker = None
    current_segment = None

    turns = usable_turns(diarization)

    for trans in transcription:
        max_score = 0
        best_dia = None
        for dia in turns:
            score = segment_score(trans, dia)
            if score > max_score:
                max_score = score
                best_dia = dia

        speaker = best_dia['speaker'] if best_dia else 'Unknown'
        start = _numeric(trans.get('start'))
        end = _numeric(trans.get('end'))
        text = trans.get('text')
        text = '' if text is None else str(text)

        # A segment with no usable timestamps is placed at the end of the one
        # before it. That is a guess about when it was spoken, but the text has
        # to reach the output, and everything downstream (the SRT writer,
        # _print_transcript) needs these to be numbers.
        if start is None:
            start = current_segment['end'] if current_segment else 0.0
        if end is None or end < start:
            end = start

        extendable = (
            # Only a matched speaker extends. Two consecutive unlabelled
            # segments are not known to be the same person, so they stay
            # separate - which is also what this combiner did before the guards
            # below were added, and the merge behaviour is deliberately
            # unchanged by this fix.
            best_dia is not None
            and current_segment is not None
            and speaker == current_speaker
            and start - current_segment['end'] < MERGE_GAP_SECONDS
        )

        if extendable:
            current_segment['end'] = max(current_segment['end'], end)
            current_segment['text'] += ' ' + text
        else:
            if current_segment:
                combined_results.append(current_segment)
            current_segment = {
                'speaker': speaker,
                'text': text,
                'start': start,
                'end': end
            }
            current_speaker = speaker

    # Add the last segment
    if current_segment:
        combined_results.append(current_segment)

    unlabelled = sum(1 for s in combined_results if s['speaker'] == 'Unknown')
    if unlabelled:
        logger.warning("[Combiner] %d of %d segments had no overlapping speaker turn.",
                       unlabelled, len(combined_results))

    return combined_results


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
