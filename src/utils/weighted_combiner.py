import logging

logger = logging.getLogger(__name__)

def segment_score(transcript_segment, diarization_segment, weights=None):
    if weights is None:
        weights = {
            'overlap': 0.5,
            'coverage': 0.3,
            'center_distance': 0.2
        }

    overlap_start = max(transcript_segment['start'], diarization_segment['start'])
    overlap_end = min(transcript_segment['end'], diarization_segment['end'])
    overlap_duration = max(0, overlap_end - overlap_start)

    if overlap_duration == 0:
        return 0

    transcript_duration = transcript_segment['end'] - transcript_segment['start']
    diarization_duration = diarization_segment['end'] - diarization_segment['start']

    overlap_ratio = overlap_duration / transcript_duration
    coverage_ratio = overlap_duration / diarization_duration

    transcript_center = (transcript_segment['start'] + transcript_segment['end']) / 2
    diarization_center = (diarization_segment['start'] + diarization_segment['end']) / 2
    center_distance = abs(transcript_center - diarization_center)
    max_duration = max(transcript_duration, diarization_duration)
    normalized_distance = center_distance / max_duration

    score = (weights['overlap'] * overlap_ratio +
             weights['coverage'] * coverage_ratio -
             weights['center_distance'] * normalized_distance)

    return max(score, 0)

def combine(transcription, diarization):
    combined_results = []
    current_speaker = None
    current_segment = None

    for trans in transcription:
        max_score = 0
        best_dia = None
        for dia in diarization:
            score = segment_score(trans, dia)
            if score > max_score:
                max_score = score
                best_dia = dia

        if best_dia:
            if best_dia['speaker'] == current_speaker and trans['start'] - current_segment['end'] < 1.0:  # 1 second gap threshold
                # Extend the current segment
                current_segment['end'] = trans['end']
                current_segment['text'] += ' ' + trans['text']
            else:
                # Start a new segment
                if current_segment:
                    combined_results.append(current_segment)
                current_segment = {
                    'speaker': best_dia['speaker'],
                    'text': trans['text'],
                    'start': trans['start'],
                    'end': trans['end']
                }
                current_speaker = best_dia['speaker']
        else:
            # No matching diarization segment, create a new "Unknown" segment
            if current_segment:
                combined_results.append(current_segment)
            current_segment = {
                'speaker': 'Unknown',
                'text': trans['text'],
                'start': trans['start'],
                'end': trans['end']
            }
            current_speaker = 'Unknown'

    # Add the last segment
    if current_segment:
        combined_results.append(current_segment)

    return combined_results