import logging

logger = logging.getLogger(__name__)

def segment_score(transcript_segment: dict, diarization_segment: dict) -> float:
    duration = transcript_segment['end'] - transcript_segment['start']
    if duration <= 0:
        return 0.0

    overlap = min(transcript_segment['end'], diarization_segment['end']) - max(transcript_segment['start'], diarization_segment['start'])
    overlap_ratio = overlap / duration
    return overlap_ratio

def combine(transcription, diarization):
    combined_results = []

    for trans in transcription:
        max_score = 0
        best_dia = None
        for dia in diarization:
            score = segment_score(trans, dia)
            if score > max_score:
                max_score = score
                best_dia = dia

        # A segment with no overlapping speaker turn still has to be emitted.
        # Skipping it silently deleted transcribed speech from the output.
        combined_results.append({
            'speaker': best_dia['speaker'] if best_dia else 'Unknown',
            'text': trans['text'],
            'start': trans['start'],
            'end': trans['end']
        })

    unlabelled = sum(1 for s in combined_results if s['speaker'] == 'Unknown')
    if unlabelled:
        logger.warning("[Combiner] %d of %d segments had no overlapping speaker turn.",
                       unlabelled, len(combined_results))

    return combined_results