import logging

logger = logging.getLogger(__name__)

def segment_score(transcript_segment, diarization_segment):
    overlap = min(transcript_segment['end'], diarization_segment['end']) - max(transcript_segment['start'], diarization_segment['start'])
    overlap_ratio = overlap / (transcript_segment['end'] - transcript_segment['start'])
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

        if best_dia:
            combined_results.append({
                'speaker': best_dia['speaker'],
                'text': trans['text'],
                'start': trans['start'],
                'end': trans['end']
            })

    return combined_results