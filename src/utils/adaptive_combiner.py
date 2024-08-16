import logging
import numpy as np

logger = logging.getLogger(__name__)

class AdaptiveCombiner:
    def __init__(self, initial_overlap_threshold=0.5, initial_gap_threshold=1.0):
        self.overlap_threshold = initial_overlap_threshold
        self.gap_threshold = initial_gap_threshold

    def segment_score(self, transcript_segment, diarization_segment):
        overlap_start = max(transcript_segment['start'], diarization_segment['start'])
        overlap_end = min(transcript_segment['end'], diarization_segment['end'])
        overlap = max(0, overlap_end - overlap_start)
        
        transcript_duration = transcript_segment['end'] - transcript_segment['start']
        diarization_duration = diarization_segment['end'] - diarization_segment['start']
        
        overlap_ratio = overlap / transcript_duration
        coverage_ratio = overlap / diarization_duration
        
        return (overlap_ratio + coverage_ratio) / 2

    def adapt_thresholds(self, transcription, diarization):
        overlap_ratios = []
        gaps = []
        
        for trans in transcription:
            for dia in diarization:
                score = self.segment_score(trans, dia)
                overlap_ratios.append(score)
                
                gap = abs(trans['start'] - dia['start'])
                gaps.append(gap)
        
        self.overlap_threshold = np.mean(overlap_ratios) - 0.5 * np.std(overlap_ratios)
        self.gap_threshold = np.mean(gaps) + np.std(gaps)
        
        logger.info(f"Adapted thresholds - Overlap: {self.overlap_threshold:.2f}, Gap: {self.gap_threshold:.2f}")

    def combine(self, transcription, diarization):
        self.adapt_thresholds(transcription, diarization)
        
        combined_results = []
        current_speaker = None
        current_segment = None

        for trans in transcription:
            max_score = 0
            best_dia = None
            for dia in diarization:
                score = self.segment_score(trans, dia)
                if score > max_score and score > self.overlap_threshold:
                    max_score = score
                    best_dia = dia

            if best_dia:
                if best_dia['speaker'] == current_speaker and trans['start'] - current_segment['end'] < self.gap_threshold:
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

def combine(transcription, diarization):
    combiner = AdaptiveCombiner()
    return combiner.combine(transcription, diarization)