import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SemanticCombiner:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.7, gap_threshold: float = 1.0):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.gap_threshold = gap_threshold
        self._embedding_cache = {}

    def get_embedding(self, text: str) -> "np.ndarray":
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.model.encode(text)
        return self._embedding_cache[text]

    def segment_score(self, transcript_segment, diarization_segment):
        overlap_start = max(transcript_segment['start'], diarization_segment['start'])
        overlap_end = min(transcript_segment['end'], diarization_segment['end'])
        overlap = max(0, overlap_end - overlap_start)
        
        transcript_duration = transcript_segment['end'] - transcript_segment['start']
        diarization_duration = diarization_segment['end'] - diarization_segment['start']
        
        overlap_ratio = overlap / transcript_duration
        coverage_ratio = overlap / diarization_duration
        
        return (overlap_ratio + coverage_ratio) / 2

    def semantic_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return cosine_similarity([emb1], [emb2])[0][0]

    def combine(self, transcription, diarization):
        combined_results = []
        current_speaker = None
        current_segment = None

        for i, trans in enumerate(transcription):
            max_score = 0
            best_dia = None
            for dia in diarization:
                score = self.segment_score(trans, dia)
                if score > max_score:
                    max_score = score
                    best_dia = dia

            if best_dia:
                if best_dia['speaker'] == current_speaker and trans['start'] - current_segment['end'] < self.gap_threshold:
                    # Check semantic similarity before extending the segment
                    similarity = self.semantic_similarity(current_segment['text'], trans['text'])
                    if similarity > self.similarity_threshold:
                        # Extend the current segment
                        current_segment['end'] = trans['end']
                        current_segment['text'] += ' ' + trans['text']
                    else:
                        # Start a new segment due to semantic discontinuity
                        combined_results.append(current_segment)
                        current_segment = {
                            'speaker': best_dia['speaker'],
                            'text': trans['text'],
                            'start': trans['start'],
                            'end': trans['end']
                        }
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
    combiner = SemanticCombiner()
    return combiner.combine(transcription, diarization)