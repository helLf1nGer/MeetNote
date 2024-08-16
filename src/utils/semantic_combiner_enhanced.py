import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

logger = logging.getLogger(__name__)

class EnhancedSemanticCombiner:
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2', similarity_threshold=0.7, gap_threshold=1.0):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.gap_threshold = gap_threshold
        logger.info(f"Initialized EnhancedSemanticCombiner with model: {model_name}")
        logger.info(f"Thresholds - Similarity: {self.similarity_threshold:.2f}, Gap: {self.gap_threshold:.2f}")

    def semantic_similarity(self, text1, text2):
        embeddings = self.model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def split_whisper_segment(self, segment):
        sentences = re.split('(?<=[.!?]) +', segment['text'])
        sub_segments = []
        total_length = len(segment['text'])
        current_position = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            sub_segment_duration = (sentence_length / total_length) * (segment['end'] - segment['start'])
            sub_segment = {
                'text': sentence,
                'start': segment['start'] + current_position * (segment['end'] - segment['start']) / total_length,
                'end': segment['start'] + (current_position + sentence_length) * (segment['end'] - segment['start']) / total_length
            }
            sub_segments.append(sub_segment)
            current_position += sentence_length

        return sub_segments

    def assign_speaker(self, sub_segment, diarization):
        for dia in diarization:
            if sub_segment['start'] >= dia['start'] and sub_segment['end'] <= dia['end']:
                return dia['speaker']
        return 'Unknown'

    def combine(self, transcription, diarization):
        logger.info("Starting enhanced semantic combination process")
        combined_results = []
        current_segment = None

        for trans in transcription:
            sub_segments = self.split_whisper_segment(trans)
            
            for sub_segment in sub_segments:
                sub_segment['speaker'] = self.assign_speaker(sub_segment, diarization)
                
                if current_segment and current_segment['speaker'] == sub_segment['speaker']:
                    time_gap = sub_segment['start'] - current_segment['end']
                    if time_gap < self.gap_threshold:
                        similarity = self.semantic_similarity(current_segment['text'], sub_segment['text'])
                        if similarity > self.similarity_threshold:
                            current_segment['end'] = sub_segment['end']
                            current_segment['text'] += ' ' + sub_segment['text']
                            continue

                if current_segment:
                    combined_results.append(current_segment)
                
                current_segment = sub_segment

        if current_segment:
            combined_results.append(current_segment)

        logger.info(f"Enhanced semantic combination completed. Total segments: {len(combined_results)}")
        return combined_results

def combine(transcription, diarization):
    logger.info("Starting enhanced transcription and diarization combination")
    combiner = EnhancedSemanticCombiner()
    result = combiner.combine(transcription, diarization)
    logger.info("Enhanced combination process completed")
    return result