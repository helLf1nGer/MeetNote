import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

class AdaptiveSemanticCombiner:
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2', initial_similarity_threshold=0.7, initial_gap_threshold=1.0):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = initial_similarity_threshold
        self.gap_threshold = initial_gap_threshold
        logger.info(f"Initialized AdaptiveSemanticCombiner with model: {model_name}")
        logger.info(f"Initial thresholds - Similarity: {self.similarity_threshold:.2f}, Gap: {self.gap_threshold:.2f}")

    def semantic_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        logger.debug(f"Semantic similarity between segments: {similarity:.4f}")
        return similarity

    def analyze_transcript(self, transcription):
        logger.info("Analyzing transcript for initial threshold setting")
        similarities = []
        gaps = []
        for i in range(len(transcription) - 1):
            similarities.append(self.semantic_similarity(transcription[i]['text'], transcription[i+1]['text']))
            gaps.append(transcription[i+1]['start'] - transcription[i]['end'])

        self.similarity_threshold = np.mean(similarities) - 0.5 * np.std(similarities)
        self.gap_threshold = np.mean(gaps) + np.std(gaps)

        logger.info(f"Initial adaptive thresholds set - Similarity: {self.similarity_threshold:.2f}, Gap: {self.gap_threshold:.2f}")

    def update_thresholds(self, recent_similarities, recent_gaps):
        old_sim = self.similarity_threshold
        old_gap = self.gap_threshold
        
        if recent_similarities:
            self.similarity_threshold = 0.7 * self.similarity_threshold + 0.3 * np.mean(recent_similarities)
        if recent_gaps:
            self.gap_threshold = 0.7 * self.gap_threshold + 0.3 * np.mean(recent_gaps)
        
        logger.debug(f"Updated thresholds - Similarity: {old_sim:.2f} -> {self.similarity_threshold:.2f}, "
                     f"Gap: {old_gap:.2f} -> {self.gap_threshold:.2f}")

    def combine(self, transcription, diarization):
        logger.info("Starting adaptive semantic combination process")
        self.analyze_transcript(transcription)

        combined_results = []
        current_segment = None
        recent_similarities = []
        recent_gaps = []

        for i, trans in enumerate(transcription):
            max_score = 0
            best_dia = None
            for dia in diarization:
                if (dia['start'] <= trans['start'] < dia['end']) or (dia['start'] < trans['end'] <= dia['end']):
                    score = self.semantic_similarity(trans['text'], dia['text'] if 'text' in dia else '')
                    if score > max_score:
                        max_score = score
                        best_dia = dia

            if best_dia:
                if current_segment and best_dia['speaker'] == current_segment['speaker']:
                    time_gap = trans['start'] - current_segment['end']
                    similarity = self.semantic_similarity(current_segment['text'], trans['text'])

                    recent_similarities.append(similarity)
                    recent_gaps.append(time_gap)

                    if len(recent_similarities) > 5:
                        recent_similarities.pop(0)
                        recent_gaps.pop(0)

                    self.update_thresholds(recent_similarities, recent_gaps)

                    if time_gap < self.gap_threshold and similarity > self.similarity_threshold:
                        logger.debug(f"Extending current segment. Gap: {time_gap:.2f}, Similarity: {similarity:.2f}")
                        current_segment['end'] = trans['end']
                        current_segment['text'] += ' ' + trans['text']
                    else:
                        logger.debug(f"Starting new segment. Gap: {time_gap:.2f}, Similarity: {similarity:.2f}")
                        combined_results.append(current_segment)
                        current_segment = {
                            'speaker': best_dia['speaker'],
                            'text': trans['text'],
                            'start': trans['start'],
                            'end': trans['end']
                        }
                else:
                    if current_segment:
                        combined_results.append(current_segment)
                    current_segment = {
                        'speaker': best_dia['speaker'],
                        'text': trans['text'],
                        'start': trans['start'],
                        'end': trans['end']
                    }
            else:
                if current_segment:
                    combined_results.append(current_segment)
                current_segment = {
                    'speaker': 'Unknown',
                    'text': trans['text'],
                    'start': trans['start'],
                    'end': trans['end']
                }

        if current_segment:
            combined_results.append(current_segment)

        logger.info(f"Adaptive semantic combination completed. Total segments: {len(combined_results)}")
        return combined_results

def combine(transcription, diarization):
    combiner = AdaptiveSemanticCombiner()
    return combiner.combine(transcription, diarization)