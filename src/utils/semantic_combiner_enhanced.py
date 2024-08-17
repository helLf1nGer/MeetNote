import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ImprovedEnhancedSemanticCombiner:
    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=0.2, gap_threshold=0.1, short_utterance_threshold=1):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.gap_threshold = gap_threshold
        self.short_utterance_threshold = short_utterance_threshold
        self.speaker_mapping = {}
        logger.info(f"Initialized ImprovedEnhancedSemanticCombiner with model: {model_name}")

    def semantic_similarity(self, text1, text2):
        embeddings = self.model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def combine(self, transcription, diarization):
        logger.info("Starting improved enhanced semantic combination process")
        
        combined_results = []
        current_segment = None
        previous_segment = None
        
        # Initialize speaker mapping
        self.initialize_speaker_mapping(diarization)

        for trans in transcription:
            best_speaker = self.get_best_speaker(trans['start'], trans['end'], diarization)
            
            if not current_segment or self.should_start_new_segment(trans, current_segment, best_speaker):
                if current_segment:
                    self.finalize_segment(current_segment, combined_results)
                
                current_segment = self.create_new_segment(trans, best_speaker)
            else:
                if self.is_likely_response(trans, previous_segment):
                    best_speaker = self.get_response_speaker(previous_segment['speaker'])
                
                if best_speaker == current_segment['speaker']:
                    current_segment = self.extend_segment(current_segment, trans)
                else:
                    self.finalize_segment(current_segment, combined_results)
                    current_segment = self.create_new_segment(trans, best_speaker)
            
            previous_segment = current_segment

        if current_segment:
            self.finalize_segment(current_segment, combined_results)

        logger.info("Improved enhanced semantic combination completed")
        return combined_results

    def initialize_speaker_mapping(self, diarization):
        speakers = sorted(set(d['speaker'] for d in diarization))
        self.speaker_mapping = {s: f"SPEAKER_{i:02d}" for i, s in enumerate(speakers)}

    def get_best_speaker(self, start, end, diarization):
        overlapping_segments = [d for d in diarization if d['start'] < end and d['end'] > start]
        if not overlapping_segments:
            return "Unknown"

        best_speaker = max(overlapping_segments, key=lambda d: min(d['end'], end) - max(d['start'], start))
        return self.speaker_mapping.get(best_speaker['speaker'], "Unknown")

    def should_start_new_segment(self, trans, current_segment, best_speaker):
        if not current_segment:
            return True
        if best_speaker != current_segment['speaker']:
            return True
        if trans['start'] - current_segment['end'] > self.gap_threshold:
            return True
        if self.semantic_similarity(current_segment['text'], trans['text']) < self.similarity_threshold:
            return True
        return False

    def is_likely_response(self, trans, previous_segment):
        if not previous_segment:
            return False
        if trans['start'] - previous_segment['end'] > self.gap_threshold:
            return False
        if len(trans['text'].split()) <= self.short_utterance_threshold:
            return True
        return False

    def get_response_speaker(self, previous_speaker):
        speakers = list(self.speaker_mapping.values())
        return speakers[(speakers.index(previous_speaker) + 1) % len(speakers)]

    def create_new_segment(self, trans, speaker):
        return {
            'speaker': speaker,
            'text': trans['text'],
            'start': trans['start'],
            'end': trans['end']
        }

    def extend_segment(self, current_segment, trans):
        return {
            'speaker': current_segment['speaker'],
            'text': f"{current_segment['text']} {trans['text']}",
            'start': current_segment['start'],
            'end': trans['end']
        }

    def finalize_segment(self, segment, combined_results):
        combined_results.append(segment)

def combine(transcription, diarization):
    combiner = ImprovedEnhancedSemanticCombiner()
    return combiner.combine(transcription, diarization)