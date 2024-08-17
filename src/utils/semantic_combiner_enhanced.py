import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

logger = logging.getLogger(__name__)

class EnhancedSemanticCombiner:
    def __init__(self, model_name='all-MiniLM-L6-v2', initial_similarity_threshold=0.7, initial_gap_threshold=1.0, short_utterance_threshold=5):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = initial_similarity_threshold
        self.gap_threshold = initial_gap_threshold
        self.short_utterance_threshold = short_utterance_threshold
        self.speaker_mapping = {}

    def semantic_similarity(self, text1, text2):
        embeddings = self.model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def split_whisper_segment(self, segment):
        sentences = re.split(r'(?<=[.!?])\s+', segment['text'])
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
        max_overlap = 0
        best_speaker = 'Unknown'
        for dia in diarization:
            overlap = min(sub_segment['end'], dia['end']) - max(sub_segment['start'], dia['start'])
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = dia['speaker']
        return self.speaker_mapping.get(best_speaker, 'Unknown')

    def initialize_speaker_mapping(self, diarization):
        speakers = sorted(set(d['speaker'] for d in diarization))
        self.speaker_mapping = {s: f"SPEAKER_{i:02d}" for i, s in enumerate(speakers)}

    def is_likely_response(self, sub_segment, previous_segment):
        if not previous_segment:
            return False
        if sub_segment['start'] - previous_segment['end'] > self.gap_threshold:
            return False
        if len(sub_segment['text'].split()) <= self.short_utterance_threshold:
            return True
        return False

    def get_response_speaker(self, previous_speaker):
        speakers = list(self.speaker_mapping.values())
        if previous_speaker == 'Unknown':
            return 'Unknown'
        return speakers[(speakers.index(previous_speaker) + 1) % len(speakers)]

    def combine(self, transcription, diarization):
        logger.info("Starting updated improved enhanced semantic combination process")
        self.initialize_speaker_mapping(diarization)
        
        combined_results = []
        current_segment = None
        previous_segment = None

        for trans in transcription:
            sub_segments = self.split_whisper_segment(trans)
            
            for sub_segment in sub_segments:
                sub_segment['speaker'] = self.assign_speaker(sub_segment, diarization)
                
                if self.is_likely_response(sub_segment, previous_segment):
                    sub_segment['speaker'] = self.get_response_speaker(previous_segment['speaker'])
                
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
                previous_segment = current_segment

        if current_segment:
            combined_results.append(current_segment)

        logger.info(f"Updated improved enhanced semantic combination completed. Total segments: {len(combined_results)}")
        return combined_results

def combine(transcription, diarization):
    combiner = EnhancedSemanticCombiner()
    return combiner.combine(transcription, diarization)