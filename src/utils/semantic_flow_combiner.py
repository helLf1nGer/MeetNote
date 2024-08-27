import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)

class SemanticFlowCombiner:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 time_window: float = 1.0,
                 semantic_threshold: float = 0.5):
        self.model = SentenceTransformer(model_name)
        self.time_window = time_window
        self.semantic_threshold = semantic_threshold

    def semantic_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def combine(self, transcription: List[Dict], diarization: List[Dict]) -> List[Dict]:
        logger.info("Starting Semantic Flow combination process")
        
        # Step 1: Assign initial speakers based on diarization
        assigned_segments = self.assign_initial_speakers(transcription, diarization)
        
        # Step 2: Refine speaker assignments based on semantic flow
        refined_segments = self.refine_speaker_assignments(assigned_segments)

        logger.info(f"Semantic Flow combination completed. Total segments: {len(refined_segments)}")
        return refined_segments

    def assign_initial_speakers(self, transcription: List[Dict], diarization: List[Dict]) -> List[Dict]:
        assigned_segments = []
        for trans in transcription:
            overlapping_dia = [d for d in diarization if self.segments_overlap(trans, d)]
            if overlapping_dia:
                # If there are overlapping diarization segments, choose the one with the most overlap
                best_dia = max(overlapping_dia, key=lambda d: self.overlap_duration(trans, d))
                assigned_speaker = best_dia['speaker']
            else:
                # If no overlap, assign the nearest speaker in time
                nearest_dia = min(diarization, key=lambda d: min(abs(d['start'] - trans['end']), abs(d['end'] - trans['start'])))
                assigned_speaker = nearest_dia['speaker']
            
            assigned_segments.append({**trans, 'speaker': assigned_speaker})

        return assigned_segments

    def refine_speaker_assignments(self, segments: List[Dict]) -> List[Dict]:
        refined_segments = []
        current_speaker = None
        current_context = ""

        for i, segment in enumerate(segments):
            if current_speaker is None:
                current_speaker = segment['speaker']
                current_context = segment['text']
                refined_segments.append(segment)
            else:
                similarity = self.semantic_similarity(current_context, segment['text'])
                time_gap = segment['start'] - refined_segments[-1]['end']

                if segment['speaker'] == current_speaker and similarity > self.semantic_threshold:
                    # Merge with the previous segment
                    refined_segments[-1]['end'] = segment['end']
                    refined_segments[-1]['text'] += ' ' + segment['text']
                    current_context = refined_segments[-1]['text']
                elif similarity <= self.semantic_threshold or time_gap > self.time_window:
                    # Start a new segment
                    refined_segments.append(segment)
                    current_speaker = segment['speaker']
                    current_context = segment['text']
                else:
                    # Keep the current speaker but add as a new segment
                    refined_segments.append({**segment, 'speaker': current_speaker})
                    current_context += ' ' + segment['text']

        return refined_segments

    def segments_overlap(self, seg1: Dict, seg2: Dict) -> bool:
        return (seg1['start'] <= seg2['start'] < seg1['end']) or \
               (seg2['start'] <= seg1['start'] < seg2['end'])

    def overlap_duration(self, seg1: Dict, seg2: Dict) -> float:
        return max(0, min(seg1['end'], seg2['end']) - max(seg1['start'], seg2['start']))

def combine(transcription: List[Dict], diarization: List[Dict]) -> List[Dict]:
    combiner = SemanticFlowCombiner()
    return combiner.combine(transcription, diarization)