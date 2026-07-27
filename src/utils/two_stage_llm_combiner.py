import logging
from typing import List, Dict

from .groq_api_helper import get_default_model
from .groq_llm_combiner import GroqLLMCombiner
from .semantic_flow_combiner import SemanticFlowCombiner

logger = logging.getLogger(__name__)

class TwoStageLLMCombiner:
    def __init__(self, model=None):
        self.model = model or get_default_model()
        self.semantic_flow = SemanticFlowCombiner()
        self.groq_combiner = GroqLLMCombiner(self.model)

    # Rate limiting lives in groq_api_helper.groq_api_call, which is where the
    # requests are actually issued.
    def combine(self, transcription: List[Dict], diarization: List[Dict]) -> List[Dict]:
        # Stage 1: Apply Semantic Flow
        semantic_flow_output = self.semantic_flow.combine(transcription, diarization)
        logger.info(f"Semantic Flow produced {len(semantic_flow_output)} segments")
        
        if not semantic_flow_output:
            logger.warning("Semantic Flow produced an empty result, skipping Groq LLM stage.")
            return []

        # Stage 2: Use Groq LLM Combiner for final speaker assignment
        final_output = self.groq_combiner.combine(semantic_flow_output)
        logger.info(f"Groq LLM Combiner produced {len(final_output)} segments")

        return final_output

def combine(transcription: List[Dict], diarization: List[Dict]) -> List[Dict]:
    two_stage_combiner = TwoStageLLMCombiner()
    return two_stage_combiner.combine(transcription, diarization)