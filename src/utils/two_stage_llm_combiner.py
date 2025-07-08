import logging
from .groq_api_helper import groq_api_call, count_tokens
from .rate_limiter import RateLimiter
from typing import List, Dict
from .semantic_flow_combiner import SemanticFlowCombiner
from .groq_llm_combiner import GroqLLMCombiner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwoStageLLMCombiner:
    def __init__(self, model="llama3-groq-70b-8192-tool-use-preview"):
        self.model = model
        self.semantic_flow = SemanticFlowCombiner()
        self.groq_combiner = GroqLLMCombiner(model)

    @RateLimiter(max_calls=30, period=60)
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