import logging
from . import simple_combiner
from . import weighted_combiner
from . import adaptive_combiner
from . import adaptive_rule_combiner
from . import semantic_combiner
from . import semantic_combiner_adaptive
from . import semantic_combiner_enhanced
from . import local_llama_tiny_combiner

logger = logging.getLogger(__name__)

def combine_transcription_diarization(transcription, diarization, pipeline_model, method='semantic'):
    logger.info(f"Combining transcription and diarization results using {method} method...")
    try:
        if method == 'simple':
            return simple_combiner.combine(transcription, diarization)
        elif method == 'weighted':
            return weighted_combiner.combine(transcription, diarization)
        elif method == 'adaptive':
            return adaptive_combiner.combine(transcription, diarization)
        elif method == 'adaptive_rule':
            return adaptive_rule_combiner.combine(transcription, diarization)
        elif method == 'semantic':
            return semantic_combiner.combine(transcription, diarization)
        elif method == 'semantic_adaptive':
            return semantic_combiner_adaptive.combine(transcription, diarization)
        elif method == 'semantic_enhanced':
            return semantic_combiner_enhanced.combine(transcription, diarization)
        elif method == 'local_llama_tiny':
            return local_llama_tiny_combiner.combine(transcription, diarization)
        else:
            raise ValueError(f"Unknown combination method: {method}")
    except Exception as e:
        logger.error(f"Error during combination of transcription and diarization: {str(e)}")
        raise