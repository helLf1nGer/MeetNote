# Utils Package: Combiner Documentation

## Overview

The utils package contains various combiner modules that are responsible for merging the results of transcription and diarization. These combiners use different strategies to align speaker segments with transcribed text, aiming to produce accurate and coherent final outputs.

## Types of Combiners

### 1. Semantic Flow Combiner (semantic_flow_combiner.py)

**Default Combiner**

The Semantic Flow Combiner focuses on the semantic flow of the conversation to determine speaker changes and segment boundaries. This is the default combiner as it provides the best balance of accuracy and performance.

Key features:
- Analyzes semantic continuity across segments using sentence embeddings
- Considers both temporal and semantic aspects for speaker assignment
- Handles complex conversational structures more effectively
- Uses a time window and semantic threshold for decision making

Usage example:

```python
from utils.semantic_flow_combiner import SemanticFlowCombiner
combiner = SemanticFlowCombiner()
combined_results = combiner.combine(transcription, diarization)
```


### 2. Semantic Combiner (semantic_combiner.py)

The Semantic Combiner uses sentence embeddings to measure the semantic similarity between adjacent segments. This approach helps in maintaining context and reducing erroneous speaker changes.

Key features:
- Uses the SentenceTransformer library for generating embeddings
- Considers both temporal overlap and semantic similarity
- Adjustable similarity and gap thresholds

Usage example:

```python
from utils.semantic_combiner import SemanticCombiner
combiner = SemanticCombiner()
combined_results = combiner.combine(transcription, diarization)
```


### 3. Adaptive Semantic Combiner (semantic_combiner_adaptive.py)

An extension of the semantic combiner that dynamically adjusts its thresholds based on the input data.

Key features:
- Analyzes the entire transcript to set initial thresholds
- Continuously updates thresholds during processing
- May offer improved performance on varied inputs

Usage example:

```python
from utils.semantic_combiner_adaptive import AdaptiveSemanticCombiner
combiner = AdaptiveSemanticCombiner()
combined_results = combiner.combine(transcription, diarization)
```


### 4. Enhanced Semantic Combiner (semantic_combiner_enhanced.py)

A more sophisticated version of the semantic combiner that splits longer segments into sub-segments for finer-grained analysis.

Key features:
- Splits Whisper segments into sentence-level sub-segments
- Assigns speakers to sub-segments individually
- May provide more accurate speaker transitions within long segments
- Handles short utterances and likely responses

Usage example:

```python
from utils.semantic_combiner_enhanced import EnhancedSemanticCombiner
combiner = EnhancedSemanticCombiner()
combined_results = combiner.combine(transcription, diarization)
```

### 5. Simple Combiner (simple_combiner.py)

A basic combiner that uses temporal overlap as the sole criterion for merging segments.

Key features:
- Lightweight and fast
- Suitable for simple audio files with clear speaker separation

Usage example:

```python
from utils.simple_combiner import SimpleCombiner
combiner = SimpleCombiner()
combined_results = combiner.combine(transcription, diarization)
```


### 6. Weighted Combiner (weighted_combiner.py)

Extends the simple combiner by introducing weights for different factors in the combining process.

Key features:
- Considers overlap ratio, coverage ratio, and center distance
- Allows fine-tuning of the combining process through weight adjustments

Usage example:

```python
from utils.weighted_combiner import WeightedCombiner
combiner = WeightedCombiner()
combined_results = combiner.combine(transcription, diarization)
```


### 7. Adaptive Combiner (adaptive_combiner.py)

Attempts to adapt its thresholds based on the characteristics of the input data.

Key features:
- Analyzes the entire dataset to set initial thresholds
- May improve performance on varied inputs

Usage example:

```python
from utils.adaptive_combiner import AdaptiveCombiner
combiner = AdaptiveCombiner()
combined_results = combiner.combine(transcription, diarization)
```

### 8. Adaptive Rule Combiner (adaptive_rule_combiner.py)

Similar to the adaptive combiner but with additional rules for segment merging.

Key features:
- Uses adaptive thresholds like the adaptive combiner
- Incorporates additional rules for decision making

Usage example:

```python
from utils.adaptive_rule_combiner import AdaptiveRuleCombiner
combiner = AdaptiveRuleCombiner()
combined_results = combiner.combine(transcription, diarization)
```


### 9. Groq LLM Combiner (groq_llm_combiner.py)

Utilizes the Groq API to leverage large language models for combining transcription and diarization results.

Key features:
- Uses advanced language models for context understanding
- Handles long inputs by splitting into manageable chunks
- Includes rate limiting to prevent API overuse

Usage example:

```python
from utils.groq_llm_combiner import GroqLLMCombiner
combiner = GroqLLMCombiner()
combined_results = combiner.combine(transcription, diarization)
```


### 10. Two-Stage LLM Combiner (two_stage_llm_combiner.py)

A two-stage approach that first uses a semantic flow combiner and then refines the results using a large language model.

Key features:
- Combines semantic analysis with LLM-based refinement
- Handles problematic chunks separately for improved accuracy
- Uses Groq API for LLM processing

Usage example:

```python
from utils.two_stage_llm_combiner import TwoStageLLMCombiner
combiner = TwoStageLLMCombiner()
combined_results = combiner.combine(transcription, diarization)
```


### 11. Local LLaMa Tiny Combiner (local_llama_tiny_combiner.py)

Uses a local LLaMa model for combining results without relying on external APIs.

Key features:
- Offline processing capability
- Suitable for environments without internet access or with data privacy concerns
- Uses a smaller model for faster processing

Usage example:

```python
from utils.local_llama_tiny_combiner import LocalLlamaTinyCombiner
combiner = LocalLlamaTinyCombiner()
combined_results = combiner.combine(transcription, diarization)
```

## Result Combiner (result_combiner.py)

The result combiner acts as a facade for all the individual combiners. It provides a unified interface to select and use different combining strategies.

Key features:
- Allows easy switching between different combiner types
- Provides a consistent interface for the main application
- Handles exceptions and logging for all combiner types

## Default Configuration

The application now uses the Semantic Flow Combiner (semantic_flow_combiner.py) by default. This choice provides the best balance of accuracy, reliability, and performance. The combiner configuration is stored in the config.json file under the "combiner" section:

```json
{
    "combiner": {
        "method": "semantic_flow",
        "model": "llama3-groq-70b-8192-tool-use-preview"
    }
}
```

The model parameter is only used for LLM-based combiners (two_stage_llm and groq_llm).

## Usage

The result combiner is typically used in the main application flow:

```python
from utils.result_combiner import combine_transcription_diarization

final_transcription = combine_transcription_diarization(transcription, diarization, pipeline_model)
```

You can change the combiner method in the config.json file or through the GUI. Available methods:
- 'semantic_flow' (default)
- 'semantic'
- 'semantic_adaptive'
- 'semantic_enhanced'
- 'simple'
- 'weighted'
- 'adaptive'
- 'adaptive_rule'
- 'groq_llm'
- 'two_stage_llm'
- 'local_llama_tiny'

## Development and Testing

For development and testing of combiners, use the `combiner_testing.py` module. This utility allows for easy comparison and evaluation of different combiner methods.

Example usage:

```python
from utils.combiner_testing import test_combiners

test_combiners(transcription, diarization, pipeline_model, output_directory='tests')
```


This will run all available combiners and provide comparative results and visualizations. The results include:
- JSON and PDF outputs for each combiner
- A comprehensive CSV with all combiner results
- Comparison metrics (number of segments, average segment duration, number of speaker changes, total duration)
- Visualization of speaker segments for each combiner

## Configuration

Many combiners have configurable parameters. These can be adjusted in the respective combiner files or through the configuration system. Refer to the individual combiner documentation for specific configuration options.

## Future Development

Ongoing research and development may improve the performance of various combiners. Users are encouraged to experiment with different combiners for their specific use cases. The modular design of the combiner system allows for easy integration of new combining strategies as they are developed.

For more detailed information on each combiner, including their algorithms, performance characteristics, and best use cases, please refer to their individual documentation files in the `Docs/` directory.