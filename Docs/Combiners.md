# Utils Package: Combiner Documentation

## Overview

The utils package contains various combiner modules that are responsible for merging the results of transcription and diarization. These combiners use different strategies to align speaker segments with transcribed text, aiming to produce accurate and coherent final outputs.

## Types of Combiners

### 1. Semantic Combiner (semantic_combiner.py)

**Default Combiner**

The semantic combiner uses sentence embeddings to measure the semantic similarity between adjacent segments. This approach helps in maintaining context and reducing erroneous speaker changes.

Key features:
- Uses the SentenceTransformer library for generating embeddings
- Considers both temporal overlap and semantic similarity
- Adjustable similarity and gap thresholds

### 2. Adaptive Semantic Combiner (semantic_combiner_adaptive.py)

An extension of the semantic combiner that dynamically adjusts its thresholds based on the input data.

Key features:
- Analyzes the entire transcript to set initial thresholds
- Continuously updates thresholds during processing
- May offer improved performance on varied inputs

### 3. Enhanced Semantic Combiner (semantic_combiner_enhanced.py)

A more sophisticated version of the semantic combiner that splits longer segments into sub-segments for finer-grained analysis.

Key features:
- Splits Whisper segments into sentence-level sub-segments
- Assigns speakers to sub-segments individually
- May provide more accurate speaker transitions within long segments

### 4. Simple Combiner (simple_combiner.py)

A basic combiner that uses temporal overlap as the sole criterion for merging segments.

Key features:
- Lightweight and fast
- Suitable for simple audio files with clear speaker separation

### 5. Weighted Combiner (weighted_combiner.py)

Extends the simple combiner by introducing weights for different factors in the combining process.

Key features:
- Considers overlap ratio, coverage ratio, and center distance
- Allows fine-tuning of the combining process through weight adjustments

### 6. Adaptive Combiner (adaptive_combiner.py)

Attempts to adapt its thresholds based on the characteristics of the input data.

Key features:
- Analyzes the entire dataset to set initial thresholds
- May improve performance on varied inputs
- Currently experimental and may have unresolved issues

### 7. Adaptive Rule Combiner (adaptive_rule_combiner.py)

Similar to the adaptive combiner but with additional rules for segment merging.

Key features:
- Uses adaptive thresholds like the adaptive combiner
- Incorporates additional rules for decision making
- Currently experimental and may have unresolved issues

## Result Combiner (result_combiner.py)

The result combiner acts as a facade for all the individual combiners. It provides a unified interface to select and use different combining strategies.

Key features:
- Allows easy switching between different combiner types
- Provides a consistent interface for the main application
- Handles exceptions and logging for all combiner types

## Default Configuration

By default, the application uses the Semantic Combiner (semantic_combiner.py). This choice balances effectiveness and reliability. While adaptive combiners (adaptive_combiner.py and adaptive_rule_combiner.py) show potential for improved performance, they are currently experimental and may have unresolved issues.

## Usage

The result combiner is typically used in the main application flow:

```python
from utils.result_combiner import combine_transcription_diarization

final_transcription = combine_transcription_diarization(transcription, diarization, pipeline_model, method='semantic')
```

You can change the `method` parameter to use different combiners:
- 'semantic' (default)
- 'semantic_adaptive'
- 'semantic_enhanced'
- 'simple'
- 'weighted'
- 'adaptive'
- 'adaptive_rule'

## Future Development

While the semantic combiner is currently the most reliable option, ongoing research and development may improve the performance of adaptive combiners. Users are encouraged to experiment with different combiners for their specific use cases, keeping in mind that some options are still experimental.