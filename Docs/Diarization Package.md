# Diarization Package

## Overview

The diarization package provides functionality for speaker diarization, which is the process of partitioning an audio stream into homogeneous segments according to the speaker identity. This package uses the pyannote.audio library for performing diarization.

## Module: diarizer.py

This module contains the main function for performing audio diarization.

### Functions

#### diarize_audio(pipeline: Pipeline, file_path: str, n_speakers: int) -> Tuple[List[Dict], str]

Perform speaker diarization on an audio file.

##### Parameters:
- `pipeline` (pyannote.audio.Pipeline): The pyannote.audio pipeline object for diarization.
- `file_path` (str): Path to the audio file to be diarized.
- `n_speakers` (int): The number of speakers expected in the audio.

##### Returns:
- Tuple[List[Dict], str]: A tuple containing:
  - List[Dict]: A list of diarization results. Each dictionary in the list contains:
    - 'start' (float): Start time of the speech segment.
    - 'end' (float): End time of the speech segment.
    - 'speaker' (str): Label of the speaker for this segment.
  - str: The device used for diarization ('cuda' or 'cpu').

##### Raises:
- Exception: If an error occurs during the diarization process.

##### Behavior:
1. Loads the audio file using torchaudio.
2. Determines whether to use CUDA or CPU based on availability and configuration.
3. Performs diarization using the provided pipeline.
4. Returns the diarization results and the device used.

### Usage Example

```python
from pyannote.audio import Pipeline
from diarization.diarizer import diarize_audio

# Initialize the pipeline (you need to have the appropriate model and authentication)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="YOUR_AUTH_TOKEN")

# Perform diarization
file_path = "path/to/your/audio/file.wav"
n_speakers = 2  # Expected number of speakers
diarization_results, used_device = diarize_audio(pipeline, file_path, n_speakers)

# Print results
for segment in diarization_results:
    print(f"Speaker {segment['speaker']}: {segment['start']:.2f}s - {segment['end']:.2f}s")
print(f"Diarization performed on: {used_device}")
```

### Dependencies

- torch
- torchaudio
- pyannote.audio
- logging

Make sure to install these dependencies and have the appropriate authentication for pyannote.audio before using this module.

### Configuration

The module uses a configuration manager to determine settings such as whether to use CUDA. Ensure that your configuration is properly set up before using this module.

### Note on Performance

The diarization process can be computationally intensive. When possible, it will use CUDA for improved performance. The module will log information about the device used and memory allocation when using CUDA.