# Transcription Package Documentation

## Overview

The transcription package provides functionality for transcribing audio files. Its main feature is the flexibility to choose between two transcription methods:

1. Using Groq's cloud-based API
2. Using a local Whisper model

This design allows users to balance between cloud-based processing power and local resource utilization based on their needs and constraints.

## Key Components

### 1. Groq Transcription

- Function: `transcribe_audio_with_groq(file_path)`
- Purpose: Transcribes audio using Groq's cloud API
- Features:
  - Handles large files by splitting them into chunks
  - Utilizes Groq's high-performance cloud infrastructure

### 2. Local Whisper Transcription

- Functions: `create_local_model(config)` and `transcribe_audio(model, file_path)`
- Purpose: Creates and uses a local Whisper model for transcription
- Features:
  - Supports different Whisper model sizes (e.g., 'medium.en', 'large-v3')
  - Can utilize GPU acceleration if available

### 3. Fallback Mechanism

- Function: `transcribe_with_fallback(file_path)`
- Purpose: Attempts Groq transcription first, falls back to local if Groq fails

## Usage

Users can specify their preferred transcription method when initiating the transcription process. This is typically done through the GUI or command-line interface.

### Groq Transcription

```python
transcription_result = transcribe_audio_with_groq(file_path)
```

### Local Whisper Transcription

```python
model, _ = create_local_model(config)
transcription_result = transcribe_audio(model, file_path)
```

## Configuration

The transcription methods are configured through the `config.json` file, managed by the `ConfigManager`. Key configurations include:

- Groq API model selection
- Local Whisper model selection
- CUDA usage for local transcription
- Language and task settings

## Performance Considerations

- Groq: Offers high-speed transcription but requires internet connectivity and may incur API costs
- Local Whisper: Provides offline capability but may be slower, especially without GPU acceleration

## Error Handling

Both methods include robust error handling:
- Groq transcription errors are caught and logged
- Local transcription falls back to CPU if CUDA initialization fails

## Integration with Main Application

The transcription package is integrated into the main application flow:
1. User selects transcription method via GUI
2. Main process initiates transcription based on user choice
3. Results are passed to the diarization and combination stages

## Extensibility

The modular design allows for easy addition of new transcription methods or updates to existing ones. Future improvements could include:
- Support for more cloud-based transcription services
- Integration of newer Whisper models as they become available

## Logging

Comprehensive logging is implemented throughout the transcription process, providing detailed information for troubleshooting and performance analysis.