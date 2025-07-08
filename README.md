# MeetNote: Audio Transcription and Diarization Tool

## Overview

MeetNote is a practical audio transcription and diarization tool designed to streamline the process of converting spoken content into text with speaker attribution. It combines efficient speech recognition with speaker diarization to produce accurate, speaker-attributed transcripts of audio recordings.

Key features:
- Transcription using Whisper model (local) or Groq Cloud API
- Speaker diarization using PyAnnote
- Intelligent combination of transcription and diarization results
- User-friendly GUI for easy operation
- Support for various audio and video formats
- Customizable output in PDF format

MeetNote is suitable for transcribing meetings, interviews, podcasts, and any multi-speaker audio content.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Configuration](#configuration)
4. [Common Issues](#common-issues)
5. [Advanced Features](#advanced-features)
6. [Contributing](#contributing)
7. [License](#license)
8. [Author](#author)
9. [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)
- Groq API account (for cloud-based transcription)
- Hugging Face account (for PyAnnote models)

### Step-by-step Installation

1. Clone the repository:
   ```
   git clone https://github.com/helLf1nGer/meetnote.git
   cd meetnote
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   
   Note: This includes all dependencies including sentence-transformers and scikit-learn for advanced combiner methods.

4. Install PyTorch with CUDA support (if using GPU):
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

5. Install and set up pyannote.audio:
   ```
   pip install pyannote.audio
   ```
   Then, follow these steps:
   - Go to https://huggingface.co/pyannote/segmentation-3.0 and accept the user conditions
   - Go to https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the user conditions
   - Create an access token at https://hf.co/settings/tokens

6. Set up environment variables:
- Create a `.env` file in the project root
   - Add the following lines:
     ```
     HUGGING_FACE_AUTH_TOKEN=your_huggingface_token
     GROQ_API_KEY=your_groq_api_key
     ```
   - To obtain a Groq API key:
     a. Go to [https://console.groq.com/](https://console.groq.com/) and sign up for an account
     b. Once logged in, navigate to the API Keys section
     c. Create a new API key and copy it
     d. Paste the key as the value for `GROQ_API_KEY` in your `.env` file

### Package-specific Notes

- **PyAnnote**: Requires a Hugging Face account and token for model access.
- **Whisper**: Large models may require significant disk space and RAM.
- **CUDA**: Ensure your NVIDIA drivers are up-to-date for GPU acceleration.

## Usage

1. Run the application:
   ```
   python run.py
   ```

2. Use the GUI to:
   - Select an audio/video file
   - Choose transcription method (local Whisper or Groq Cloud)
   - Set the number of speakers
   - Select diarization model
   - Choose output directory

3. Click "Start Processing" to begin transcription and diarization.

4. Once complete, find the PDF output in your specified directory.

## Configuration

- Edit `Config/config.json` to change default settings.
- Key configurations:
  - `use_cuda`: Enable/disable GPU acceleration
  - `model_options`: Choose Whisper model size for local transcription
  - `diarization`: Adjust speaker detection parameters
  - `transcription.method`: Set to "groq" to use Groq API or "local" for Whisper model
  - `combiner`: Configure the combining strategy:
    ```json
    "combiner": {
        "method": "semantic_adaptive",  // Default combiner method
        "model": "llama3-groq-70b-8192-tool-use-preview"  // Used by LLM-based combiners
    }
    ```
    Available combiner methods:
    - semantic_adaptive: Dynamically adjusts thresholds (default)
    - semantic_flow: Best balance of accuracy and performance
    - semantic: Basic semantic similarity-based combining
    - semantic_enhanced: Finer-grained segment analysis
    - simple: Basic time-based combining
    - weighted: Enhanced time-based combining
    - adaptive: Self-adjusting thresholds
    - adaptive_rule: Rule-based adaptation
    - groq_llm: Uses Groq API for LLM-based combining
    - two_stage_llm: Combines semantic and LLM approaches
    - local_llama_tiny: Uses local LLaMa model

Note: Ensure your Groq API key is correctly set in the `.env` file when using the Groq transcription method or LLM-based combiners.

## Verifying Your Setup

After installation, you can verify that everything is set up correctly by running the test script from the project root directory:

```
python test_setup.py
```

This script will check:
- Environment variables are set correctly
- Config file can be loaded
- GUI can be initialized
- Audio processing libraries are working
- Connection to Groq API is successful
- Connection to HuggingFace is successful
- Whisper model can be loaded

If all tests pass, you should see checkmarks (✅) for each component. If any test fails, you'll see a cross (❌) with an error message. Address any issues before using the application.

Note: Make sure you have an active internet connection when running the tests, as some checks require connecting to external services.

**Future Updates:** In later versions of MeetNote, the location of this script may change to improve project organization. Always refer to the most recent documentation for the correct way to run setup verification.

## Common Issues

1. **CUDA out of memory**: 
   - Solution: Use a smaller Whisper model or process on CPU
   - Edit `config.json`: Set `"device": "cpu"` under `model_options.local`

2. **PyAnnote authentication error**: 
   - Ensure your Hugging Face token is correct in the `.env` file
   - Check your token permissions on the Hugging Face website

3. **Groq API errors**:
   - Verify your Groq API key in the `.env` file
   - Check your Groq account status and quotas

4. **Unsupported audio/video format**:
   - Install additional codecs or convert your file to a supported format (e.g., MP3, WAV, MP4)

5. **PDF generation fails**:
   - Ensure you have write permissions in the output directory
   - Check if a custom font is properly installed in the `Fonts` directory

## Advanced Features

### Multiple Combiner Methods

MeetNote includes various combiner methods for merging transcription and diarization results:

- **Semantic Adaptive Combiner**: Dynamically adjusts thresholds (default)
- **Semantic Flow Combiner**: Best balance of accuracy and performance
- **Semantic Combiner**: Basic semantic similarity approach
- **Semantic Enhanced Combiner**: Finer-grained segment analysis
- **Simple Combiner**: Basic time-based combining
- **Weighted Combiner**: Enhanced time-based combining
- **Adaptive Combiner**: Self-adjusting thresholds
- **Adaptive Rule Combiner**: Rule-based adaptation
- **Groq LLM Combiner**: Advanced language model processing
- **Two-Stage LLM Combiner**: Hybrid semantic-LLM approach
- **Local LLaMa Tiny Combiner**: Offline processing option

### Development Tools

- `dev_main.py`: Development version for testing combiners
- `combiner_testing.py`: Utility for comparing different combiner methods
- Built-in rate limiter for API calls
- Enhanced GUI for easier testing and comparison

## Contributing

We welcome contributions to MeetNote! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request

Please ensure your code adheres to our coding standards and include tests for new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For more detailed information, please check our [documentation](Docs) or open an [Issue](https://github.com/helLf1nGer/meetnote/issues) if you encounter any problems.

---

## Author

**Ivan Bondarenko** - [helLf1nGer](https://github.com/helLf1nGer)

## Acknowledgments

- OpenAI Whisper team for the transcription models
- PyAnnote team for the diarization framework
- Groq for providing fast inference API
- All contributors who have helped improve this project