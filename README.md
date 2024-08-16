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
5. [Contributing](#contributing)
6. [License](#license)

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

Note: Ensure your Groq API key is correctly set in the `.env` file when using the Groq transcription method.

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

## Contributing

We welcome contributions to MeetNote! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request

Please ensure your code adheres to our coding standards and include tests for new features.

## License

Copyright (c) 2024 Ivan Bondarenko

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

For more detailed information, please check our [documentation](Docs) or open an [Issue](https://github.com/helLf1nGer/meetnote/issues) if you encounter any problems.