# MeetNote: Audio Transcription and Diarization Tool

## Overview

MeetNote is a practical audio transcription and diarization tool designed to streamline the process of converting spoken content into text with speaker attribution. It combines efficient speech recognition with speaker diarization to produce accurate, speaker-attributed transcripts of audio recordings.

Key features:
- Transcription using Whisper model (local) or Groq Cloud API
- Speaker diarization using PyAnnote
- Intelligent combination of transcription and diarization results
- User-friendly GUI for easy operation
- Command-line mode for batch and scripted use
- Multi-language transcription with optional auto-detection
- Support for various audio and video formats
- Timestamped output as PDF, plain text, JSON, SRT subtitles and Markdown notes

MeetNote is suitable for transcribing meetings, interviews, podcasts, and any multi-speaker audio content.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Configuration](#configuration)
4. [Output Formats](#output-formats)
5. [Common Issues](#common-issues)
6. [Advanced Features](#advanced-features)
7. [Contributing](#contributing)
8. [License](#license)
9. [Author](#author)
10. [Acknowledgments](#acknowledgments)

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

### GUI

1. Run the application:
   ```
   python run.py
   ```

2. Use the GUI to:
   - Select an audio/video file
   - Choose transcription method (local Whisper or Groq Cloud)
   - Choose the spoken language (or "Auto-detect")
   - Set the number of speakers
   - Select diarization model
   - Choose output directory

3. Click "Start Processing" to begin transcription and diarization.

4. Once complete, find the PDF output in your specified directory.

### Command line

Passing a file runs the same pipeline without opening the GUI, which is handy
for batch work:

```
python run.py meeting.mp4                      # uses your saved settings
python run.py interview.mp3 -l ru -s 2         # Russian, two speakers
python run.py call.m4a -l auto -o ./out        # detect the language
python run.py lecture.mkv -l uk --translate    # Ukrainian speech, English output
python run.py recording.wav -m groq            # force the Groq backend
```

Run `python run.py --help` for the full list of options.

## Languages

English is the default, but any Whisper-supported language works. Set it in the
GUI's **Language** dropdown, with `--language` on the command line, or via
`transcription.language` in `Config/config.json`:

```json
"transcription": {
    "language": "ru",
    "task": "transcribe"
}
```

Use `"auto"` to detect the language per file. Built-in choices are English,
Russian, Ukrainian, Polish, German, French, Spanish, Italian, Portuguese, Dutch,
Czech, Romanian, Turkish, Japanese, Korean and Chinese; any other Whisper
language code also works if you set it in the config directly.

Set `"task": "translate"` (or pass `--translate`) to render non-English speech
as English text instead of transcribing it verbatim.

### Two things worth knowing

**English-only models are swapped automatically.** The default local model is
`medium.en`, which is English-only — given Russian it returns fluent-looking
nonsense rather than an error. When the language is anything other than English,
the multilingual equivalent (`medium`) is selected instead. The first run in a
new language therefore downloads that model. Set
`model_options.local.model` to `medium` or `large-v3` to avoid keeping both.

**Non-Latin output needs the bundled font.** PDF generation falls back to a
built-in font that cannot represent Cyrillic. `Fonts/DejaVuSans.ttf` ships with
the project and is used automatically; if it is missing, transcripts containing
non-Latin text fail with a clear message rather than producing an empty PDF.

## Configuration

- Edit `Config/config.json` to change default settings. Missing keys are filled
  in from the defaults on load, so an older config keeps working after an update.
- Key configurations:
  - `use_cuda`: Enable/disable GPU acceleration
  - `model_options.local.model`: Whisper model size for local transcription
  - `model_options.groq.model` / `.fallback_model`: Groq speech models; the
    fallback is used when the primary is rate-limited
  - `transcription.language`: Spoken language code, or `"auto"`
  - `transcription.task`: `"transcribe"` or `"translate"`
  - `diarization`: Adjust speaker detection parameters
  - `transcription_method`: Set to "groq" to use Groq API or "local" for Whisper model
  - `combiner`: Configure the combining strategy:
    ```json
    "combiner": {
        "method": "weighted",                // Default combiner method
        "model": "llama-3.3-70b-versatile"   // Used by LLM-based combiners
    }
    ```
    Recommended combiner methods (regression-tested against known
    speaker-assignment failure cases):
    - word_level: Assigns each *word* to its speaker and splits segments at
      mid-segment speaker changes. Most accurate; needs local transcription
      with word timestamps (the default decode settings provide them) and
      falls back to `weighted` without them.
    - weighted: Segment-level maximum-overlap assignment (default). The only
      segment-level combiner that passed every regression case.
    - simple: Basic time-based combining.

    Legacy methods, kept for comparison — each has known defects found in
    testing: `semantic` and `adaptive` can hand a segment to a short
    backchannel turn; `semantic_adaptive` picks the first overlapping turn
    rather than the longest and misses contained turns; `semantic_flow`
    relabels quick on-topic replies with the previous speaker; their
    embedding models are English-only, so they carry no signal for
    Russian/Ukrainian audio. `groq_llm` ignores diarization entirely;
    `two_stage_llm` and `local_llama_tiny` build on the above.

Note: Ensure your Groq API key is correctly set in the `.env` file when using the Groq transcription method or LLM-based combiners.

## Output Formats

Every transcription writes a PDF plus a set of sidecar files sharing its
basename, all UTF-8:

| Format | Contents |
| ------ | -------- |
| `.pdf` | The primary artifact; each line prefixed `[HH:MM:SS]` |
| `.txt` | `[HH:MM:SS] SPEAKER: text`, one line per segment |
| `.json` | `[{"start", "end", "speaker", "text"}, ...]` with numeric seconds |
| `.srt` | SubRip subtitles, for captioned playback of the recording |
| `.md` | Meeting notes; consecutive segments from one speaker merged |

```json
"output": {
    "formats": ["pdf", "txt", "json", "srt", "md"],
    "timestamps_in_pdf": true
}
```

- `formats`: which files to write. Drop the ones you do not want. `pdf` is
  always produced regardless, since it is the recorded output of a run.
- `timestamps_in_pdf`: set to `false` for a PDF of bare `SPEAKER: text` lines.
  This is a PDF-layout preference only — the sidecars keep their timestamps.

Segments whose combiner did not supply timestamps degrade gracefully: the text
formats fall back to an unstamped `SPEAKER: text` line, `.json` records `null`,
and `.srt` skips them (a subtitle with no time range has nowhere to go). A
sidecar that cannot be written logs a warning and does not fail the run.

## Running the Tests

The unit tests cover language and model resolution, config handling, chunk
sizing, the rate limiter, transcription tracking and PDF output. They need no
API keys and no audio:

```
pytest
```

Tests that contact a live service are skipped unless you ask for them:

```
pytest --run-network
```

## Verifying Your Setup

To check your environment end to end — API keys, GPU, model downloads — run the
setup script from the project root:

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

6. **Transcript is English gibberish for Russian/Ukrainian audio**:
   - The language is still set to English. Change it in the GUI's Language
     dropdown, or set `transcription.language` in `Config/config.json`.

7. **"characters outside Latin-1" error when saving the PDF**:
   - `Fonts/DejaVuSans.ttf` is missing. Restore it from the `dejavu-sans`
     directory or download DejaVu Sans again.

8. **A model download starts after switching language**:
   - Expected. English-only checkpoints such as `medium.en` cannot handle other
     languages, so the multilingual build is fetched the first time you need it.

9. **"Could not locate cudnn_ops_infer64_8.dll" (or `cudnn_ops64_9.dll`)**:
   - CTranslate2 and PyTorch disagree about the cuDNN version. CTranslate2
     below 4.5.0 needs cuDNN 8, while torch's CUDA 12 wheels bundle cuDNN 9.
   - Fix: `pip install -U "ctranslate2>=4.5.0,<5"`
   - Diarization is unaffected because it goes through torch, so this shows up
     as a GPU failure during transcription only. The app now checks for this
     before transcribing and falls back to CPU rather than crashing mid-run.

## Advanced Features

### Multiple Combiner Methods

MeetNote merges transcription and diarization through pluggable combiners.
`weighted` (segment-level max-overlap) is the default; `word_level` assigns
speakers per word and is the most accurate when word timestamps are available.
See the Configuration section for the full list and the known defects of the
legacy semantic/LLM combiners.

### Crash resilience

Transcription and diarization results are saved to `data/intermediate/` the
moment they exist. If a run fails later — an unreachable output folder, a
crash while writing — re-running the same file with the same settings resumes
from the saved results instead of repeating the GPU work. The cache entry is
removed once the output files are safely written.

### Automatic speaker count

Set the speaker spinbox (or `-s`) to `0` and pyannote chooses the speaker
count itself, bounded by `diarization.min_speakers`/`max_speakers` (default
1–10). Prefer this when the count is uncertain: forcing it too low merges
different speakers irrecoverably, while auto detection is usually within one
of the truth.

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