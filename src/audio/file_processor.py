import os
import logging
from pydub import AudioSegment
import tempfile
import ffmpeg

logger = logging.getLogger(__name__)

def process_file(file_path):
    """Process the input file, extracting audio if necessary."""
    file_type = _analyze_file(file_path)
    if file_type == 'audio':
        return file_path
    elif file_type == 'video':
        return extract_audio(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def _analyze_file(file_path):
    """Analyze the input file and return its type."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext in ['.mp3', '.wav', '.m4a', '.flac']:
        return 'audio'
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return 'video'
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def extract_audio(file_path):
    """Extract audio from video file to a temporary MP3 file."""
    logger.info(f"Extracting audio from video: {file_path}")
    try:
        output_path = tempfile.mktemp(suffix='.mp3')
        (
            ffmpeg
            .input(file_path)
            .output(output_path, acodec='libmp3lame', ac=2, ar='44100')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        logger.info(f"Audio extracted successfully: {output_path}")
        return output_path
    except ffmpeg.Error as e:
        logger.error(f"Error extracting audio: {e.stderr.decode()}")
        raise