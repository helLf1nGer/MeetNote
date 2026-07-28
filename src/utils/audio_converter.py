from pydub import AudioSegment
import os
import logging
import subprocess

logger = logging.getLogger(__name__)

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def extract_mp3_from_wav(wav_path, output_dir=None, duration=None):
    """
    Extract an MP3 from a WAV file.
    
    :param wav_path: Path to the input WAV file
    :param output_dir: Directory to save the output MP3 file (default: same as input)
    :param duration: Duration in milliseconds to extract (default: entire file)
    :return: Path to the extracted MP3 file
    """
    if not check_ffmpeg():
        logger.error("ffmpeg is not installed or not in the system PATH. Please install ffmpeg and add it to your PATH.")
        raise EnvironmentError("ffmpeg is not installed or not in the system PATH")

    try:
        # Check if input file exists
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Input file not found: {wav_path}")

        # Load the WAV file
        audio = AudioSegment.from_wav(wav_path)
        
        # If duration is specified, trim the audio
        if duration:
            audio = audio[:duration]
        
        # Determine the output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.mp3")
        else:
            output_path = os.path.splitext(wav_path)[0] + ".mp3"
        
        # Export as MP3
        audio.export(output_path, format="mp3")
        
        logger.info(f"Successfully extracted MP3: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error extracting MP3 from WAV: {str(e)}")
        raise

if __name__ == "__main__":
    # Paths come from the command line rather than being hardcoded here: the
    # previous version carried a developer's own directory and recording name,
    # which is both unusable for anyone else and personal information sitting in
    # a public repository.
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Extract an MP3 from a WAV file.")
    parser.add_argument('wav_path', help="Input WAV file.")
    parser.add_argument('-o', '--output-dir',
                        help="Directory for the MP3 (default: alongside the input).")
    parser.add_argument('-d', '--duration', type=int,
                        help="Milliseconds to extract (default: the whole file).")
    args = parser.parse_args()

    try:
        extract_mp3_from_wav(args.wav_path, output_dir=args.output_dir, duration=args.duration)
    except Exception as e:
        logger.error(f"Failed to extract MP3: {str(e)}")
        raise SystemExit(1)