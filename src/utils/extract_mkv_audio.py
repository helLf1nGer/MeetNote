#!/usr/bin/env python3

'''
USAGE: python -m src.utils.extract_mkv_audio [File] [Output Format]
'''
import sys
import os
from pathlib import Path
import shutil

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from src
from src.audio.file_processor import extract_audio as process_extract_audio

def extract_audio(input_file: str, output_format: str = 'mp3') -> None:
    """
    Extract audio from a video file.
    
    Args:
        input_file (str): Path to the input video file
        output_format (str): Desired output format (default: mp3)
    """
    try:
        # Create output filename
        input_path = Path(input_file)
        output_file = input_path.with_suffix(f'.{output_format}')
        
        # Extract audio using file_processor (returns temp file path)
        temp_audio_file = process_extract_audio(str(input_path))
        
        # Move the temp file to desired location
        shutil.move(temp_audio_file, str(output_file))
        print(f"Successfully extracted audio to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_mkv_audio.py <input_video_file> [output_format]")
        sys.exit(1)
    
    # Handle paths with spaces by joining all arguments except the last one (if it's a format)
    if len(sys.argv) > 2 and not sys.argv[-1].startswith('.'):
        input_file = ' '.join(sys.argv[1:-1])
        output_format = sys.argv[-1]
    else:
        input_file = ' '.join(sys.argv[1:])
        output_format = 'mp3'
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    extract_audio(input_file, output_format)

if __name__ == "__main__":
    main() 