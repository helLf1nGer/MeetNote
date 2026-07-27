import os
import sys

# Get the absolute path of the current file (run.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the src directory to the Python path so its packages import as top-level
# modules (audio, diarization, transcription, utils), which is how they refer
# to each other.
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from main import main

if __name__ == "__main__":
    sys.exit(main())
