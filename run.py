import os
import sys

# Get the absolute path of the current file (run.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the src directory to the Python path
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

# Import and run the main function
from src.main import main

if __name__ == "__main__":
    main()