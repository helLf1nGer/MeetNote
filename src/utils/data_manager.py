import json
import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def save_results(self, file_id: str, transcription: List[Dict], diarization: List[Dict]):
        """Save transcription and diarization results to JSON files."""
        transcription_file = os.path.join(self.data_dir, f"{file_id}_transcription.json")
        diarization_file = os.path.join(self.data_dir, f"{file_id}_diarization.json")

        with open(transcription_file, 'w') as f:
            json.dump(transcription, f, indent=2)
        
        with open(diarization_file, 'w') as f:
            json.dump(diarization, f, indent=2)

        logger.info(f"Saved results for {file_id}")

    def load_results(self, file_id: str) -> tuple:
        """Load transcription and diarization results from JSON files."""
        transcription_file = os.path.join(self.data_dir, f"{file_id}_transcription.json")
        diarization_file = os.path.join(self.data_dir, f"{file_id}_diarization.json")

        try:
            with open(transcription_file, 'r') as f:
                transcription = json.load(f)
            
            with open(diarization_file, 'r') as f:
                diarization = json.load(f)

            logger.info(f"Loaded results for {file_id}")
            return transcription, diarization
        except FileNotFoundError:
            logger.error(f"Results for {file_id} not found")
            return None, None

    def results_exist(self, file_id: str) -> bool:
        """Check if results exist for a given file_id."""
        transcription_file = os.path.join(self.data_dir, f"{file_id}_transcription.json")
        diarization_file = os.path.join(self.data_dir, f"{file_id}_diarization.json")
        return os.path.exists(transcription_file) and os.path.exists(diarization_file)