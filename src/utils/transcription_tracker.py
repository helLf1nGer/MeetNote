from pathlib import Path
import json
import hashlib
import os
import logging
from datetime import datetime

# Two levels up from src/utils, matching ConfigManager.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TranscriptionTracker:
    def __init__(self):
        # Anchored to the project root rather than the working directory, so
        # launching from elsewhere still finds the same tracker file.
        self.tracker_file = PROJECT_ROOT / 'data' / 'transcribed_files.json'
        self._ensure_tracker_file()
        self.transcribed_files = self._load_tracker()
        self._hash_cache = {}
        self._migrate_existing_data()
    
    def _ensure_tracker_file(self):
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.tracker_file.exists():
            self.tracker_file.write_text("{}")
    
    def _load_tracker(self):
        try:
            return json.loads(self.tracker_file.read_text())
        except json.JSONDecodeError:
            logging.warning("Corrupted tracker file. Creating new one.")
            return {}
    
    def _migrate_existing_data(self):
        """Migrate old format data to new format if necessary"""
        need_save = False
        for file_hash, data in self.transcribed_files.items():
            if isinstance(data, dict) and 'transcriptions' not in data:
                # Convert old format to new format
                self.transcribed_files[file_hash] = {
                    'original_path': data.get('original_path', ''),
                    'transcriptions': [{
                        'date': data.get('last_modified', datetime.now().isoformat()),
                        'size': data.get('size', 0),
                        'output_path': ''  # Can't recover old output path
                    }]
                }
                need_save = True
        
        if need_save:
            self._save_tracker()
    
    def _save_tracker(self):
        try:
            self.tracker_file.write_text(json.dumps(self.transcribed_files, indent=2))
        except Exception as e:
            logging.error(f"Error saving tracker file: {e}")
    
    def _calculate_file_hash(self, file_path):
        """
        Calculate a content hash for a file, cached on its size and mtime.

        This deliberately does not use ``functools.lru_cache``: keyed on the
        path alone, that returns the old hash after a file has been edited, so
        a modified recording would be reported as already transcribed. The
        size/mtime key below is what makes the cache safe to reuse.

        Large files are sampled (first and last 1 MB) rather than read whole,
        which keeps browsing a directory of recordings responsive.
        """
        stats = os.stat(file_path)
        cache_key = (str(file_path), stats.st_size, stats.st_mtime)

        cached = self._hash_cache.get(cache_key)
        if cached is not None:
            return cached

        # Note: the digest covers only the sampled bytes. Changing what goes
        # into it would invalidate every existing entry in the tracker file.
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            if stats.st_size > 2_000_000:  # 2MB
                sha256_hash.update(f.read(1_000_000))
                f.seek(-1_000_000, 2)
                sha256_hash.update(f.read())
            else:
                sha256_hash.update(f.read())

        file_hash = sha256_hash.hexdigest()
        self._hash_cache[cache_key] = file_hash
        return file_hash
    
    def mark_as_transcribed(self, file_path, output_path=''):
        """Mark a file as transcribed, keeping history of transcriptions"""
        try:
            file_hash = self._calculate_file_hash(file_path)
            current_time = datetime.now().isoformat()
            
            if file_hash not in self.transcribed_files:
                self.transcribed_files[file_hash] = {
                    'original_path': str(file_path),
                    'transcriptions': []
                }
            
            # Add new transcription record
            self.transcribed_files[file_hash]['transcriptions'].append({
                'date': current_time,
                'size': os.path.getsize(file_path),
                'output_path': str(output_path)
            })
            
            self._save_tracker()
            return True
        except Exception as e:
            logging.error(f"Error marking file as transcribed: {e}")
            return False
    
    def get_transcription_history(self, file_path):
        """Get all transcription records for a file"""
        try:
            file_hash = self._calculate_file_hash(file_path)
            if file_hash in self.transcribed_files:
                return self.transcribed_files[file_hash]['transcriptions']
        except Exception as e:
            logging.error(f"Error getting transcription history: {e}")
        return []
    
    def is_transcribed(self, file_path):
        """Check if a file has been transcribed"""
        try:
            file_hash = self._calculate_file_hash(file_path)
            return file_hash in self.transcribed_files and \
                   len(self.transcribed_files[file_hash]['transcriptions']) > 0
        except Exception as e:
            logging.error(f"Error checking transcription status: {e}")
            return False 