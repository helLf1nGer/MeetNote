"""
Persistence for intermediate pipeline results.

Transcription and diarization are the expensive steps - tens of minutes of GPU
time for a long meeting - while everything after them is cheap. Writing them to
disk as soon as they exist means a later failure (an unwritable Google Drive
output path, a bad combiner, a closed window) costs seconds to recover from
instead of re-running the whole job.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Anchored to the project root, not the working directory, so results are found
# again no matter where the app was launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / 'data' / 'intermediate'


def cache_key(file_path, **settings) -> str:
    """
    Identity for a cached result.

    Covers the media file itself (path, size, mtime) plus every setting that
    changes what the transcript should contain. Without the settings, re-running
    the same file in a different language would silently reuse the old
    transcript.
    """
    try:
        stats = os.stat(file_path)
        identity = f"{os.path.abspath(file_path)}|{stats.st_size}|{int(stats.st_mtime)}"
    except OSError:
        identity = str(file_path)

    for key in sorted(settings):
        identity += f"|{key}={settings[key]}"

    return hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]


class DataManager:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_CACHE_DIR
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Could not create cache directory %s: %s", self.data_dir, e)

    def _paths(self, file_id: str):
        return (
            self.data_dir / f"{file_id}_transcription.json",
            self.data_dir / f"{file_id}_diarization.json",
        )

    def save_results(self, file_id: str, transcription: List[Dict], diarization: List[Dict]) -> bool:
        """
        Persist both result sets. Never raises - a caching failure must not take
        down a run that otherwise succeeded.
        """
        transcription_file, diarization_file = self._paths(file_id)
        try:
            for path, payload in ((transcription_file, transcription),
                                  (diarization_file, diarization)):
                # Written via a temp file so an interrupted write cannot leave a
                # truncated JSON file that later looks like a valid cache hit.
                temp = path.with_suffix('.tmp')
                with open(temp, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                os.replace(temp, path)

            logger.info("[Cache] Saved intermediate results (%d segments, %d turns) as %s",
                        len(transcription), len(diarization), file_id)
            return True
        except (OSError, TypeError, ValueError) as e:
            logger.warning("[Cache] Could not save intermediate results: %s", e)
            return False

    def load_results(self, file_id: str):
        """Return (transcription, diarization), or (None, None) if unavailable."""
        transcription_file, diarization_file = self._paths(file_id)
        try:
            with open(transcription_file, encoding='utf-8') as f:
                transcription = json.load(f)
            with open(diarization_file, encoding='utf-8') as f:
                diarization = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("[Cache] Could not load intermediate results for %s: %s", file_id, e)
            return None, None

        if not isinstance(transcription, list) or not isinstance(diarization, list):
            logger.warning("[Cache] Ignoring malformed cache entry %s", file_id)
            return None, None

        logger.info("[Cache] Loaded intermediate results for %s", file_id)
        return transcription, diarization

    def results_exist(self, file_id: str) -> bool:
        return all(path.is_file() for path in self._paths(file_id))

    def discard(self, file_id: str):
        """Remove a cache entry, e.g. once the final output is safely written."""
        for path in self._paths(file_id):
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                logger.debug("[Cache] Could not remove %s: %s", path, e)

    def prune(self, keep: int = 20):
        """Keep the cache from growing without bound."""
        try:
            files = sorted(self.data_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return
        for stale in files[keep * 2:]:  # two files per entry
            try:
                stale.unlink()
            except OSError:
                pass
