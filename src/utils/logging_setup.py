"""
Central logging configuration.

Logging is set up in exactly one place and called explicitly. Previously two
modules each called ``logging.basicConfig``, and since that function is a no-op
once the root logger has handlers, whichever imported first silently won - so
moving an import could switch file logging off without any error.

The log file is anchored to the project root rather than the working directory,
so it lands in the same place no matter where the app is launched from.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_FILE = LOG_DIR / 'app.log'

MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
BACKUP_COUNT = 3

_configured = False


def configure_logging(level=logging.INFO, log_file=LOG_FILE, console=True):
    """
    Install the root logger's handlers. Safe to call more than once.

    Returns the resolved log file path, or None if file logging could not be
    set up (in which case console logging still works).
    """
    global _configured
    root = logging.getLogger()

    if _configured:
        return log_file if any(
            isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers
        ) else None

    # Replace anything a stray basicConfig may have installed, so this function
    # is authoritative regardless of import order.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        # Transcripts contain Cyrillic; a cp1252 console would otherwise raise
        # UnicodeEncodeError from inside the logging call itself.
        _force_utf8(stream)
        root.addHandler(stream)

    resolved = None
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        resolved = log_file
    except OSError as e:
        root.warning("Could not open log file %s (%s); logging to console only.", log_file, e)

    # These are chatty at INFO and drown out the pipeline's own messages.
    for noisy in ('urllib3', 'httpx', 'httpcore', 'matplotlib', 'PIL',
                  'speechbrain', 'pytorch_lightning', 'filelock'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    return resolved


def _force_utf8(handler):
    """Reconfigure a stream handler for UTF-8 where the platform allows it."""
    stream = getattr(handler, 'stream', None)
    reconfigure = getattr(stream, 'reconfigure', None)
    if reconfigure is not None:
        try:
            reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
