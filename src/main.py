"""
Entry point for MeetNote.

Without arguments this launches the GUI. With a file argument it runs the same
pipeline headlessly, which is useful for batch work and for testing changes
without clicking through the interface.
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from utils.config_manager import ConfigManager
from utils.languages import DEFAULT_LANGUAGE, LANGUAGE_CODES
from utils.logging_setup import configure_logging

load_dotenv()

# Configured here, once, before anything else logs. Doing this via
# logging.basicConfig in two modules meant whichever imported first won, so
# deferring an import could silently disable file logging.
_log_file = configure_logging()
logger = logging.getLogger(__name__)
if _log_file:
    logger.info("Logging to %s", _log_file)


def main(argv=None):
    args = _parse_args(argv)
    if args.file:
        return _run_cli(args)
    return _run_gui()


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='meetnote',
        description='Transcribe and diarize audio or video. Launches the GUI when no file is given.',
    )
    parser.add_argument('file', nargs='?', help='Audio or video file to process.')
    parser.add_argument(
        '-l', '--language', choices=LANGUAGE_CODES, default=None,
        help=f"Spoken language, or 'auto' to detect it (default: from config, initially '{DEFAULT_LANGUAGE}').",
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Directory for the generated PDF (default: from config).',
    )
    parser.add_argument(
        '-s', '--speakers', type=int, default=None,
        help='Number of speakers to detect, or 0 to let pyannote decide '
             '(default: from config).',
    )
    parser.add_argument(
        '-m', '--method', choices=['local', 'groq'], default=None,
        help='Transcription backend (default: from config).',
    )
    parser.add_argument(
        '--translate', action='store_true',
        help='Translate the speech into English instead of transcribing verbatim.',
    )
    return parser.parse_args(argv)


def _run_gui():
    from gui.main_window import create_gui

    try:
        _window, root = create_gui()
        root.mainloop()
        return 0
    except Exception as e:
        logger.error("An error occurred: %s", e)
        raise


def _run_cli(args):
    from pipeline import run_pipeline

    if not os.path.isfile(args.file):
        logger.error("File not found: %s", args.file)
        return 1

    config_manager = ConfigManager()
    config = config_manager.config
    settings = {
        'file_path': os.path.abspath(args.file),
        # `or` would be wrong here: -s 0 requests auto-detection, and being
        # falsy it would silently fall back to the configured count instead.
        'num_speakers': (args.speakers if args.speakers is not None
                         else config.get('diarization', {}).get('default_num_speakers', 2)),
        'diarization_model': config.get('diarization', {}).get('model', 'speaker-diarization-3.1'),
        'transcription_method': args.method or config.get('transcription_method', 'local'),
        'processing_location': config.get('processing_location', 'local'),
        'output_directory': args.output or config.get('output_directory'),
        'combiner_method': config.get('combiner', {}).get('method'),
        'language': args.language or config.get('transcription', {}).get('language'),
    }

    # --translate applies to this run only; the pipeline persists the config it
    # runs with, so the previous value is put back afterwards rather than
    # silently becoming the new default for the GUI.
    transcription = config.setdefault('transcription', {})
    previous_task = transcription.get('task', 'transcribe')
    if args.translate:
        transcription['task'] = 'translate'

    def show_progress(percent, message):
        print(f"[{percent:3d}%] {message}", flush=True)

    try:
        result = run_pipeline(settings, progress_callback=show_progress)
    except Exception as e:
        logger.error("Processing failed: %s", e)
        return 1
    finally:
        if args.translate:
            config.setdefault('transcription', {})['task'] = previous_task
            config_manager.save_config()

    print(f"\nSaved: {result['output_pdf']}")
    print(f"Elapsed: {result['elapsed']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
