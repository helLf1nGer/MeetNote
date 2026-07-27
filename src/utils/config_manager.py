import copy
import json
import logging
import os
import shutil
import threading

from .languages import DEFAULT_LANGUAGE

logger = logging.getLogger(__name__)

# Decode parameters handed to faster-whisper's ``model.transcribe``. They live
# here rather than in the transcriber so the defaults, the config template and
# the runtime fallback cannot drift apart; this module also stays free of the ML
# imports, so the GUI and tests can read the defaults without loading torch.
#
# The tuning targets 30-60 minute meetings, where Whisper's dominant failure
# mode is hallucinating training-set priors ("Thank you.", subtitle boilerplate)
# over silence.
DEFAULT_DECODE_OPTIONS = {
    # Strips silence before the decoder ever sees it - the root-cause fix for
    # hallucinated filler, rather than filtering it out afterwards.
    'vad_filter': True,
    'vad_parameters': {
        'threshold': 0.5,
        # Shorter than the 2000 ms default: meeting speech has frequent pauses,
        # and merging across them produces oversized segments that diarization
        # then cannot attribute to one speaker.
        'min_silence_duration_ms': 1000,
        'speech_pad_ms': 400,
    },
    # The library default (True) feeds each window's output back as the next
    # window's prompt, so one hallucination re-seeds itself for the rest of the
    # file. Turning it off costs a little cross-sentence coherence.
    'condition_on_previous_text': False,
    # Prerequisite for hallucination_silence_threshold, and the basis for
    # word-level speaker assignment later.
    'word_timestamps': True,
    # Skip silence longer than this before a suspected hallucination.
    'hallucination_silence_threshold': 2.0,
    # Already the library default; explicit so it is visible and tunable.
    'beam_size': 5,
    # Persistent domain vocabulary (company and participant names). Use this
    # rather than initial_prompt: with condition_on_previous_text=False,
    # prompt_reset_since is reset every window, so initial_prompt only reaches
    # the first 30 seconds, whereas hotwords are re-injected on every window.
    'hotwords': None,
}

# Files written per transcription. Kept here rather than in output_generator so
# the GUI and the config template can read it without importing fpdf; the
# writers themselves live in utils/output_generator.py, which validates names
# against this list.
DEFAULT_OUTPUT_FORMATS = ['pdf', 'txt', 'json', 'srt', 'md']


class ConfigManager:
    """
    Loads and persists ``Config/config.json``.

    Instances are shared per config file: several modules construct a
    ``ConfigManager()`` at import time, and previously each held a private copy
    of the settings, so a value saved by one was invisible to the others until
    the next process start.
    """

    _instances = {}
    _instances_lock = threading.Lock()

    def __new__(cls, config_file='config.json'):
        with cls._instances_lock:
            instance = cls._instances.get(config_file)
            if instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[config_file] = instance
            return instance

    def __init__(self, config_file='config.json'):
        if self._initialized:
            return

        # Project root is two levels up from src/utils.
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.config_dir = os.path.join(self.project_root, 'Config')
        self.config_file = os.path.join(self.config_dir, config_file)
        self.template_file = os.path.join(self.config_dir, 'config.template.json')
        self._lock = threading.RLock()

        logger.debug("Looking for config at: %s", self.config_file)
        self.config = self._load_config()
        self._initialized = True

    def _load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return self._merge_defaults(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Could not read %s (%s); falling back to defaults.", self.config_file, e)
                return self._get_default_config()
        return self._create_config_from_template()

    def _create_config_from_template(self):
        if os.path.exists(self.template_file):
            try:
                os.makedirs(self.config_dir, exist_ok=True)
                shutil.copy(self.template_file, self.config_file)
                logger.info("Created new configuration file: %s", self.config_file)
                logger.info("Please update the configuration with your specific settings.")
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return self._merge_defaults(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Could not apply config template (%s); falling back to defaults.", e)
        return self._get_default_config()

    def _merge_defaults(self, config):
        """
        Fill in keys the defaults define but the on-disk config lacks.

        Without this, settings added after a user's ``config.json`` was written
        would raise ``KeyError`` instead of taking their default value.
        """
        defaults = self._get_default_config()

        def merge(base, incoming):
            for key, value in base.items():
                if key not in incoming:
                    incoming[key] = value
                elif isinstance(value, dict) and isinstance(incoming[key], dict):
                    merge(value, incoming[key])
            return incoming

        return merge(defaults, config)

    def _get_default_config(self):
        return {
            'misc': {
                'print_to_terminal': True
            },
            'model_options': {
                'local': {
                    'model': 'medium.en',
                    'device': 'cuda',
                    'compute_type': 'float16'
                },
                'groq': {
                    'model': 'whisper-large-v3-turbo',
                    'fallback_model': 'whisper-large-v3'
                }
            },
            'use_cuda': True,
            'output_directory': 'transcriptions',
            'diarization': {
                # The range pyannote searches when default_num_speakers is 0.
                'min_speakers': 1,
                'max_speakers': 10,
                # 0 means auto: pyannote picks the count within the range above.
                'default_num_speakers': 2,
                # pyannote defaults both to 1, which leaves the GPU idle between
                # per-chunk kernel launches. Measured on a 6 GB RTX 3060 over a
                # 5-minute file: 1 -> 29.5s/137 MB peak, 16 -> 14.8s/860 MB,
                # 32 -> 13.6s/1661 MB. 16 keeps nearly all the speedup for half
                # the VRAM; peak scales with batch size, not file length.
                'segmentation_batch_size': 16,
                'embedding_batch_size': 16
            },
            'transcription': {
                # 'auto' detects the language per file; a specific code such as
                # 'ru' or 'uk' forces it. See utils/languages.py.
                'language': DEFAULT_LANGUAGE,
                'task': 'transcribe',
                # Deep-copied so a caller mutating the returned config cannot
                # rewrite the module-level defaults for the whole process.
                'decode': copy.deepcopy(DEFAULT_DECODE_OPTIONS)
            },
            'pdf_output': {
                'font_size': 12,
                'line_spacing': 1.2
            },
            'output': {
                # Which files to write per transcription. 'pdf' is produced
                # regardless - it is what the pipeline returns and what the
                # transcription tracker records - so removing it has no effect.
                # Copied so a caller trimming the list cannot shorten the
                # defaults for the rest of the process.
                'formats': list(DEFAULT_OUTPUT_FORMATS),
                'timestamps_in_pdf': True
            },
            'combiner': {
                'method': 'weighted',
                'model': 'llama-3.3-70b-versatile'
            }
        }

    def load_config(self):
        """
        Re-read the config from disk.

        The existing dict is updated in place rather than replaced: the GUI
        panels bind ``self.config = config_manager.config`` once at startup, and
        rebinding here would leave them mutating an orphaned copy whose changes
        never reach disk.
        """
        with self._lock:
            if os.path.exists(self.config_file):
                try:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        loaded = self._merge_defaults(json.load(f))
                except (json.JSONDecodeError, OSError) as e:
                    logger.error("Could not reload %s (%s); keeping in-memory config.", self.config_file, e)
                    return self.config
            else:
                loaded = self._create_config_from_template()

            self.config.clear()
            self.config.update(loaded)
            return self.config

    def save_config(self):
        with self._lock:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)

    def get(self, key, default=None):
        """Read a setting. Supports dotted paths, e.g. ``transcription.language``."""
        value = self.config
        for part in key.split('.'):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def set(self, key, value):
        """Write a setting. Supports dotted paths, creating intermediate dicts."""
        with self._lock:
            parts = key.split('.')
            target = self.config
            for part in parts[:-1]:
                if not isinstance(target.get(part), dict):
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        self.save_config()

    def update(self, new_config):
        with self._lock:
            self.config.update(new_config)
        self.save_config()
