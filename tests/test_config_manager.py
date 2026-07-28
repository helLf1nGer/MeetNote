"""Config loading, defaults merging and dotted access."""

import json
import os

from utils.config_manager import (
    DEFAULT_DECODE_OPTIONS,
    DEFAULT_OUTPUT_FORMATS,
    SUPPORTED_OUTPUT_FORMATS,
    ConfigManager,
)


def test_instances_are_shared_per_file():
    # Several modules build a ConfigManager at import time; if each held its
    # own copy, a value saved by one would be invisible to the others.
    assert ConfigManager() is ConfigManager()


def test_template_lives_next_to_the_config():
    manager = ConfigManager()
    assert manager.template_file.endswith('config.template.json')
    # The template ships inside Config/, not at the project root.
    assert manager.config_dir in manager.template_file


def test_template_decode_block_matches_the_code_defaults():
    """
    A new install is seeded from the template, an upgrade from
    _get_default_config; if the two disagree the two paths transcribe
    differently for no visible reason.
    """
    manager = ConfigManager()
    with open(os.path.join(manager.config_dir, 'config.template.json'), encoding='utf-8') as f:
        template = json.load(f)
    assert template['transcription']['decode'] == DEFAULT_DECODE_OPTIONS


def test_template_diarization_block_matches_the_code_defaults():
    """Same drift risk as the decode block: two seeding paths, one behaviour."""
    manager = ConfigManager()
    with open(os.path.join(manager.config_dir, 'config.template.json'), encoding='utf-8') as f:
        template = json.load(f)
    # The template omits `model`, which the GUI writes on first use.
    defaults = manager._get_default_config()['diarization']
    assert template['diarization'] == defaults


def test_template_output_block_matches_the_code_defaults():
    """Same drift risk as the decode block: two seeding paths, one behaviour."""
    manager = ConfigManager()
    with open(os.path.join(manager.config_dir, 'config.template.json'), encoding='utf-8') as f:
        template = json.load(f)
    assert template['output']['formats'] == DEFAULT_OUTPUT_FORMATS


def test_the_default_output_is_pdf_only():
    """
    A fresh install writes one file, not five.

    The PDF is what nearly every run is read from; the other four are opt-in.
    Emitting them unasked is churn, and worse when the output directory is a
    synced cloud folder.
    """
    assert DEFAULT_OUTPUT_FORMATS == ['pdf']


def test_every_default_format_is_a_supported_one():
    assert set(DEFAULT_OUTPUT_FORMATS) <= set(SUPPORTED_OUTPUT_FORMATS)


def test_defaults_and_supported_formats_are_distinct_objects():
    # SUPPORTED_FORMATS in output_generator was briefly derived from the
    # defaults, which turned every opt-in format into an 'unknown format'.
    assert DEFAULT_OUTPUT_FORMATS is not SUPPORTED_OUTPUT_FORMATS
    assert len(SUPPORTED_OUTPUT_FORMATS) > len(DEFAULT_OUTPUT_FORMATS)


def test_decode_defaults_are_not_shared_between_configs():
    # _get_default_config hands out a nested dict; if it were the module-level
    # object, one caller's edit would rewrite the defaults process-wide.
    manager = ConfigManager()
    first = manager._get_default_config()['transcription']['decode']
    first['vad_parameters']['threshold'] = 0.99
    second = manager._get_default_config()['transcription']['decode']
    assert second['vad_parameters']['threshold'] == 0.5
    assert DEFAULT_DECODE_OPTIONS['vad_parameters']['threshold'] == 0.5


class TestMergeDefaults:
    def test_keeps_existing_values(self, isolated_config):
        merged = isolated_config._merge_defaults({'transcription': {'task': 'translate'}})
        assert merged['transcription']['task'] == 'translate'

    def test_fills_missing_nested_keys(self, isolated_config):
        # A config written before `language` existed must not raise KeyError.
        merged = isolated_config._merge_defaults({'transcription': {'task': 'transcribe'}})
        assert merged['transcription']['language'] == 'en'

    def test_adds_absent_sections(self, isolated_config):
        merged = isolated_config._merge_defaults({})
        assert 'pdf_output' in merged
        assert 'combiner' in merged

    def test_fills_in_the_decode_block(self, isolated_config):
        # Configs written before the decode block existed must still get the
        # anti-hallucination tuning rather than faster-whisper's raw defaults.
        merged = isolated_config._merge_defaults({'transcription': {'language': 'ru'}})
        assert merged['transcription']['language'] == 'ru'
        assert merged['transcription']['decode']['vad_filter'] is True
        assert merged['transcription']['decode']['condition_on_previous_text'] is False

    def test_fills_in_the_diarization_batch_sizes(self, isolated_config):
        # A config written before these existed must still get the throughput
        # fix rather than pyannote's batch size of 1.
        merged = isolated_config._merge_defaults({'diarization': {'max_speakers': 4}})
        assert merged['diarization']['max_speakers'] == 4
        assert merged['diarization']['segmentation_batch_size'] > 1
        assert merged['diarization']['embedding_batch_size'] > 1

    def test_decode_overrides_survive_the_merge(self, isolated_config):
        merged = isolated_config._merge_defaults(
            {'transcription': {'decode': {'beam_size': 1}}}
        )
        assert merged['transcription']['decode']['beam_size'] == 1
        # Untouched sibling keys still arrive from the defaults.
        assert merged['transcription']['decode']['word_timestamps'] is True

    def test_does_not_clobber_user_sections(self, isolated_config):
        merged = isolated_config._merge_defaults({'output_directory': '/somewhere/custom'})
        assert merged['output_directory'] == '/somewhere/custom'


class TestDottedAccess:
    def test_get_nested(self, isolated_config):
        assert isolated_config.get('transcription.language') == 'en'
        assert isolated_config.get('model_options.local.model') == 'medium.en'

    def test_get_missing_returns_default(self, isolated_config):
        assert isolated_config.get('nope.not.here', 'fallback') == 'fallback'

    def test_get_flat_key_still_works(self, isolated_config):
        assert isolated_config.get('use_cuda') is True

    def test_set_nested_creates_intermediates(self, isolated_config):
        isolated_config.set('a.b.c', 42)
        assert isolated_config.get('a.b.c') == 42

    def test_set_persists(self, isolated_config):
        isolated_config.set('transcription.language', 'uk')
        with open(isolated_config.config_file, encoding='utf-8') as f:
            assert json.load(f)['transcription']['language'] == 'uk'


class TestPersistence:
    def test_round_trip(self, isolated_config):
        isolated_config.config['output_directory'] = '/tmp/out'
        isolated_config.save_config()
        isolated_config.config = {}
        assert isolated_config.load_config()['output_directory'] == '/tmp/out'

    def test_non_ascii_survives(self, isolated_config):
        # Output paths and transcripts may contain Cyrillic.
        isolated_config.config['output_directory'] = '/dom/Транскрипции'
        isolated_config.save_config()
        isolated_config.config = {}
        assert isolated_config.load_config()['output_directory'] == '/dom/Транскрипции'

    def test_reload_keeps_the_same_dict_object(self, isolated_config):
        """
        Regression test.

        The GUI panels bind `self.config = config_manager.config` once at
        startup. create_pdf calls load_config(), so if reloading rebound the
        attribute to a fresh dict, every settings change made after the first
        transcription would be written to an orphaned copy and lost.
        """
        isolated_config.save_config()
        held_by_gui = isolated_config.config

        isolated_config.load_config()

        assert isolated_config.config is held_by_gui

        # A write through the held reference must still reach disk.
        held_by_gui['output_directory'] = '/changed/after/reload'
        isolated_config.save_config()
        with open(isolated_config.config_file, encoding='utf-8') as f:
            assert json.load(f)['output_directory'] == '/changed/after/reload'

    def test_corrupt_file_falls_back_to_defaults(self, isolated_config):
        with open(isolated_config.config_file, 'w', encoding='utf-8') as f:
            f.write('{ not valid json')
        # A damaged config should not take the whole application down.
        assert isolated_config._load_config()['transcription']['language'] == 'en'
