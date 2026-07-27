"""Language handling and model resolution."""

import pytest

from utils.languages import (
    SUPPORTED_LANGUAGES,
    code_for,
    describe,
    is_english_only,
    label_for,
    normalize_language,
    resolve_groq_model,
    resolve_local_model,
)


@pytest.mark.parametrize('value,expected', [
    ('en', 'en'),
    ('EN', 'en'),
    ('  ru  ', 'ru'),
    ('uk', 'uk'),
    ('auto', None),
    ('Auto-Detect', None),
    ('', None),
    (None, None),
])
def test_normalize_language(value, expected):
    assert normalize_language(value) == expected


def test_is_english_only_excludes_auto_detect():
    assert is_english_only('en')
    assert not is_english_only('ru')
    # Auto-detect must not count as English, or an English-only model would be
    # kept for a file that turns out to be Russian.
    assert not is_english_only('auto')


class TestLocalModelResolution:
    """`.en` checkpoints emit confident English nonsense for other languages."""

    def test_english_keeps_english_only_model(self):
        assert resolve_local_model('medium.en', 'en') == 'medium.en'

    @pytest.mark.parametrize('language', ['ru', 'uk', 'pl', 'de'])
    def test_other_languages_get_multilingual_variant(self, language):
        assert resolve_local_model('medium.en', language) == 'medium'

    def test_auto_detect_gets_multilingual_variant(self):
        assert resolve_local_model('medium.en', 'auto') == 'medium'

    def test_multilingual_model_is_left_alone(self):
        assert resolve_local_model('large-v3', 'ru') == 'large-v3'
        assert resolve_local_model('medium', 'uk') == 'medium'

    def test_distil_models_fall_back_to_large(self):
        # Every distil-whisper checkpoint is an English-only distillation, so
        # stripping a suffix would not produce a usable name.
        assert resolve_local_model('distil-large-v3', 'ru') == 'large-v3'

    def test_unknown_en_suffix_falls_back(self):
        assert resolve_local_model('custom-thing.en', 'ru') == 'large-v3'


class TestGroqModelResolution:
    def test_multilingual_models_pass_through(self):
        assert resolve_groq_model('whisper-large-v3-turbo', 'ru') == 'whisper-large-v3-turbo'
        assert resolve_groq_model('whisper-large-v3', 'uk') == 'whisper-large-v3'

    @pytest.mark.parametrize('language', ['en', 'ru', 'auto'])
    def test_retired_model_is_replaced_for_every_language(self, language):
        # Groq removed this checkpoint; keeping it would 404 regardless of
        # which language was requested.
        assert resolve_groq_model('distil-whisper-large-v3-en', language) == 'whisper-large-v3'

    def test_empty_model_uses_default(self):
        assert resolve_groq_model('', 'ru') == 'whisper-large-v3'
        assert resolve_groq_model(None, 'en') == 'whisper-large-v3'


def test_labels_round_trip():
    for _, code in SUPPORTED_LANGUAGES:
        assert code_for(label_for(code)) == code


def test_label_for_handles_auto_and_unknown():
    assert label_for('auto') == 'Auto-detect'
    assert label_for(None) == 'Auto-detect'
    assert label_for('definitely-not-a-language') == 'English'


def test_code_for_unknown_label_is_auto():
    assert code_for('Klingon') == 'auto'


def test_describe():
    assert describe('ru') == 'Russian'
    assert describe('auto') == 'auto-detect'
    assert describe(None) == 'auto-detect'
