"""
Language support for transcription.

English is the primary use case for this tool, but Whisper is multilingual and
several models ship in English-only variants that silently produce garbage when
handed other languages. This module keeps the language list and the model
resolution rules in one place so both the local and Groq backends agree.
"""

import logging

logger = logging.getLogger(__name__)

AUTO_DETECT = 'auto'
DEFAULT_LANGUAGE = 'en'

# (display name, Whisper language code). Deliberately a short list rather than
# all 99 Whisper languages - these are the ones this tool actually gets used for.
SUPPORTED_LANGUAGES = [
    ('Auto-detect', AUTO_DETECT),
    ('English', 'en'),
    ('Russian', 'ru'),
    ('Ukrainian', 'uk'),
    ('Polish', 'pl'),
    ('German', 'de'),
    ('French', 'fr'),
    ('Spanish', 'es'),
    ('Italian', 'it'),
    ('Portuguese', 'pt'),
    ('Dutch', 'nl'),
    ('Czech', 'cs'),
    ('Romanian', 'ro'),
    ('Turkish', 'tr'),
    ('Japanese', 'ja'),
    ('Korean', 'ko'),
    ('Chinese', 'zh'),
]

LANGUAGE_CODES = [code for _, code in SUPPORTED_LANGUAGES]

# Standard Whisper checkpoint sizes that support every language.
_MULTILINGUAL_SIZES = {
    'tiny', 'base', 'small', 'medium',
    'large', 'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo', 'turbo',
}

# Used when an English-only model has no direct multilingual counterpart.
_MULTILINGUAL_FALLBACK = 'large-v3'

# Groq now serves only whisper-large-v3 and whisper-large-v3-turbo, both
# multilingual. The distil checkpoint was English-only and has since been
# retired, so configs still naming it are redirected rather than left to 404.
_GROQ_RETIRED = {'distil-whisper-large-v3-en'}
_GROQ_MULTILINGUAL_DEFAULT = 'whisper-large-v3'


def normalize_language(language):
    """
    Return a Whisper language code, or None for auto-detect.

    Both Whisper backends treat ``language=None`` as "detect it yourself", so
    that is what auto-detect maps to.
    """
    if language is None:
        return None
    code = str(language).strip().lower()
    if code in ('', AUTO_DETECT, 'auto-detect', 'none'):
        return None
    return code


def is_english_only(code):
    """True if the resolved language code is English (not auto-detect)."""
    return normalize_language(code) == 'en'


def resolve_local_model(model, language):
    """
    Pick a faster-whisper checkpoint that can actually handle ``language``.

    ``medium.en`` and friends are English-only. Asking them for Russian returns
    confident-looking English nonsense rather than an error, so swap in the
    multilingual variant instead of letting that happen.
    """
    name = (model or '').strip()
    if is_english_only(language):
        return name

    if name.startswith('distil-'):
        # Every distil-whisper checkpoint is an English-only distillation.
        logger.warning(
            "[Language] '%s' is English-only; using '%s' for multilingual transcription.",
            name, _MULTILINGUAL_FALLBACK,
        )
        return _MULTILINGUAL_FALLBACK

    if name.endswith('.en'):
        base = name[:-3]
        resolved = base if base in _MULTILINGUAL_SIZES else _MULTILINGUAL_FALLBACK
        logger.warning(
            "[Language] '%s' is English-only; using '%s' for multilingual transcription.",
            name, resolved,
        )
        return resolved

    return name


def resolve_groq_model(model, language):
    """Pick a Groq Whisper model that exists and can handle ``language``."""
    name = (model or '').strip() or _GROQ_MULTILINGUAL_DEFAULT

    if name in _GROQ_RETIRED:
        # Retired regardless of language, so this is checked before the
        # English-only shortcut below.
        logger.warning(
            "[Language] Groq model '%s' is no longer available; using '%s' instead.",
            name, _GROQ_MULTILINGUAL_DEFAULT,
        )
        return _GROQ_MULTILINGUAL_DEFAULT

    return name


def label_for(code):
    """
    Exact UI label for a language code.

    Unlike :func:`describe`, the result is always one of the labels in
    ``SUPPORTED_LANGUAGES``, so it round-trips through a combobox.
    """
    normalized = normalize_language(code)
    target = AUTO_DETECT if normalized is None else normalized
    for name, value in SUPPORTED_LANGUAGES:
        if value == target:
            return name
    return 'English'


def code_for(label):
    """Inverse of :func:`label_for`; falls back to auto-detect."""
    for name, value in SUPPORTED_LANGUAGES:
        if name == label:
            return value
    return AUTO_DETECT


def describe(code):
    """Human-readable name for a language code, for logs and the UI."""
    normalized = normalize_language(code)
    if normalized is None:
        return 'auto-detect'
    for name, value in SUPPORTED_LANGUAGES:
        if value == normalized:
            return name
    return normalized
