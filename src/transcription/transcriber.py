import logging
import os
import re
import tempfile
import time

import torch
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions
from groq import Groq, RateLimitError
from pydub import AudioSegment
from tqdm import tqdm

from utils.config_manager import DEFAULT_DECODE_OPTIONS, ConfigManager
from utils.languages import (
    describe,
    normalize_language,
    resolve_groq_model,
    resolve_local_model,
)

logger = logging.getLogger(__name__)
config_manager = ConfigManager()

# Groq rejects uploads above 25 MB, so chunks target a little under that.
MAX_UPLOAD_MB = 24
# Bitrate used when exporting chunks; also determines how long a chunk can be.
CHUNK_BITRATE = '128k'
# Only whisper-large-v3 supports the translation endpoint.
TRANSLATION_MODEL = 'whisper-large-v3'

# Decode keys that may be set from `transcription.decode` in config. Anything
# outside this set is dropped rather than forwarded, because model.transcribe
# raises TypeError on unknown keywords - and the set of accepted keywords shifts
# between faster-whisper releases.
SUPPORTED_DECODE_KEYS = frozenset({
    'beam_size',
    'best_of',
    'patience',
    'length_penalty',
    'repetition_penalty',
    'no_repeat_ngram_size',
    'temperature',
    'compression_ratio_threshold',
    'log_prob_threshold',
    'no_speech_threshold',
    'condition_on_previous_text',
    'prompt_reset_on_temperature',
    # Only reaches the first 30s window when condition_on_previous_text is
    # False, since prompt_reset_since is then reset on every window. Prefer
    # `hotwords` for vocabulary that must persist across the whole file.
    'initial_prompt',
    'hotwords',
    'word_timestamps',
    'hallucination_silence_threshold',
    'vad_filter',
    'vad_parameters',
    'max_new_tokens',
    'chunk_length',
})


def create_groq_client():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return Groq(api_key=api_key)


def chunk_duration_ms(max_size_mb=MAX_UPLOAD_MB, bitrate=CHUNK_BITRATE):
    """
    How long a chunk may be before its export exceeds ``max_size_mb``.

    At a constant bitrate, encoded size is a function of duration, so the size
    limit converts directly into a duration limit. 5% is held back for
    container overhead.
    """
    kbps = int(str(bitrate).lower().rstrip('k'))
    bytes_per_ms = (kbps * 1000 / 8) / 1000
    return int((max_size_mb * 1024 * 1024 * 0.95) / bytes_per_ms)


def split_audio(file_path, max_size_mb=MAX_UPLOAD_MB, bitrate=CHUNK_BITRATE):
    """
    Split audio into chunks that stay under the upload size limit.

    Returns a list of ``(chunk_path, offset_seconds)`` pairs. The offset is the
    chunk's absolute position in the source audio, so segment timestamps stay
    correct even if a chunk later fails and is skipped.

    pydub measures and slices ``AudioSegment`` objects in *milliseconds*, so the
    chunk length is derived from the export bitrate rather than from a byte
    count.
    """
    audio = AudioSegment.from_file(file_path)
    chunk_ms = chunk_duration_ms(max_size_mb, bitrate)

    chunks = []
    for start_ms in tqdm(range(0, len(audio), chunk_ms), desc="[Transcription] Splitting audio"):
        segment = audio[start_ms:start_ms + chunk_ms]
        # Close the handle before exporting; writing to an open NamedTemporaryFile
        # fails on Windows.
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
        segment.export(temp_path, format='mp3', bitrate=bitrate)
        chunks.append((temp_path, start_ms / 1000.0))

    logger.info("[Transcription] Split audio into %d chunk(s) of up to %.1f min.",
                len(chunks), chunk_ms / 60000.0)
    return chunks


def parse_wait_time(error_message):
    """
    Extract the wait time in seconds from a rate-limit error message.

    Example: '... Please try again in 2m30.092s. ...' or '... in 30.5s ...'
    """
    match = re.search(r'try again in (?:(\d+)m)?([\d.]+)s', error_message)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return 120  # Default: 2 minutes


def _field(obj, key, default=None):
    """Read a field from a Groq response object, which may be a dict or a model."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _request_transcription(client, chunk_path, model, language, task):
    """Send a single chunk to Groq and return the raw response."""
    with open(chunk_path, 'rb') as audio_file:
        if task == 'translate':
            # Translation always targets English, so it takes no language hint.
            return client.audio.translations.create(
                file=audio_file,
                model=TRANSLATION_MODEL,
                response_format='verbose_json',
                temperature=0.0,
            )

        request = {
            'file': audio_file,
            'model': model,
            'response_format': 'verbose_json',
            'temperature': 0.0,
        }
        # Omitting `language` entirely is what triggers auto-detection.
        if language is not None:
            request['language'] = language
        return client.audio.transcriptions.create(**request)


def transcribe_audio_with_groq(file_path):
    logger.info("[Transcription] Transcribing audio file with Groq: %s", file_path)
    client = create_groq_client()
    config = config_manager.config

    groq_options = config.get('model_options', {}).get('groq', {})
    transcription_options = config.get('transcription', {})
    language = normalize_language(transcription_options.get('language'))
    task = transcription_options.get('task', 'transcribe')

    primary_model = resolve_groq_model(groq_options.get('model', 'whisper-large-v3-turbo'), language)
    fallback_model = resolve_groq_model(groq_options.get('fallback_model', 'whisper-large-v3'), language)

    logger.info("[Transcription] Groq model: %s (language: %s, task: %s)",
                primary_model, describe(language), task)

    file_size = os.path.getsize(file_path)
    if file_size > MAX_UPLOAD_MB * 1024 * 1024:
        chunks = split_audio(file_path)
        owns_chunks = True
    else:
        chunks = [(file_path, 0.0)]
        owns_chunks = False

    max_retries = 5
    transcription_result = []
    active_model = primary_model
    failed_chunks = 0

    try:
        for chunk_path, offset_seconds in chunks:
            model = active_model
            retries = 0
            succeeded = False

            while retries < max_retries and not succeeded:
                try:
                    logger.info("[Transcription] Processing chunk %s with model %s", chunk_path, model)
                    response = _request_transcription(client, chunk_path, model, language, task)

                    # Offsets come from each chunk's own position, so a skipped
                    # chunk cannot shift the timestamps of the ones after it.
                    for segment in (_field(response, 'segments') or []):
                        transcription_result.append({
                            'start': _field(segment, 'start', 0.0) + offset_seconds,
                            'end': _field(segment, 'end', 0.0) + offset_seconds,
                            'text': (_field(segment, 'text', '') or '').strip(),
                        })

                    detected = _field(response, 'language')
                    if detected and language is None:
                        logger.info("[Transcription] Detected language: %s", detected)

                    succeeded = True
                    logger.info("[Transcription] Successfully processed chunk: %s", chunk_path)

                except RateLimitError as e:
                    error_message = str(e)
                    logger.warning("[Transcription] Rate limit hit on %s: %s", model, error_message)

                    if model != fallback_model:
                        logger.info("[Transcription] Switching to fallback model: %s", fallback_model)
                        active_model = fallback_model
                        model = fallback_model
                    else:
                        wait_time = parse_wait_time(error_message)
                        logger.info("[Transcription] Waiting %.2fs before retrying %s", wait_time, model)
                        time.sleep(wait_time)

                    retries += 1

                except Exception as e:
                    logger.error("[Transcription] Unexpected error: %s", e)
                    retries += 1
                    if retries < max_retries:
                        time.sleep(10)

            if not succeeded:
                failed_chunks += 1
                logger.error("[Transcription] Failed to process chunk %s after %d attempts. Skipping.",
                             chunk_path, max_retries)
    finally:
        if owns_chunks:
            for chunk_path, _ in chunks:
                try:
                    os.remove(chunk_path)
                except OSError as e:
                    logger.warning("[Transcription] Could not remove temp chunk %s: %s", chunk_path, e)

    if failed_chunks:
        if not transcription_result:
            raise RuntimeError(f"Groq transcription failed for all {failed_chunks} chunk(s).")
        logger.warning("[Transcription] %d chunk(s) were skipped; the transcript has gaps.", failed_chunks)

    transcription_result.sort(key=lambda item: item['start'])
    logger.info("[Transcription] Groq transcription completed using %s.", active_model)
    return transcription_result


def required_cudnn_library():
    """
    Name of the cuDNN runtime CTranslate2 will try to load.

    CTranslate2 switched from cuDNN 8 to cuDNN 9 in 4.5.0, and torch's CUDA 12
    wheels bundle cuDNN 9, so an older CTranslate2 alongside a modern torch has
    no cuDNN it can use.
    """
    try:
        import ctranslate2
        major, minor = (int(part) for part in ctranslate2.__version__.split('.')[:2])
    except Exception:
        return None
    return 'cudnn_ops64_9.dll' if (major, minor) >= (4, 5) else 'cudnn_ops_infer64_8.dll'


def check_cudnn_available():
    """
    Confirm the cuDNN CTranslate2 needs can actually be loaded.

    CTranslate2 resolves cuDNN lazily, at the first convolution, and exits the
    process outright when it cannot find it - so a mismatch otherwise surfaces
    as a hard crash partway through a long transcription, after the model has
    already downloaded. Checking up front turns that into a clean CPU fallback.

    Windows only: on Linux the libraries usually live in paths ctypes will not
    search even when they work, which would produce false negatives.
    """
    if os.name != 'nt':
        return True, ''

    library = required_cudnn_library()
    if not library:
        return True, ''

    import ctypes
    try:
        ctypes.CDLL(library)
        return True, ''
    except OSError:
        return False, library


def create_local_model(config):
    logger.info("[Transcription] Creating local Whisper model...")
    local_model_options = config['model_options']['local']
    language = normalize_language(config.get('transcription', {}).get('language'))

    # An English-only checkpoint returns plausible-looking nonsense for other
    # languages rather than failing, so swap it out before loading.
    model_name = resolve_local_model(local_model_options['model'], language)

    want_cuda = (
        config['use_cuda']
        and torch.cuda.is_available()
        and local_model_options['device'] != 'cpu'
    )

    if want_cuda:
        usable, missing = check_cudnn_available()
        if not usable:
            logger.error(
                "[Transcription] %s could not be loaded, so GPU transcription is "
                "unavailable. This usually means CTranslate2 and torch disagree "
                "about the cuDNN version. Try: pip install -U 'ctranslate2>=4.5.0,<5'",
                missing,
            )
            logger.warning("[Transcription] Falling back to CPU (slower, but it will finish).")
            want_cuda = False

    if want_cuda:
        device = 'cuda'
        compute_type = local_model_options.get('compute_type', 'float16')
    else:
        device = 'cpu'
        # float16 is not supported on CPU; int8 is the standard CPU choice.
        compute_type = 'int8'

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        logger.info("[Transcription] Local Whisper model (%s) created on %s.", model_name, device.upper())
        return model, device
    except Exception as e:
        logger.error("[Transcription] Error initializing WhisperModel with %s: %s", device.upper(), e)
        raise


def build_decode_options(config):
    """
    Assemble the keyword arguments for ``model.transcribe`` from config.

    Anything the installed faster-whisper does not accept is dropped instead of
    forwarded: ``model.transcribe`` raises ``TypeError`` on an unknown keyword,
    and ``VadOptions`` gained and lost fields between releases (``window_size_samples``
    exists in 1.0.2 but was removed in 1.1), so a config written for one version
    would otherwise break the app outright on another.
    """
    decode_config = config.get('transcription', {}).get('decode')
    if not isinstance(decode_config, dict):
        decode_config = DEFAULT_DECODE_OPTIONS

    options = {}
    for key, value in decode_config.items():
        if key not in SUPPORTED_DECODE_KEYS:
            logger.warning("[Transcription] Ignoring unsupported decode option: %s", key)
            continue
        # None means "leave the library default alone", so it is not forwarded.
        # An empty hotwords string is likewise nothing to inject.
        if value is None or (key in ('hotwords', 'initial_prompt') and not str(value).strip()):
            continue
        options[key] = value

    if 'vad_parameters' in options:
        vad_parameters = options['vad_parameters']
        if not isinstance(vad_parameters, dict):
            # A hand-edited config can put a list or a string here, and
            # faster-whisper only unpacks it much later, failing inside the VAD
            # after the model has already loaded.
            logger.warning("[Transcription] transcription.decode.vad_parameters must be "
                           "an object, got %s; using the defaults.",
                           type(vad_parameters).__name__)
            vad_parameters = DEFAULT_DECODE_OPTIONS['vad_parameters']

        accepted = {}
        for key, value in vad_parameters.items():
            if key not in VadOptions._fields:
                logger.warning("[Transcription] Ignoring VAD option not present in the "
                               "installed faster-whisper: %s", key)
            elif value is None:
                # VadOptions does the arithmetic unguarded, so a JSON null here
                # is a TypeError rather than "use the default".
                logger.warning("[Transcription] Ignoring null VAD option: %s", key)
            else:
                accepted[key] = value
        options['vad_parameters'] = accepted

    # faster-whisper only consults the threshold inside its word-timestamp
    # branch, so without word timestamps it would silently do nothing.
    if options.get('hallucination_silence_threshold') is not None and not options.get('word_timestamps'):
        logger.warning("[Transcription] hallucination_silence_threshold needs "
                       "word_timestamps=True; dropping it.")
        options.pop('hallucination_silence_threshold')

    return options


def _segment_to_dict(segment):
    """
    Convert a faster-whisper segment to the plain dict the pipeline passes around.

    Word timestamps are kept when the decoder produced them: the word_level
    combiner assigns speakers per word, which is the only way to split a Whisper
    segment that contains a speaker change. Everything stays JSON-serializable,
    because DataManager round-trips these dicts through the intermediate cache.

    The key is omitted rather than set to None when there are no words, so
    ``'words' in segment`` is a straight answer to "can this be done per word".
    """
    result = {
        'start': segment.start,
        'end': segment.end,
        'text': segment.text.strip(),
    }

    words = getattr(segment, 'words', None) or []
    converted = []
    for word in words:
        start = getattr(word, 'start', None)
        end = getattr(word, 'end', None)
        text = getattr(word, 'word', None)
        # A word without both timestamps cannot be placed against a diarization
        # turn, and half a word list is worse than none: the combiner treats the
        # list as all-or-nothing per segment.
        if start is None or end is None or text is None:
            logger.debug("[Transcription] Dropping word timestamps for segment at %.2fs: "
                         "incomplete word entry.", segment.start)
            converted = []
            break
        # Word.word carries its own leading space; stripping it here would lose
        # the word boundaries that the combiner rejoins the text with.
        converted.append({'start': start, 'end': end, 'word': text})

    if converted:
        result['words'] = converted

    return result


def transcribe_audio(model, file_path):
    logger.info("[Transcription] Transcribing audio file: %s...", file_path)
    config = config_manager.config
    transcription_options = config.get('transcription', {})
    language = normalize_language(transcription_options.get('language'))
    task = transcription_options.get('task', 'transcribe')
    decode_options = build_decode_options(config)

    logger.info("[Transcription] Language: %s, task: %s", describe(language), task)
    logger.info("[Transcription] Decode options: %s", decode_options)

    # faster-whisper treats language=None as auto-detect.
    segments, info = model.transcribe(file_path, language=language, task=task, **decode_options)
    transcription = [_segment_to_dict(segment) for segment in segments]

    with_words = sum(1 for segment in transcription if 'words' in segment)
    if with_words:
        logger.info("[Transcription] %d of %d segment(s) carry word timestamps.",
                    with_words, len(transcription))

    if language is None and getattr(info, 'language', None):
        logger.info("[Transcription] Detected language: %s (confidence %.2f)",
                    info.language, getattr(info, 'language_probability', 0.0))

    logger.info("[Transcription] Local transcription completed.")
    return transcription


def transcribe_with_fallback(file_path):
    config = config_manager.config
    try:
        return transcribe_audio_with_groq(file_path)
    except Exception as e:
        logger.warning("[Transcription] Groq transcription failed: %s. Falling back to local Whisper.", e)
        model_whisper, _ = create_local_model(config)
        return transcribe_audio(model_whisper, file_path)
