"""
Transcription and diarization pipeline.

This module holds the actual processing work and knows nothing about Tkinter, so
it can run on a background thread (or headless) without touching widgets. The
GUI supplies a ``progress_callback`` and marshals updates back to the UI thread
itself.
"""

import gc
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from pyannote.audio import Pipeline

from audio.file_processor import process_file
from diarization.diarizer import diarize_audio, speaker_cache_identity
from transcription.transcriber import (
    create_local_model,
    transcribe_audio,
    transcribe_audio_with_groq,
)
from utils.config_manager import ConfigManager
from utils.data_manager import DataManager, cache_key
from utils.languages import describe, normalize_language
from utils.output_generator import create_pdf
from utils.result_combiner import combine_transcription_diarization
from utils.transcription_tracker import TranscriptionTracker

logger = logging.getLogger(__name__)
config_manager = ConfigManager()

# Fallback when diarization.{segmentation,embedding}_batch_size is missing or
# unusable. Kept in step with the config default in utils/config_manager.py.
DEFAULT_BATCH_SIZE = 16


def run_pipeline(settings, progress_callback=None):
    """
    Transcribe and diarize a media file, then write a PDF.

    ``settings`` is the dict produced by the GUI's settings panel, and must
    contain at least ``file_path``. Returns a summary dict with the output path,
    the combined transcript, the elapsed time and any non-fatal warnings.
    """
    def report(percent, message=''):
        if message:
            logger.info("[Pipeline] %s", message)
        if progress_callback:
            progress_callback(percent, message)

    start_time = time.time()

    file_path = settings['file_path']
    num_speakers = settings.get('num_speakers', 2)
    diarization_model = settings.get('diarization_model', 'speaker-diarization-3.1')
    transcription_method = settings.get('transcription_method', 'local')
    processing_location = settings.get('processing_location', 'local')
    output_directory = settings.get('output_directory')

    config = config_manager.config

    # Persist the choices this run was launched with so the next launch and the
    # modules that read config directly (transcriber, combiners) agree.
    if output_directory:
        config['output_directory'] = output_directory
    if settings.get('language'):
        config.setdefault('transcription', {})['language'] = settings['language']
    if settings.get('combiner_method'):
        config.setdefault('combiner', {})['method'] = settings['combiner_method']
    config_manager.save_config()

    language = normalize_language(config.get('transcription', {}).get('language'))
    logger.info("[Pipeline] Processing %s (language: %s, method: %s)",
                file_path, describe(language), transcription_method)

    # Everything that changes what the transcript should contain goes into the
    # key, so switching language or backend never reuses a stale result.
    cache = DataManager()
    key = cache_key(
        file_path,
        method=transcription_method,
        language=language,
        task=config.get('transcription', {}).get('task', 'transcribe'),
        # Not num_speakers itself: 0 means auto, and what auto resolves to comes
        # from config, so a widened min/max range must invalidate the entry.
        speakers=speaker_cache_identity(num_speakers, config),
        diarizer=diarization_model,
        location=processing_location,
        model=config.get('model_options', {}).get('local', {}).get('model'),
    )

    processed_file = None
    warnings = []
    try:
        report(10, 'Preparing audio')
        processed_file = process_file(file_path)

        # A previous run may already have paid for the expensive part and then
        # failed on something cheap downstream, such as an unwritable output
        # directory on a cloud drive.
        transcription, diarization = (None, None)
        if cache.results_exist(key):
            transcription, diarization = cache.load_results(key)

        if transcription is not None and diarization is not None:
            report(70, 'Reusing transcription saved by an earlier run')
            logger.info("[Pipeline] Cached results reused; skipping transcription and diarization.")
            diarization_device = 'cached'
        else:
            hugging_face_token = (
                os.getenv('HUGGING_FACE_AUTH_TOKEN') or config.get('hugging_face_auth_token')
            )
            if not hugging_face_token:
                raise ValueError(
                    "HUGGING_FACE_AUTH_TOKEN not found in environment variables or config."
                )

            transcription, diarization, diarization_device = _transcribe_and_diarize(
                processed_file=processed_file,
                config=config,
                report=report,
                transcription_method=transcription_method,
                diarization_model=diarization_model,
                num_speakers=num_speakers,
                processing_location=processing_location,
                hugging_face_token=hugging_face_token,
            )
            # Written to disk before anything downstream gets a chance to fail.
            cache.save_results(key, transcription, diarization)
            cache.prune()

        if not transcription:
            raise RuntimeError(
                "Transcription produced no segments. The audio may be silent, "
                "corrupt, or contain no detectable speech."
            )
        if not diarization:
            warnings.append(
                "Diarization found no speaker turns, so speaker labels may be missing or uniform."
            )

        report(75, 'Combining results')
        final_transcription = combine_transcription_diarization(
            transcription, diarization, f"pyannote/{diarization_model}"
        )

        report(90, 'Writing output')
        output_pdf = create_pdf(final_transcription, file_path)

        TranscriptionTracker().mark_as_transcribed(file_path, output_pdf)
        # The real output is safely on disk, so the intermediate copy can go.
        cache.discard(key)

        elapsed = time.time() - start_time
        report(100, 'Done')
        logger.info("[Pipeline] Finished in %.2fs -> %s", elapsed, output_pdf)

        if config.get('misc', {}).get('print_to_terminal'):
            _print_transcript(final_transcription)

        return {
            'output_pdf': output_pdf,
            'transcription': final_transcription,
            'elapsed': elapsed,
            'diarization_device': diarization_device,
            'warnings': warnings,
        }

    except Exception:
        if cache.results_exist(key):
            logger.warning(
                "[Pipeline] Run failed, but transcription and diarization are saved. "
                "Re-running the same file with the same settings will resume from them."
            )
        raise

    finally:
        # process_file returns a temporary file when it had to extract audio
        # from a video; the original input must be left alone.
        if processed_file and processed_file != file_path and os.path.exists(processed_file):
            try:
                os.remove(processed_file)
                logger.debug("[Pipeline] Removed temporary audio: %s", processed_file)
            except OSError as e:
                logger.warning("[Pipeline] Could not remove temporary audio %s: %s",
                               processed_file, e)


def _cuda_in_use(config):
    return bool(config.get('use_cuda')) and torch.cuda.is_available()


def _move_pipeline_off_gpu(pipeline, config):
    """
    Move the pyannote pipeline's models back to system RAM.

    This, not the ``del`` that follows it at the call site, is the step that
    actually returns VRAM: ``empty_cache`` only releases cached blocks that
    nothing references, so as long as pyannote's segmentation and speaker
    embedding models sit on the device their memory stays claimed.
    """
    if not _cuda_in_use(config):
        return
    try:
        pipeline.to(torch.device('cpu'))
    except Exception as e:
        # Losing the headroom is survivable; failing the run over it is not.
        logger.warning("[Pipeline] Could not move the diarization pipeline to CPU: %s", e)


def _apply_batch_sizes(pipeline, config):
    """
    Raise pyannote's inference batch sizes.

    Both default to 1 in pyannote.audio 3.3.1, so the segmentation and embedding
    models are fed one chunk at a time and the GPU spends most of its time idle
    between kernel launches. ``hasattr`` guards the assignment because these are
    attributes of SpeakerDiarization specifically, and the model dropdown also
    offers pipelines that do not have them.

    Raising these costs VRAM roughly linearly (measured: 137 MB at 1, 860 MB at
    16, 1661 MB at 32), so the default is a compromise rather than the maximum.
    """
    diarization_config = config.get('diarization', {}) or {}
    for attribute in ('segmentation_batch_size', 'embedding_batch_size'):
        if not hasattr(pipeline, attribute):
            logger.debug("[Pipeline] %s has no %s; leaving it alone.",
                         type(pipeline).__name__, attribute)
            continue
        try:
            size = int(diarization_config.get(attribute, DEFAULT_BATCH_SIZE))
        except (TypeError, ValueError):
            logger.warning("[Pipeline] diarization.%s is not an integer; using %d.",
                           attribute, DEFAULT_BATCH_SIZE)
            size = DEFAULT_BATCH_SIZE
        if size < 1:
            logger.warning("[Pipeline] diarization.%s must be at least 1; using %d.",
                           attribute, DEFAULT_BATCH_SIZE)
            size = DEFAULT_BATCH_SIZE
        try:
            setattr(pipeline, attribute, size)
        except Exception as e:
            # Throughput is worth having, but not at the cost of the run.
            logger.warning("[Pipeline] Could not set %s: %s", attribute, e)
        else:
            logger.info("[Pipeline] %s = %d", attribute, size)


def _log_peak_vram():
    """
    Report torch's peak device allocation so headroom stops being guesswork.

    Two things this deliberately does not cover: faster-whisper, since
    CTranslate2 allocates outside torch's accounting entirely, and the combiners,
    which load after this runs. In practice it measures diarization.
    """
    if not torch.cuda.is_available():
        return
    logger.info("[Pipeline] Peak torch VRAM through transcription: %.0f MB",
                torch.cuda.max_memory_allocated() / (1024 * 1024))


def _transcribe_and_diarize(*, processed_file, config, report, transcription_method,
                            diarization_model, num_speakers, processing_location,
                            hugging_face_token):
    """Run the two expensive stages. Returns (transcription, diarization, device)."""
    if _cuda_in_use(config):
        # Without this the peak below is the process-lifetime peak, so in a
        # long-running GUI every run after the first reports the largest earlier
        # run's figure instead of its own.
        torch.cuda.reset_peak_memory_stats()

    report(20, 'Loading diarization pipeline')
    pipeline = Pipeline.from_pretrained(
        f"pyannote/{diarization_model}", use_auth_token=hugging_face_token
    )
    _apply_batch_sizes(pipeline, config)

    if transcription_method == 'groq':
        # Groq transcription is network-bound and diarization is local, so the
        # two overlap cleanly.
        report(30, 'Diarizing and transcribing')
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                diarization_future = executor.submit(
                    diarize_audio, pipeline, processed_file, num_speakers, processing_location
                )
                transcription_future = executor.submit(
                    transcribe_audio_with_groq, processed_file
                )
                # Collected in a way that surfaces whichever fails first rather
                # than blocking on diarization while a failed transcription sits
                # unread.
                diarization, diarization_device = diarization_future.result()
                transcription = transcription_future.result()
        finally:
            # Outside the `with`, so the executor has already joined: moving a
            # model a worker thread is still running would fail. In a finally so
            # that a failed run gives the VRAM back too - the propagating
            # traceback keeps this frame, and with it the pipeline, alive.
            _move_pipeline_off_gpu(pipeline, config)
        logger.info("[Pipeline] Diarization ran on: %s", diarization_device.upper())
        logger.info("[Pipeline] Transcription ran via Groq API.")
    else:
        # Local transcription competes with diarization for the same GPU, so it
        # stays sequential.
        report(30, 'Diarizing speakers')
        diarization, diarization_device = diarize_audio(
            pipeline, processed_file, num_speakers, processing_location
        )
        logger.info("[Pipeline] Diarization ran on: %s", diarization_device.upper())

        # Whisper is about to be loaded onto the same device, so evict pyannote's
        # segmentation and speaker-embedding weights first. Measured on a 6 GB
        # RTX 3060: ~33 MB allocated / ~30 MB reserved, so this is hygiene rather
        # than what decides whether large-v3 fits. The ~2 GB of inference
        # activations that dominate the peak are unreferenced by this point and
        # were already being returned by empty_cache alone.
        _move_pipeline_off_gpu(pipeline, config)
        del pipeline
        gc.collect()
        if _cuda_in_use(config):
            torch.cuda.empty_cache()

        report(55, 'Transcribing audio')
        model_whisper, whisper_device = create_local_model(config)
        transcription = transcribe_audio(model_whisper, processed_file)
        logger.info("[Pipeline] Transcription ran on: %s", whisper_device.upper())

    _log_peak_vram()

    # Still worth doing on the Groq path, where the pipeline was only just moved
    # off the device, and before the combiner loads a model of its own.
    if _cuda_in_use(config):
        gc.collect()
        torch.cuda.empty_cache()

    return transcription, diarization, diarization_device


def _print_transcript(final_transcription):
    print("\nTranscription Output:")
    for item in final_transcription:
        speaker = item.get('speaker', 'Unknown')
        if 'start' in item and 'end' in item:
            print(f"{speaker} ({item['start']:.2f} - {item['end']:.2f}): {item['text']}")
        else:
            print(f"{speaker}: {item['text']}")
