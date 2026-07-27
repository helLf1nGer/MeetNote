import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import logging
from utils.config_manager import ConfigManager
import os
import requests
import base64

# google.auth is deliberately NOT imported here. It is only needed by
# diarize_cloud, which imports it locally, and a module-level import would make
# the whole pipeline fail to load when google-auth is absent - including for
# purely local diarization, which does not use it at all.

logger = logging.getLogger(__name__)
config_manager = ConfigManager()

# What pyannote's segmentation and embedding models both run at, so anything
# else is resampled internally regardless of what we hand them.
TARGET_SAMPLE_RATE = 16000


def _coerce_speaker_count(n_speakers):
    """
    Normalise a speaker count to an int, treating anything unusable as auto.

    ``default_num_speakers`` is written by the GUI as an int, but a hand-edited
    config can hold ``"2"`` or worse, and comparing a str to an int raises
    TypeError rather than falling back. Auto is the safe interpretation: it is
    also what a user who cleared the field most likely wanted.
    """
    if n_speakers is None:
        return 0
    try:
        return int(n_speakers)
    except (TypeError, ValueError):
        logger.warning("[Diarization] Speaker count %r is not a number; "
                       "letting pyannote decide.", n_speakers)
        return 0


def resolve_speaker_args(n_speakers, config):
    """
    Decide what to tell pyannote about the speaker count.

    ``n_speakers`` of 0 or less means "let pyannote work it out", which is the
    safer default: forcing the count too low makes AgglomerativeClustering merge
    genuinely different speakers, and the turn boundaries that destroys are gone
    before the combiner ever sees them. Forcing it too high only splits one voice
    across several labels, which a reader can still repair by hand.

    Returns the keyword arguments for ``pipeline(...)`` - either ``num_speakers``
    alone, or the ``min_speakers``/``max_speakers`` range.
    """
    n_speakers = _coerce_speaker_count(n_speakers)
    if n_speakers > 0:
        return {'num_speakers': n_speakers}

    diarization_config = config.get('diarization', {}) or {}
    try:
        minimum = int(diarization_config.get('min_speakers', 1))
        maximum = int(diarization_config.get('max_speakers', 10))
    except (TypeError, ValueError):
        logger.warning("[Diarization] min_speakers/max_speakers are not integers; "
                       "using 1-10.")
        minimum, maximum = 1, 10

    # pyannote raises outright on an inverted range, so a hand-edited config
    # would otherwise fail the run several minutes in.
    if minimum < 1 or maximum < minimum:
        logger.warning("[Diarization] Ignoring invalid speaker range %s-%s; using 1-10.",
                       minimum, maximum)
        minimum, maximum = 1, 10

    return {'min_speakers': minimum, 'max_speakers': maximum}


def speaker_cache_identity(n_speakers, config):
    """
    A stable string describing what the speaker count will actually request.

    Auto mode resolves to a range from config, so the range - not the 0 that
    stands for it - is what determines the diarization. Without it, widening
    max_speakers reuses the cached result from the narrower range.

    A forced count returns its plain number so cache entries written before auto
    mode existed stay valid.
    """
    args = resolve_speaker_args(n_speakers, config)
    if 'num_speakers' in args:
        return str(args['num_speakers'])
    return "auto:%d-%d" % (args['min_speakers'], args['max_speakers'])


def prepare_waveform(waveform, sample_rate, target_sample_rate=TARGET_SAMPLE_RATE,
                     file_path=None):
    """
    Downmix to mono and resample, returning ``(waveform, sample_rate)``.

    Meant to run before the tensor is moved to the GPU. pyannote does this
    conversion itself, but only after it has the audio, so handing it raw
    44.1 kHz stereo means transferring roughly 5.5x more data than it will use -
    about 1.27 GB of VRAM for a 60-minute meeting, on a card where that is a
    fifth of the total.

    Raises ValueError when there are no samples: a zero-length file otherwise
    reaches resample and fails with "cannot reshape tensor of 0 elements", which
    says nothing about the actual problem.
    """
    if waveform.numel() == 0 or waveform.shape[-1] == 0:
        raise ValueError(
            "Audio file contains no samples: %s" % (file_path or 'unknown file')
        )

    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)

    # (channel, time) -> (1, time). pyannote rejects anything that is not 2-D.
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)

    return waveform, target_sample_rate


def diarize_audio(pipeline, file_path, n_speakers, processing_location):
    """
    Main diarization function that routes to appropriate processing method
    based on processing_location parameter.
    """
    logger.info(f"[Diarization] Starting diarization for file: {file_path}")
    
    try:
        if processing_location == 'local':
            return diarize_local(pipeline, file_path, n_speakers)
        elif processing_location == 'cloud':
            return diarize_cloud(pipeline, file_path, n_speakers)
        else:
            raise ValueError(f"Unknown processing location: {processing_location}")
            
    except Exception as e:
        logger.error(f"[Diarization] Error during diarization: {str(e)}")
        raise

def diarize_local(pipeline, file_path, n_speakers):
    """Handle local diarization processing (CPU or GPU)"""
    config = config_manager.config

    # Load audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Downmixed and resampled while still on the CPU, so only the audio pyannote
    # actually consumes crosses onto the GPU.
    original_shape, original_rate = tuple(waveform.shape), sample_rate
    waveform, sample_rate = prepare_waveform(waveform, sample_rate, file_path=file_path)
    if (tuple(waveform.shape), sample_rate) != (original_shape, original_rate):
        logger.info("[Diarization] Prepared audio: %s @ %dHz -> %s @ %dHz",
                    original_shape, original_rate, tuple(waveform.shape), sample_rate)

    # Set up device (GPU if available and enabled, otherwise CPU)
    if config['use_cuda'] and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        waveform = waveform.to(device)
        pipeline = pipeline.to(device)
        logger.info(f"[Diarization] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("[Diarization] Using CPU for diarization")

    speaker_args = resolve_speaker_args(n_speakers, config)
    logger.info("[Diarization] Speaker count: %s",
                speaker_args.get('num_speakers')
                or "auto (%s-%s)" % (speaker_args['min_speakers'], speaker_args['max_speakers']))

    # Perform diarization
    with ProgressHook() as hook:
        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            hook=hook,
            **speaker_args
        )

    # Get device used for reporting
    used_device = 'cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu'
    
    # Format results
    diarization_results = [
        {
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        }
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    
    return diarization_results, used_device

def resolve_cloud_speaker_count(n_speakers, config):
    """
    Pick the ``num_speakers`` to send to the Vertex AI endpoint.

    The payload schema here is ``num_speakers`` only - there is no min/max range
    field, and the serving code is not in this repository, so an auto request
    cannot be expressed. Sending 0 verbatim is the one thing that is certainly
    wrong: it either errors or asks for zero speakers.

    So auto falls back to the configured ``default_num_speakers`` (2 if unset)
    and says so, rather than omitting the field. Omitting it would be a guess
    about a server we cannot inspect - if it is required, the request fails and
    the run silently drops to local diarization, which is a confusing way to
    learn that cloud mode does not support auto.
    """
    resolved = _coerce_speaker_count(n_speakers)
    if resolved > 0:
        return resolved

    diarization_config = config.get('diarization', {}) or {}
    fallback = _coerce_speaker_count(diarization_config.get('default_num_speakers', 2))
    if fallback <= 0:
        # The configured default is itself auto, so there is nothing to fall
        # back to and a number still has to be sent.
        fallback = 2

    logger.warning("[Diarization] Cloud diarization cannot auto-detect the speaker "
                   "count; sending %d. Use local processing for auto.", fallback)
    return fallback


def diarize_cloud(pipeline, file_path, n_speakers):
    """
    Handle cloud-based diarization processing using Vertex AI endpoint.
    Falls back to local processing if cloud request fails.
    """
    config = config_manager.config
    cloud_config = get_cloud_provider()
    
    logger.info(f"[Diarization] Attempting cloud processing with GCP Vertex AI endpoint: {cloud_config['endpoint_id']}")
    
    try:
        # Get cloud credentials
        from google.auth import default
        credentials, project_id = default()
        
        # Build endpoint URL
        endpoint_url = (f"https://{cloud_config['region']}-aiplatform.googleapis.com/v1/"
                       f"projects/{cloud_config['project_id']}/locations/{cloud_config['region']}/"
                       f"endpoints/{cloud_config['endpoint_id']}:predict")
        
        # Prepare audio file
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        # Build request payload
        payload = {
            "instances": [{
                "file": {
                    "b64": base64.b64encode(audio_bytes).decode("utf-8")
                },
                "num_speakers": resolve_cloud_speaker_count(n_speakers, config)
            }]
        }
        
        # Get access token
        from google.auth.transport.requests import Request
        credentials.refresh(Request())
        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }
        
        # Send request to Vertex AI endpoint
        response = requests.post(endpoint_url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Process response
        predictions = response.json().get("predictions", [])
        if not predictions:
            raise ValueError("No predictions returned from cloud endpoint")
        
        # Convert to standard format
        diarization_results = []
        for segment in predictions[0].get("segments", []):
            diarization_results.append({
                "start": segment["start"],
                "end": segment["end"],
                "speaker": segment["speaker"]
            })
            
        return diarization_results, "gcp"
        
    except Exception as cloud_error:
        logger.error(f"[Diarization] Cloud processing failed: {str(cloud_error)}")
        logger.warning("[Diarization] Falling back to local processing")
        
        try:
            # Attempt local processing as fallback
            local_results, device = diarize_local(pipeline, file_path, n_speakers)
            return local_results, f"local-fallback ({device})"
        except Exception as local_error:
            logger.error(f"[Diarization] Local fallback failed: {str(local_error)}")
            raise RuntimeError("Both cloud and local processing failed") from local_error

def get_cloud_provider():
    """
    Get the configured cloud provider details.
    This can be expanded as more providers are added.
    """
    config = config_manager.config
    cloud_config = config.get('cloud_provider', {})
    
    return {
        'provider': cloud_config.get('provider', 'gcp'),
        'project_id': cloud_config.get('project_id'),
        'region': cloud_config.get('region', 'us-central1'),
        'endpoint_id': cloud_config.get('endpoint_id'),
    }