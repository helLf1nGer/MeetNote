import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import logging
from utils.config_manager import ConfigManager
import os
import requests
import base64
from google.auth import default

logger = logging.getLogger(__name__)
config_manager = ConfigManager()

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

    # Perform diarization
    with ProgressHook() as hook:
        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            hook=hook,
            num_speakers=n_speakers
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
                "num_speakers": n_speakers
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