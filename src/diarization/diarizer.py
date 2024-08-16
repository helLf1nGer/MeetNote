import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import logging
from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)
config_manager = ConfigManager()

def diarize_audio(pipeline, file_path, n_speakers):
    config = config_manager.config
    logger.info(f"[Diarization] Starting diarization for file: {file_path}")
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        
        if config['use_cuda'] and torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.empty_cache()
            waveform = waveform.to(device)
            logger.info(f"[Diarization] CUDA available: {torch.cuda.is_available()}")
            logger.info(f"[Diarization] Current device: {torch.cuda.current_device()}")
            logger.info(f"[Diarization] Device name: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.warning("[Diarization] CUDA not available or not enabled. Using CPU for diarization.")

        # Perform diarization
        with ProgressHook() as hook:
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, 
                                   hook=hook, num_speakers=n_speakers)
        
        # Remove the device check from here
        used_device = 'cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu'
        
        if used_device == 'cuda':
            logger.info(f"[Diarization] GPU memory allocated after diarization: {torch.cuda.memory_allocated()/1e6:.2f} MB")
        
        diarization_results = [{'start': turn.start, 'end': turn.end, 'speaker': speaker} 
                               for turn, _, speaker in diarization.itertracks(yield_label=True)]

        logger.info(f"[Diarization] Completed successfully on {used_device.upper()}.")
        return diarization_results, used_device
    except Exception as e:
        logger.error(f"[Diarization] Error during diarization: {str(e)}")
        raise