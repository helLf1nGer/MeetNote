import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import logging
from utils.config_manager import ConfigManager
from .colab_service import ColabGPUService

logger = logging.getLogger(__name__)
config_manager = ConfigManager()

def diarize_audio(pipeline, file_path, n_speakers, processing_location):
    config = config_manager.config
    logger.info(f"[Diarization] Starting diarization for file: {file_path}")
    
    try:
        if processing_location == 'colab_gpu':
            # Use Colab GPU service
            if not config.get('COLAB_SERVICE_URL'):
                raise ValueError("COLAB_SERVICE_URL not found in config")
            
            colab_service = ColabGPUService()
            waveform, sample_rate = torchaudio.load(file_path)
            audio_data = {
                'waveform': waveform.numpy().tolist(),
                'sample_rate': sample_rate
            }
            diarization_results = colab_service.diarize_with_colab(audio_data, n_speakers)
            return diarization_results, 'colab-gpu'
            
        else:
            # Local processing (GPU or CPU)
            waveform, sample_rate = torchaudio.load(file_path)
            
            if config['use_cuda'] and torch.cuda.is_available():
                device = torch.device("cuda:0")
                torch.cuda.empty_cache()
                waveform = waveform.to(device)
                pipeline = pipeline.to(device)
                logger.info(f"[Diarization] Using local CUDA: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.warning("[Diarization] Using CPU for diarization")

            # Perform diarization
            with ProgressHook() as hook:
                diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, 
                                   hook=hook, num_speakers=n_speakers)
            
            used_device = 'cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu'
            
            diarization_results = [{'start': turn.start, 'end': turn.end, 'speaker': speaker} 
                                 for turn, _, speaker in diarization.itertracks(yield_label=True)]
            
            return diarization_results, used_device
            
    except Exception as e:
        logger.error(f"[Diarization] Error during diarization: {str(e)}")
        raise