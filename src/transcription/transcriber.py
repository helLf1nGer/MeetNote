import os
import logging
import torch
from groq import Groq
from faster_whisper import WhisperModel
from pydub import AudioSegment
import tempfile
from tqdm import tqdm
from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)
config_manager = ConfigManager()

def create_groq_client():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return Groq(api_key=api_key)

def split_audio(file_path, max_size_mb=24):
    audio = AudioSegment.from_file(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024
    chunks = []
    
    for i in tqdm(range(0, len(audio), max_size_bytes // 32), desc="[Transcription] Splitting audio"):
        chunk = audio[i:i + max_size_bytes // 32]
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            chunk.export(temp_file.name, format="mp3")
            chunks.append(temp_file.name)
    
    return chunks

def transcribe_audio_with_groq(file_path):
    logger.info(f"[Transcription] Transcribing audio file with Groq: {file_path}")
    client = create_groq_client()
    config = config_manager.config
    
    try:
        file_size = os.path.getsize(file_path)
        if file_size > 25 * 1024 * 1024:  # If file is larger than 25 MB
            chunks = split_audio(file_path)
            transcription_result = []
            for chunk in tqdm(chunks, desc="[Transcription] Processing chunks"):
                with open(chunk, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        file=audio_file,
                        model=config['model_options']['groq']['model'],
                        response_format="verbose_json",
                        language=config['transcription']['language'],
                        temperature=0.0
                    )
                transcription_result.extend([
                    {'start': segment['start'], 'end': segment['end'], 'text': segment['text'].strip()}
                    for segment in transcription.segments
                ])
                os.remove(chunk)  # Clean up temporary file
        else:
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model=config['model_options']['groq']['model'],
                    response_format="verbose_json",
                    language=config['transcription']['language'],
                    temperature=0.0
                )
            transcription_result = [
                {'start': segment['start'], 'end': segment['end'], 'text': segment['text'].strip()}
                for segment in transcription.segments
            ]
        
        logger.info(f"[Transcription] Groq transcription completed successfully.")
        return transcription_result
    except Exception as e:
        logger.error(f"[Transcription] Error during Groq transcription: {str(e)}")
        raise

def create_local_model(config):
    logger.info("[Transcription] Creating local Whisper model...")
    local_model_options = config['model_options']['local']
    if config['use_cuda'] and torch.cuda.is_available() and local_model_options['device'] != 'cpu':
        device = "cuda"
    else:
        device = "cpu"
    
    try:
        model = WhisperModel(local_model_options['model'],
                             device=device,
                             compute_type=local_model_options['compute_type'])
        logger.info(f"[Transcription] Local Whisper model ({local_model_options['model']}) created with {device.upper()}.")
        return model, device
    except Exception as e:
        logger.error(f'[Transcription] Error initializing WhisperModel with {device.upper()}: {e}')
        raise

def transcribe_audio(model, file_path):
    logger.info(f"[Transcription] Transcribing audio file: {file_path}...")
    config = config_manager.config
    segments, info = model.transcribe(file_path, 
                                      language=config['transcription']['language'],
                                      task=config['transcription']['task'])
    transcription = [{'start': segment.start, 'end': segment.end, 'text': segment.text} for segment in segments]
    logger.info("[Transcription] Local transcription completed.")
    return transcription

def transcribe_with_fallback(file_path):
    config = config_manager.config
    try:
        return transcribe_audio_with_groq(file_path)
    except Exception as e:
        logger.warning(f"[Transcription] Groq transcription failed: {str(e)}. Falling back to local Whisper.")
        model_whisper, _ = create_local_model(config)
        return transcribe_audio(model_whisper, file_path)