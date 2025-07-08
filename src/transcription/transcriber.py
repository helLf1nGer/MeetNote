import os
import logging
import torch
from groq import Groq, RateLimitError
from faster_whisper import WhisperModel
from pydub import AudioSegment
import tempfile
from tqdm import tqdm
from utils.config_manager import ConfigManager
import time
from datetime import datetime, timedelta
import re

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

def parse_wait_time(error_message):
    """
    Extracts the wait time in seconds from the error message.
    Example message: '... Please try again in 2m30.092s. ...'
    """
    match = re.search(r'Please try again in (\d+)m([\d\.]+)s', error_message)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    else:
        # Default wait time if parsing fails
        return 120  # 2 minutes

def transcribe_audio_with_groq(file_path):
    logger.info(f"[Transcription] Transcribing audio file with Groq: {file_path}")
    client = create_groq_client()
    config = config_manager.config
    
    try:
        file_size = os.path.getsize(file_path)
        chunks = split_audio(file_path) if file_size > 25 * 1024 * 1024 else [file_path]
        
        transcription_result = []
        cumulative_duration = 0
        max_retries = 5
        fallback_model = "distil-whisper-large-v3-en"  # Primary model
        active_model = "whisper-large-v3-turbo"    # Fallback option
        
        # Initialize list of chunks to process
        chunks_to_process = list(chunks)
        processed_chunks = set()
        
        while chunks_to_process:
            chunk = chunks_to_process.pop(0)
            if chunk in processed_chunks:
                continue
            
            chunk_audio = AudioSegment.from_file(chunk)
            chunk_duration = len(chunk_audio) / 1000

            retries = 0
            model = active_model  # Start with primary model
            
            while retries < max_retries:
                try:
                    logger.info(f"[Transcription] Processing chunk: {chunk} with model: {model}")
                    with open(chunk, "rb") as audio_file:
                        transcription = client.audio.transcriptions.create(
                            file=audio_file,
                            model=model,
                            response_format="verbose_json",
                            language=config['transcription']['language'],
                            temperature=0.0
                        )
                    
                    adjusted_segments = [
                        {
                            'start': segment['start'] + cumulative_duration,
                            'end': segment['end'] + cumulative_duration,
                            'text': segment['text'].strip()
                        }
                        for segment in transcription.segments
                    ]
                    transcription_result.extend(adjusted_segments)
                    cumulative_duration += chunk_duration
                    
                    if chunk != file_path:
                        os.remove(chunk)  # Clean up temporary file
                    processed_chunks.add(chunk)
                    logger.info(f"[Transcription] Successfully processed chunk: {chunk}")
                    break  # Exit retry loop on success
                except RateLimitError as e:
                    error_message = str(e)
                    logger.warning(f"[Transcription] Rate limit hit on {model}: {error_message}")
                    
                    if model == active_model:
                        # Switch to fallback model for subsequent chunks
                        logger.info(f"[Transcription] Switching to fallback model: {fallback_model}")
                        active_model = fallback_model
                        model = fallback_model  # Retry current chunk with fallback
                    else:
                        # If already on fallback, just wait
                        wait_time = parse_wait_time(error_message)
                        logger.info(f"[Transcription] Waiting {wait_time:.2f}s before retrying {model}")
                        time.sleep(wait_time)
                    
                    retries += 1
                    logger.info(f"[Transcription] Retrying chunk: {chunk} (Attempt {retries}/{max_retries})")
                except Exception as e:
                    logger.error(f"[Transcription] Unexpected error: {str(e)}")
                    # Optionally, implement additional error handling or retries
                    retries += 1
                    time.sleep(10)  # Wait before retrying
                    logger.info(f"[Transcription] Retrying chunk due to unexpected error: {chunk} (Attempt {retries}/{max_retries})")
            
            if retries == max_retries:
                logger.error(f"[Transcription] Failed to process chunk: {chunk} after {max_retries} attempts. Skipping...")
        
        logger.info(f"[Transcription] Groq transcription completed using {active_model}.")
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
