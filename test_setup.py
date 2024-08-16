import os
import sys
import tkinter as tk
from dotenv import load_dotenv
from groq import Groq
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from pydub import AudioSegment
import torch
from src.utils.config_manager import ConfigManager

def test_env_variables():
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')
    huggingface_token = os.getenv('HUGGING_FACE_AUTH_TOKEN')
    
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in .env file")
    else:
        print("✅ GROQ_API_KEY found in .env file")
    
    if not huggingface_token:
        print("❌ HUGGING_FACE_AUTH_TOKEN not found in .env file")
    else:
        print("✅ HUGGING_FACE_AUTH_TOKEN found in .env file")

def test_config():
    try:
        config = ConfigManager().config
        print("✅ Config file loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config file: {str(e)}")

def test_gui():
    try:
        root = tk.Tk()
        root.title("Test GUI")
        root.geometry("300x200")
        label = tk.Label(root, text="GUI Test Successful!")
        label.pack(expand=True)
        root.after(2000, root.destroy)  # Close after 2 seconds
        root.mainloop()
        print("✅ GUI initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing GUI: {str(e)}")

def test_audio_processing():
    try:
        # Create a silent audio segment for testing
        silent_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
        print("✅ Audio processing libraries working")
    except Exception as e:
        print(f"❌ Error with audio processing: {str(e)}")

def test_groq_connection():
    try:
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, World!"}],
            model="mixtral-8x7b-32768",
        )
        print("✅ Groq API connection successful")
    except Exception as e:
        print(f"❌ Error connecting to Groq API: {str(e)}")

def test_huggingface_connection():
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                            use_auth_token=os.getenv('HUGGING_FACE_AUTH_TOKEN'))
        print("✅ HuggingFace connection successful")
    except Exception as e:
        print(f"❌ Error connecting to HuggingFace: {str(e)}")

def test_whisper_model():
    try:
        model = WhisperModel("tiny")
        print("✅ Whisper model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading Whisper model: {str(e)}")

def test_cuda_support():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA support available with {device_count} GPU(s)")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {device_name}")
    else:
        print("❌ CUDA support not available")

if __name__ == "__main__":
    print("Running MeetNote setup tests...")
    test_env_variables()
    test_config()
    test_gui()
    test_audio_processing()
    test_groq_connection()
    test_huggingface_connection()
    test_whisper_model()
    test_cuda_support()
    print("Setup tests completed.")