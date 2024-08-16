import os
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import torch
import time
from audio.file_processor import process_file
from transcription.transcriber import transcribe_audio_with_groq, create_local_model, transcribe_audio
from diarization.diarizer import diarize_audio
from utils.result_combiner import combine_transcription_diarization
from utils.output_generator import create_pdf
from utils.config_manager import ConfigManager
from gui.main_window import create_gui

from pyannote.audio import Pipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config_manager = ConfigManager()

def update_progress(window, progress):
    window.update_progress(progress)

def main():
    try:
        # Create GUI
        window, root = create_gui()
        
        # Start GUI main loop and periodically check if process has started
        root.after(100, lambda: check_process_start(window, root))
        root.mainloop()
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

def check_process_start(window, root):
    if window.process_started:
        # Process has started, begin actual processing
        process(window, root)
    else:
        # Check again after 100ms
        root.after(100, lambda: check_process_start(window, root))

def process(window, root):
    try:
        # Get user input through GUI
        user_input = window.get_process_result()
        if user_input is None:
            logging.info("User cancelled the operation or closed the GUI without starting the process.")
            return

        start_time = time.time()
        file_path = user_input['file_path']
        num_speakers = user_input['num_speakers']
        pipeline_model = f"pyannote/{user_input['diarization_model']}"
        transcription_method = user_input['transcription_method']
        output_directory = user_input['output_directory']

        # Update config with the new output directory
        config = config_manager.config
        config['output_directory'] = output_directory
        config_manager.save_config()

        # Process the input file
        update_progress(window, 10)
        processed_file = process_file(file_path)

        # Initialize pipeline
        hugging_face_token = os.getenv('HUGGING_FACE_AUTH_TOKEN')
        if not hugging_face_token:
            raise ValueError("HUGGING_FACE_AUTH_TOKEN not found in environment variables")
        pipeline = Pipeline.from_pretrained(pipeline_model, use_auth_token=hugging_face_token)

        update_progress(window, 20)

        with ThreadPoolExecutor(max_workers=2) as executor:
            if transcription_method == 'groq':
                # Run diarization and Groq transcription concurrently
                diarization_future = executor.submit(diarize_audio, pipeline, processed_file, num_speakers)
                transcription_future = executor.submit(transcribe_audio_with_groq, processed_file)

                # Wait for both tasks to complete
                diarization, diarization_device = diarization_future.result()
                transcription = transcription_future.result()

                print(f"\nDiarization was performed on: {diarization_device.upper()}")
                print("Transcription was performed using Groq API.")
            else:
                # For local transcription, keep the sequential process
                diarization, diarization_device = diarize_audio(pipeline, processed_file, num_speakers)
                update_progress(window, 40)
                model_whisper, whisper_device = create_local_model(config)
                transcription = transcribe_audio(model_whisper, processed_file)

                print(f"\nDiarization was performed on: {diarization_device.upper()}")
                print(f"Transcription was performed on: {whisper_device.upper()}")
                print(f"Using local model: {config['model_options']['local']['model']}")

        # Clear CUDA cache after diarization if GPU was used
        if config['use_cuda'] and torch.cuda.is_available():
            torch.cuda.empty_cache()

        update_progress(window, 60)

        # Combine transcription and diarization information
        final_transcription = combine_transcription_diarization(transcription, diarization, pipeline_model)

        update_progress(window, 80)

        # Create PDF from the final transcription
        output_pdf = create_pdf(final_transcription, file_path)

        update_progress(window, 100)

        # Print final confirmation, transcription text, and elapsed time
        print_results(final_transcription, output_pdf, start_time)

        # Close the GUI
        root.quit()

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

def print_results(final_transcription, output_pdf, start_time):
    logging.info(f"Transcription PDF saved as {output_pdf}")
    if config_manager.config['misc']['print_to_terminal']:
        print("\nTranscription Output:")
        for item in final_transcription:
            if 'start' in item and 'end' in item:
                print(f"Speaker {item['speaker']} ({item['start']:.2f} - {item['end']:.2f}): {item['text']}")
            else:
                print(f"Speaker {item['speaker']}: {item['text']}")
    logging.info(f"Script executed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()