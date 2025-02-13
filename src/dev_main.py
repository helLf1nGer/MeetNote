import os
import logging
from dotenv import load_dotenv
import time
from audio.file_processor import process_file
from transcription.transcriber import transcribe_audio_with_groq, create_local_model, transcribe_audio
from diarization.diarizer import diarize_audio
from utils.data_manager import DataManager
from utils.combiner_testing import test_combiners
from utils.config_manager import ConfigManager
from gui.dev_main_window import create_dev_gui

from pyannote.audio import Pipeline

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging(output_directory):
    log_file = os.path.join(output_directory, 'combiner_test_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

config_manager = ConfigManager()
data_manager = DataManager()

def main():
    try:
        # Get user input through development GUI
        user_input = create_dev_gui()
        if user_input is None:
            print("User cancelled the operation or closed the GUI without starting the process.")
            return

        file_path = user_input['file_path']
        num_speakers = user_input['num_speakers']
        pipeline_model = f"pyannote/{user_input['diarization_model']}"
        transcription_method = user_input['transcription_method']
        output_directory = user_input['output_directory']

        # Setup logging
        log_file = setup_logging(output_directory)
        logging.info(f"Logging to file: {log_file}")

        # Generate a unique file_id based on the input file name
        file_id = os.path.splitext(os.path.basename(file_path))[0]

        # Check if results already exist
        if data_manager.results_exist(file_id):
            transcription, diarization = data_manager.load_results(file_id)
        else:
            # Process the input file
            processed_file = process_file(file_path)

            # Initialize pipeline
            hugging_face_token = os.getenv('HUGGING_FACE_AUTH_TOKEN')
            if not hugging_face_token:
                raise ValueError("HUGGING_FACE_AUTH_TOKEN not found in environment variables")
            pipeline = Pipeline.from_pretrained(pipeline_model, use_auth_token=hugging_face_token)

            # Perform diarization
            diarization, _ = diarize_audio(pipeline, processed_file, num_speakers)

            # Perform transcription
            if transcription_method == 'groq':
                transcription = transcribe_audio_with_groq(processed_file)
            else:
                model_whisper, _ = create_local_model(config_manager.config)
                transcription = transcribe_audio(model_whisper, processed_file)

            # Save the results
            data_manager.save_results(file_id, transcription, diarization)

        # Test all combiners
        test_results = test_combiners(transcription, diarization, pipeline_model, output_directory)

        # Display results
        display_test_results(test_results, output_directory)

        logging.info("Testing completed.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

def display_test_results(test_results, output_directory):
    logging.info("Combiner testing completed. Results saved in the following directory:")
    logging.info(output_directory)
    logging.info("The following files have been generated:")
    logging.info("- Individual JSON files for each combiner's results")
    logging.info("- combiner_comparison.json: Detailed comparison of all combiners")
    logging.info("- combiner_comparison.csv: Summary comparison in CSV format")
    logging.info("- combiner_visualization.png: Visual comparison of speaker segments")
    logging.info(f"- combiner_test_log.txt: Complete log of the testing process")

if __name__ == "__main__":
    main()