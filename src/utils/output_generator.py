import os
import logging
from fpdf import FPDF
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)
config_manager = ConfigManager()

def create_pdf(final_transcription, original_file_path):
    config = config_manager.config
    logger.info("[Output] Creating PDF document...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Check for custom font
    font_path, font_name = check_custom_font()
    if font_path and font_name:
        pdf.add_font(font_name, "", font_path, uni=True)
        pdf.set_font(font_name, size=config['pdf_output']['font_size'])
        logger.info(f"[Output] Using custom font: {font_name} ({font_path})")
    else:
        pdf.set_font("Arial", size=config['pdf_output']['font_size'])
        logger.info("[Output] Using system font: Arial")

    # Add transcription with speaker info to the PDF
    for item in final_transcription:
        pdf.multi_cell(0, config['pdf_output']['line_spacing'] * config['pdf_output']['font_size'], 
                       f"Speaker {item['speaker']}: {item['text']}", align='L', border=0)
        pdf.ln()

    # Construct the PDF file name based on the original file name
    output_dir = config['output_directory']
    pdf_file_name = os.path.join(output_dir, os.path.splitext(os.path.basename(original_file_path))[0] + '_transcription.pdf')
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the PDF to a file
    pdf.output(pdf_file_name)
    logger.info(f"[Output] Transcription PDF saved as {pdf_file_name}")
    return pdf_file_name

def check_custom_font():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    font_dir = os.path.join(root_dir, 'Fonts')
    
    if os.path.exists(font_dir):
        ttf_fonts = [f for f in os.listdir(font_dir) if f.lower().endswith('.ttf')]
        if ttf_fonts:
            # Sort the fonts alphabetically and choose the first one
            chosen_font = sorted(ttf_fonts)[0]
            font_path = os.path.join(font_dir, chosen_font)
            font_name = os.path.splitext(chosen_font)[0]
            return font_path, font_name
    
    return None, None