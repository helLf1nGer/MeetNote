import os
import logging
from fpdf import FPDF
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)
config_manager = ConfigManager()

def create_pdf(final_transcription, original_file_path):
    # Reload the config to ensure we have the latest version
    config = config_manager.load_config()
    output_dir = config['output_directory']
    logger.info(f"[Output] Using output directory: {output_dir}")

    logger.info("[Output] Creating PDF document...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Check for custom font
    font_path, font_name = check_custom_font()
    if font_path and font_name:
        try:
            pdf.add_font(font_name, "", font_path, uni=True)
            pdf.set_font(font_name, size=config['pdf_output']['font_size'])
            logger.info(f"[Output] Using custom font: {font_name} ({font_path})")
        except Exception as e:
            logger.error(f"[Output] Failed to add custom font. Falling back to Arial. Error: {e}")
            pdf.set_font("Arial", size=config['pdf_output']['font_size'])
            logger.info("[Output] Using system font: Arial")
    else:
        pdf.set_font("Arial", size=config['pdf_output']['font_size'])
        logger.info("[Output] Using system font: Arial")

    # Add transcription with speaker info to the PDF
    for item in final_transcription:
        text = f"{item['speaker']}: {item['text']}"
        try:
            pdf.multi_cell(
                0, 
                config['pdf_output']['line_spacing'] * config['pdf_output']['font_size'], 
                text, 
                align='L', 
                border=0
            )
            pdf.ln()
        except Exception as e:
            logger.error(f"[Output] Error adding text to PDF: {e}")
            continue

    # Construct the PDF file name based on the original file name
    pdf_file_name = os.path.join(
        output_dir, 
        os.path.splitext(os.path.basename(original_file_path))[0] + '_transcription.pdf'
    )
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the PDF to a file
    try:
        pdf.output(pdf_file_name)
        logger.info(f"[Output] Transcription PDF saved as {pdf_file_name}")
    except Exception as e:
        logger.error(f"[Output] Failed to save PDF. Error: {e}")
        raise

    return pdf_file_name

def check_custom_font():
    """
    Checks for the presence of a Unicode-compatible TTF font in the 'Fonts' directory at the project root.
    Returns the path and name of the first found font.
    """
    # Navigate to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    font_dir = os.path.join(project_root, 'Fonts')
    
    logger.info(f"[Output] Looking for custom fonts in: {font_dir}")

    desired_font = 'DejaVuSans.ttf'
    font_path = os.path.join(font_dir, desired_font)

    if os.path.exists(font_path):
        font_name = os.path.splitext(desired_font)[0]
        logger.info(f"[Output] Found custom font: {font_name} at {font_path}")
        return font_path, font_name
    else:
        logger.warning(f"[Output] Desired font '{desired_font}' not found in Fonts directory.")
    
    return None, None