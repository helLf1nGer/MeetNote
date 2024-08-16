# Output Generator Documentation

## Overview

The Output Generator is responsible for creating the final PDF document containing the transcribed and diarized audio content. It takes the combined results of transcription and diarization and formats them into a readable PDF document.

## Key Features

1. **PDF Generation**: Uses the FPDF library to create PDF documents.
2. **Custom Font Support**: Ability to use custom fonts for better Unicode support.
3. **Configurable Output**: Uses settings from the Config Manager for customization.
4. **Automatic Directory Creation**: Ensures the output directory exists before saving the PDF.

## Implementation Details

### Main Function: create_pdf

```python
def create_pdf(final_transcription, original_file_path):
    config = config_manager.config
    logger.info("[Output] Creating PDF document...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Font setup and content addition...

    pdf_file_name = os.path.join(output_dir, os.path.splitext(os.path.basename(original_file_path))[0] + '_transcription.pdf')
    
    os.makedirs(output_dir, exist_ok=True)
    pdf.output(pdf_file_name)
    logger.info(f"[Output] Transcription PDF saved as {pdf_file_name}")
    return pdf_file_name
```

This function is the core of the Output Generator. It creates a new PDF document, adds the transcribed content, and saves it to the specified output directory.

### Key Components

1. **Font Handling**
   ```python
   font_path, font_name = check_custom_font()
   if font_path and font_name:
       pdf.add_font(font_name, "", font_path, uni=True)
       pdf.set_font(font_name, size=config['pdf_output']['font_size'])
   else:
       pdf.set_font("Arial", size=config['pdf_output']['font_size'])
   ```
   The generator attempts to use a custom font (typically for better Unicode support) if available. Otherwise, it falls back to Arial.

2. **Content Addition**
   ```python
   for item in final_transcription:
       pdf.multi_cell(0, config['pdf_output']['line_spacing'] * config['pdf_output']['font_size'], 
                      f"Speaker {item['speaker']}: {item['text']}", align='L', border=0)
       pdf.ln()
   ```
   This loop adds each segment of the transcription to the PDF, formatting it with speaker labels and configurable line spacing.

3. **Custom Font Check**
   ```python
   def check_custom_font():
       root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
       font_dir = os.path.join(root_dir, 'Fonts')
       
       if os.path.exists(font_dir):
           ttf_fonts = [f for f in os.listdir(font_dir) if f.lower().endswith('.ttf')]
           if ttf_fonts:
               chosen_font = sorted(ttf_fonts)[0]
               font_path = os.path.join(font_dir, chosen_font)
               font_name = os.path.splitext(chosen_font)[0]
               return font_path, font_name
       
       return None, None
   ```
   This function checks for the presence of custom TTF fonts in a 'Fonts' directory and returns the path and name of the first font found.

## Usage

The Output Generator is typically called at the end of the transcription and diarization process:

```python
from utils.output_generator import create_pdf

# After transcription and diarization are complete and combined...
output_pdf_path = create_pdf(final_transcription, original_audio_file_path)
print(f"Transcription saved to: {output_pdf_path}")
```

## Customization

The Output Generator uses several configuration options from the Config Manager:

- `output_directory`: Where the PDF will be saved
- `pdf_output.font_size`: The font size for the PDF content
- `pdf_output.line_spacing`: The line spacing for the PDF content

These can be adjusted in the configuration file or through the Config Manager to customize the PDF output.

## Best Practices

1. Ensure that the 'Fonts' directory contains appropriate TTF fonts if you need support for special characters or non-Latin scripts.
2. Use a descriptive naming convention for the output PDFs to easily identify them later.
3. Regularly check and manage the output directory to prevent clutter.

## Potential Enhancements

1. Support for more complex PDF layouts (e.g., multi-column, headers/footers)
2. Options for including metadata (e.g., transcription date, audio duration)
3. Generation of other output formats (e.g., plain text, HTML)
4. Thumbnail generation of the original audio waveform

The Output Generator plays a crucial role in presenting the results of the transcription and diarization process in a readable and professional format.