# GUI Documentation: main_window.py

## Overview

The `main_window.py` file contains the implementation of the graphical user interface (GUI) for the audio transcription and diarization application. It uses the `ttkbootstrap` library, which is an extension of the standard Tkinter library, to create a modern and responsive interface.

## Key Components

### 1. CustomFileBrowser

A custom tree view widget that displays audio and video files in a selected directory.

#### Purpose:
- Provides an easy way for users to browse and select files for processing.
- Displays file metadata like modification date, type, size, and duration.

#### Key Features:
- Sortable columns
- Double-click to select a file
- Custom file duration extraction for both audio and video files

### 2. MainWindow

The main application window that contains all GUI elements and handles user interactions.

#### Purpose:
- Centralizes all GUI components and user interaction logic.
- Manages the application state and user inputs.

#### Key Components:
- File browser and selection controls
- Settings for transcription and diarization
- Progress bar for processing feedback
- Theme selection
- Output directory selection

### 3. Configuration Management

Utilizes a `ConfigManager` to handle application settings persistently.

#### Purpose:
- Saves and loads user preferences (e.g., last used directory, theme)
- Ensures consistent behavior across application restarts

## Integration with main.py

The `main_window.py` file is designed to work closely with `main.py`. Here's how they interact:

1. **Initialization**: `main.py` calls `create_gui()` from `main_window.py` to set up the GUI.

2. **Event Loop**: `main.py` starts the Tkinter event loop and periodically checks if the user has initiated the processing.

3. **Processing Trigger**: When the user clicks "Start Processing", `main_window.py` sets a flag (`process_started`) and prepares the processing parameters.

4. **Data Handoff**: `main.py` retrieves the processing parameters from the GUI using `window.get_process_result()`.

5. **Progress Updates**: During processing, `main.py` calls `window.update_progress(value)` to update the progress bar.

## Key Design Decisions

1. **File Browser**: A custom file browser was implemented to provide a more user-friendly and feature-rich file selection experience compared to standard file dialogs.

2. **Theming**: The application supports multiple themes to cater to user preferences and potentially reduce eye strain in different lighting conditions.

3. **Adaptive Layout**: The window size adjusts based on the content, ensuring all elements are visible regardless of the file list size.

4. **Progress Feedback**: A progress bar was included to provide real-time feedback during the potentially long-running transcription and diarization processes.

5. **Configuration Persistence**: User settings are saved and loaded to provide a consistent experience across sessions.

6. **Separation of Concerns**: The GUI is designed to be largely independent of the processing logic, allowing for easier maintenance and potential future updates to either component.

## Usage Flow

1. The user selects an audio or video file using the file browser.
2. They configure the transcription and diarization settings.
3. The user clicks "Start Processing".
4. The GUI disables the start button and displays a progress bar.
5. `main.py` takes over, performing the transcription and diarization.
6. Progress updates are sent back to the GUI.
7. Once complete, the GUI is closed, and the results are saved to the specified output directory.

## Extensibility

The modular design of the GUI allows for easy additions of new features:
- New settings can be added to the settings frame.
- Additional file types can be supported by extending the CustomFileBrowser class.
- New themes can be easily added to the theme selection dropdown.

This design ensures that the application can evolve with new requirements while maintaining a consistent and user-friendly interface.