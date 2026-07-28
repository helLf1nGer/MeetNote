"""
Main Window Module for Audio Transcription & Diarization Application.

This module provides the GUI interface for the application, allowing users to:
- Browse and select audio/video files
- Configure transcription and diarization settings
- Process media files and view results
"""

import os
import sys
import time
import logging
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox

import cv2
from mutagen import File as MutagenFile

from audio.file_processor import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from utils.config_manager import (
    DEFAULT_OUTPUT_FORMATS,
    SUPPORTED_OUTPUT_FORMATS,
    ConfigManager,
)
from utils.languages import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    code_for,
    label_for,
)
from utils.transcription_tracker import TranscriptionTracker

# Logging is configured centrally in utils.logging_setup, called from main.
# A basicConfig call here would be a no-op whenever main ran first, and would
# silently take over the root logger when it did not.
logger = logging.getLogger(__name__)

# Initialize configuration manager
config_manager = ConfigManager()

# Choices for the browser's type filter. An empty value means no restriction.
# Built from audio.file_processor so a format added there shows up here without
# a second edit, then followed by the individual extensions actually present -
# picking one specific container is the common case when a directory holds
# camera video next to voice recordings.
# The one format that is always written: create_pdf returns its path and the
# transcription tracker records it, so deselecting it would break both.
PRIMARY_OUTPUT_FORMAT = 'pdf'

FILTER_ALL = 'All supported'
FILTER_AUDIO = 'Audio only'
FILTER_VIDEO = 'Video only'

FILE_TYPE_FILTERS = {
    FILTER_ALL: (),
    FILTER_AUDIO: tuple(sorted(AUDIO_EXTENSIONS)),
    FILTER_VIDEO: tuple(sorted(VIDEO_EXTENSIONS)),
    **{ext: (ext,) for ext in sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)},
}


class MediaFile:
    """Model class representing a media file with its properties."""
    
    # Taken from audio.file_processor rather than restated here. The two lists
    # had drifted apart: .opus, .wma, .aiff, .m4v, .mpg, .mpeg and .wmv all
    # process correctly but were invisible in the browser, so the only way to
    # transcribe one was the command line.
    SUPPORTED_EXTENSIONS = {
        'audio': sorted(AUDIO_EXTENSIONS),
        'video': sorted(VIDEO_EXTENSIONS),
    }
    
    def __init__(self, path: str):
        """Initialize a MediaFile object."""
        self.path = path
        self.name = os.path.basename(path)
        self.extension = os.path.splitext(self.name)[1].lower()
        self.stats = os.stat(path)
        self.size_mb = self.stats.st_size / (1024 * 1024)
        self.modified_date = time.strftime('%Y-%m-%d %H:%M:%S', 
                                          time.localtime(self.stats.st_mtime))
        self._duration = None
        self._transcription_history = None
        self._is_transcribed = None
    
    @property
    def duration(self) -> str:
        """Get the duration of the media file."""
        if self._duration is not None:
            return self._duration
            
        try:
            # Try to get duration from audio metadata
            audio = MutagenFile(self.path)
            if hasattr(audio, 'info') and hasattr(audio.info, 'length'):
                self._duration = self._format_duration(audio.info.length)
                return self._duration
                
            # If not available, try to get from video
            video = cv2.VideoCapture(self.path)
            if not video.isOpened():
                self._duration = "N/A"
                return self._duration
                
            fps = video.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                self._duration = "N/A"
                return self._duration
                
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = frame_count / fps
            video.release()
            
            self._duration = self._format_duration(duration_seconds)
            return self._duration
        except Exception as e:
            logger.error(f"Error getting duration for {self.path}: {e}")
            self._duration = "N/A"
            return self._duration
    
    def _format_duration(self, seconds: float) -> str:
        """Format seconds into a human-readable duration string."""
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    @property
    def transcription_history(self) -> List[Dict]:
        """Get the transcription history for this file."""
        if self._transcription_history is None:
            tracker = TranscriptionTracker()
            self._transcription_history = tracker.get_transcription_history(self.path)
        return self._transcription_history
    
    @property
    def is_transcribed(self) -> bool:
        """Check if this file has been transcribed."""
        if self._is_transcribed is None:
            tracker = TranscriptionTracker()
            self._is_transcribed = tracker.is_transcribed(self.path)
        return self._is_transcribed
    
    @property
    def transcription_count(self) -> int:
        """Get the number of times this file has been transcribed."""
        return len(self.transcription_history)
    
    @property
    def status(self) -> str:
        """Get the transcription status of this file."""
        count = self.transcription_count
        if count > 0:
            return f"✓ Transcribed ({count}x)"
        return "Not Transcribed"
    
    @property
    def file_type(self) -> str:
        """Get the file type."""
        return self.extension
    
    @property
    def size_formatted(self) -> str:
        """Get the formatted file size."""
        return f"{self.size_mb:.2f} MB"
    
    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Check if a file is a supported media file."""
        ext = os.path.splitext(file_path)[1].lower()
        all_extensions = MediaFile.SUPPORTED_EXTENSIONS['audio'] + MediaFile.SUPPORTED_EXTENSIONS['video']
        return ext in all_extensions


class FileBrowser(ttk.Treeview):
    """Custom file browser for media files with sorting and filtering capabilities."""
    
    def __init__(self, parent, main_window, *args, **kwargs):
        """Initialize the file browser."""
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.main_window = main_window
        self.tracker = TranscriptionTracker()
        
        # Configure columns
        self["columns"] = ("Date", "Type", "Size", "Duration", "Status", "Count")
        
        # Configure headings with sorting
        self.heading("#0", text="Name", anchor=tk.W, 
                    command=lambda: self.sort_column("#0", False))
        self.heading("Date", text="Date Modified", anchor=tk.W, 
                    command=lambda: self.sort_column("Date", False))
        self.heading("Type", text="Type", anchor=tk.W, 
                    command=lambda: self.sort_column("Type", False))
        self.heading("Size", text="Size", anchor=tk.W, 
                    command=lambda: self.sort_column("Size", False))
        self.heading("Duration", text="Duration", anchor=tk.W, 
                    command=lambda: self.sort_column("Duration", False))
        self.heading("Status", text="Status", anchor=tk.W, 
                    command=lambda: self.sort_column("Status", False))
        self.heading("Count", text="Times Transcribed", anchor=tk.W,
                    command=lambda: self.sort_column("Count", False))
        
        # Configure tags
        self.tag_configure('transcribed', foreground='green')
        
        # Initialize variables
        self.directory_path = None
        self.media_files = {}  # Cache for media files
        self.file_loading_thread = None
        self.loading_queue = queue.Queue()
        # Extensions currently shown. None means "everything supported"; the
        # filter narrows the view only, so it never hides a file the pipeline
        # could not process anyway.
        self.extension_filter = None
        
        # Bind events
        self.bind("<Double-1>", self.on_double_click)
        self.bind("<Return>", lambda e: self.main_window.select_file())
        self.bind("<<TreeviewSelect>>", lambda e: self.main_window.select_file())
        
        # Create right-click context menu
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Open File", 
                                     command=self.open_selected_file)
        self.context_menu.add_command(label="Open Containing Folder", 
                                     command=self.open_containing_folder)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Copy Path", 
                                     command=self.copy_path_to_clipboard)
        self.bind("<Button-3>", self.show_context_menu)
        
        # Start the queue processing
        self.process_loading_queue()
    
    def on_double_click(self, event):
        """Handle double-click event on a file."""
        item = self.identify('item', event.x, event.y)
        if item:
            self.selection_set(item)
            self.main_window.select_file()
    
    def show_context_menu(self, event):
        """Show the context menu on right-click."""
        item = self.identify('item', event.x, event.y)
        if item:
            self.selection_set(item)
            self.main_window.select_file()
            self.context_menu.tk_popup(event.x_root, event.y_root)
    
    def open_selected_file(self):
        """Open the selected file with the default application."""
        selected_file = self.get_selected_file()
        if selected_file:
            if sys.platform == 'win32':
                os.startfile(selected_file)
            elif sys.platform == 'darwin':
                os.system(f'open "{selected_file}"')
            else:
                os.system(f'xdg-open "{selected_file}"')
    
    def open_containing_folder(self):
        """Open the folder containing the selected file."""
        selected_file = self.get_selected_file()
        if selected_file:
            folder = os.path.dirname(selected_file)
            if sys.platform == 'win32':
                os.startfile(folder)
            elif sys.platform == 'darwin':
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')
    
    def copy_path_to_clipboard(self):
        """Copy the path of the selected file to clipboard."""
        selected_file = self.get_selected_file()
        if selected_file:
            self.clipboard_clear()
            self.clipboard_append(selected_file)
            self.update()
    
    def populate(self, directory_path: str):
        """Populate the file browser with media files from the given directory."""
        if not os.path.isdir(directory_path):
            logger.error(f"Invalid directory path: {directory_path}")
            return
        
        self.directory_path = directory_path
        self.delete(*self.get_children())
        
        # Show loading indicator
        self.main_window.set_status(f"Loading files from {directory_path}...")
        
        # Clear the queue
        while not self.loading_queue.empty():
            try:
                self.loading_queue.get_nowait()
                self.loading_queue.task_done()
            except queue.Empty:
                break
        
        # Stop any existing loading thread
        if self.file_loading_thread and self.file_loading_thread.is_alive():
            self.file_loading_thread = None
        
        # Load files directly for small directories (faster response)
        try:
            files = [f for f in os.listdir(directory_path)
                    if os.path.isfile(os.path.join(directory_path, f))
                    and MediaFile.is_supported(os.path.join(directory_path, f))
                    and self._passes_filter(f)]
            
            # If we have a small number of files, load them directly
            if len(files) < 20:
                for file in files:
                    full_path = os.path.join(directory_path, file)
                    media_file = MediaFile(full_path)
                    self.media_files[full_path] = media_file
                    self._add_file_to_treeview(media_file)
                
                self.main_window.set_status(f"Loaded {len(files)} files")
                self.main_window.adjust_column_widths()
                return
        except Exception as e:
            logger.error(f"Error during direct file loading: {e}")
        
        # For larger directories, use the threaded approach
        self.file_loading_thread = threading.Thread(
            target=self._load_files_async,
            args=(directory_path,),
            daemon=True
        )
        self.file_loading_thread.start()
    
    def _load_files_async(self, directory_path: str):
        """Load media files asynchronously."""
        try:
            # Clear existing media files cache for this directory
            self.media_files = {}
            
            # Get all files in the directory
            files = os.listdir(directory_path)
            logger.info(f"Found {len(files)} items in {directory_path}")
            
            for item in files:
                if self.file_loading_thread is None:
                    return
                
                full_path = os.path.join(directory_path, item)
                if os.path.isfile(full_path):
                    # Check if it's a supported media file
                    ext = os.path.splitext(item)[1].lower()
                    all_extensions = MediaFile.SUPPORTED_EXTENSIONS['audio'] + MediaFile.SUPPORTED_EXTENSIONS['video']

                    if ext in all_extensions and self._passes_filter(item):
                        # Create a MediaFile object
                        media_file = MediaFile(full_path)
                        self.media_files[full_path] = media_file
                        
                        # Queue the file for display
                        self.loading_queue.put(media_file)
                        logger.debug(f"Queued file: {full_path}")
            
            # Signal that loading is complete
            self.loading_queue.put(None)
            logger.info(f"Finished loading {len(self.media_files)} media files")
        except Exception as e:
            logger.error(f"Error loading files: {e}")
            self.loading_queue.put(None)
    
    def process_loading_queue(self):
        """Process the loading queue and update the UI."""
        processed_files = 0
        try:
            # Process up to 10 files at a time to keep UI responsive
            for _ in range(10):
                if self.loading_queue.empty():
                    break
                
                media_file = self.loading_queue.get_nowait()
                if media_file is None:
                    # Loading is complete
                    self.main_window.set_status(f"Ready - {len(self.get_children())} files loaded")
                    self.main_window.adjust_column_widths()
                    # Continue processing in case there are more files
                    self.after(50, self.process_loading_queue)
                    return
                
                # Add the file to the treeview
                self._add_file_to_treeview(media_file)
                processed_files += 1
                
                # Mark the task as done
                self.loading_queue.task_done()
        except queue.Empty:
            # Queue is empty but we'll continue processing
            pass
        
        # Update status if we processed files
        if processed_files > 0:
            self.main_window.set_status(f"Loading files... ({len(self.get_children())} so far)")
        
        # Always schedule the next processing
        self.after(50, self.process_loading_queue)
    
    def _add_file_to_treeview(self, media_file: MediaFile):
        """Add a media file to the treeview."""
        values = (
            media_file.modified_date,
            media_file.file_type,
            media_file.size_formatted,
            media_file.duration,
            media_file.status,
            media_file.transcription_count
        )
        
        tags = ('transcribed',) if media_file.is_transcribed else ()
        
        self.insert("", tk.END, text=media_file.name, values=values, tags=tags)
    
    def sort_column(self, column: str, reverse: bool):
        """Sort the treeview by the given column."""
        # Get all items
        items = [(self.item(k)["text"] if column == "#0" else self.set(k, column), k) 
                for k in self.get_children('')]
        
        # Custom sorting for size column
        if column == "Size":
            # Extract numeric value from size string (e.g., "10.5 MB" -> 10.5)
            items = [(float(i[0].split()[0]) if i[0] != "N/A" else 0, i[1]) for i in items]
        # Custom sorting for duration column
        elif column == "Duration":
            # Convert duration to seconds for sorting
            def duration_to_seconds(duration):
                if duration == "N/A":
                    return 0
                parts = duration.split(":")
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                return 0
            
            items = [(duration_to_seconds(i[0]), i[1]) for i in items]
        # Custom sorting for count column
        elif column == "Count":
            items = [(int(i[0]) if i[0].isdigit() else 0, i[1]) for i in items]
        else:
            # Default string sorting
            items = [(i[0].lower() if isinstance(i[0], str) else i[0], i[1]) for i in items]
        
        # Sort the items
        items.sort(reverse=reverse)
        
        # Rearrange items in the sorted positions
        for index, (_, k) in enumerate(items):
            self.move(k, '', index)
        
        # Reverse the sort order for the next click
        self.heading(column, command=lambda: self.sort_column(column, not reverse))
    
    def set_extension_filter(self, extensions):
        """
        Restrict the view to ``extensions``, or show everything when falsy.

        Re-reads the directory rather than hiding rows in place: Treeview has no
        real hidden state (detach/reattach loses ordering), and repopulating
        keeps one code path for what is on screen.
        """
        self.extension_filter = {e.lower() for e in extensions} if extensions else None
        if self.directory_path:
            self.populate(self.directory_path)

    def _passes_filter(self, file_name: str) -> bool:
        """Whether a supported file also passes the active extension filter."""
        if not self.extension_filter:
            return True
        return os.path.splitext(file_name)[1].lower() in self.extension_filter

    def get_selected_file(self) -> Optional[str]:
        """Get the path of the selected file."""
        selected_items = self.selection()
        if not selected_items:
            return None
        
        item = self.item(selected_items[0])
        return os.path.join(self.directory_path, item['text'])
    
    def get_selected_media_file(self) -> Optional[MediaFile]:
        """Get the MediaFile object for the selected file."""
        file_path = self.get_selected_file()
        if not file_path:
            return None
        
        if file_path in self.media_files:
            return self.media_files[file_path]
        
        # If not in cache, create a new MediaFile object
        media_file = MediaFile(file_path)
        self.media_files[file_path] = media_file
        return media_file


class SettingsPanel(ttk.Frame):
    """Panel for configuring transcription and diarization settings."""
    
    def __init__(self, parent, main_window, *args, **kwargs):
        """Initialize the settings panel."""
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.main_window = main_window
        self.config = config_manager.config
        
        # Create variables
        self.num_speakers = ttk.IntVar(value=self.config.get('diarization', {}).get('default_num_speakers', 2))
        self.diarization_model = ttk.StringVar(value='speaker-diarization-3.1')
        self.transcription_method = ttk.StringVar(value=self.config.get('transcription_method', 'groq'))
        self.output_directory = ttk.StringVar(value=self.config.get('output_directory', 'transcriptions'))
        self.processing_location = ttk.StringVar(value=self.config.get('processing_location', 'local'))
        self.combiner_method = ttk.StringVar(value=self.config.get('combiner', {}).get('method', 'semantic_adaptive'))
        self.language = ttk.StringVar(
            value=self.config.get('transcription', {}).get('language', DEFAULT_LANGUAGE)
        )

        # One BooleanVar per format. Unknown names in a hand-edited config are
        # ignored here and warned about by output_generator, so a typo cannot
        # make the panel fail to build.
        selected = self.config.get('output', {}).get('formats')
        if not isinstance(selected, (list, tuple)):
            selected = DEFAULT_OUTPUT_FORMATS
        selected = {str(name).strip().lower() for name in selected}
        self.output_formats = {
            fmt: ttk.BooleanVar(
                value=True if fmt == PRIMARY_OUTPUT_FORMAT else fmt in selected
            )
            for fmt in SUPPORTED_OUTPUT_FORMATS
        }

        # Add traces to update config when values change
        self.num_speakers.trace_add('write', self._update_config)
        self.diarization_model.trace_add('write', self._update_config)
        self.transcription_method.trace_add('write', self._update_config)
        self.output_directory.trace_add('write', self._update_config)
        self.processing_location.trace_add('write', self._update_config)
        self.combiner_method.trace_add('write', self._update_config)
        self.language.trace_add('write', self._update_config)
        for variable in self.output_formats.values():
            variable.trace_add('write', self._update_config)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the widgets for the settings panel."""
        # Output Settings
        output_frame = ttk.LabelFrame(self, text="Output Settings", padding="10", bootstyle="primary")
        output_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(output_frame, text='Output Directory:', font=("TkDefaultFont", 10)).pack(fill=X, pady=(0, 5))
        dir_frame = ttk.Frame(output_frame)
        dir_frame.pack(fill=X)
        ttk.Entry(dir_frame, textvariable=self.output_directory).pack(side=LEFT, fill=X, expand=YES, padx=(0, 5))
        ttk.Button(dir_frame, text="📁", command=self._browse_output_directory,
                  style='primary-outline.TButton', width=3).pack(side=RIGHT)

        # Output formats. The PDF checkbox is shown but disabled: it is the
        # pipeline's return value and what the transcription tracker records, so
        # it is written whether or not it is ticked. Showing it greyed is more
        # honest than a list that silently ignores one entry.
        ttk.Label(output_frame, text='Also write:',
                  font=("TkDefaultFont", 10)).pack(fill=X, pady=(10, 5))
        formats_frame = ttk.Frame(output_frame)
        formats_frame.pack(fill=X)

        for fmt in SUPPORTED_OUTPUT_FORMATS:
            always_on = fmt == PRIMARY_OUTPUT_FORMAT
            label = f"{fmt} (always)" if always_on else fmt
            ttk.Checkbutton(
                formats_frame, text=label, variable=self.output_formats[fmt],
                state=DISABLED if always_on else NORMAL,
                bootstyle="primary-round-toggle" if not always_on else "secondary",
            ).pack(side=LEFT, padx=(0, 10))
        
        # Diarization Settings
        diar_frame = ttk.LabelFrame(self, text="Diarization Settings", padding="10", bootstyle="primary")
        diar_frame.pack(fill=X, pady=(0, 10))
        
        # Basic diarization settings
        speaker_frame = ttk.Frame(diar_frame)
        speaker_frame.pack(fill=X, pady=(0, 5))
        # 0 is spelled out in the label rather than displayed as "Auto": the
        # Spinbox `format` option is printf-style and so cannot substitute a
        # word for a number, and swapping in a StringVar would mean every reader
        # of self.num_speakers having to parse it.
        ttk.Label(speaker_frame, text='Number of Speakers (0 = Auto):',
                  font=("TkDefaultFont", 10)).pack(side=LEFT)
        ttk.Spinbox(speaker_frame, from_=0, to=10, textvariable=self.num_speakers, width=5).pack(side=LEFT, padx=(5, 0))
        
        # Advanced diarization settings
        advanced_frame = ttk.Labelframe(diar_frame, text="Advanced Options", padding="5", bootstyle="secondary")
        advanced_frame.pack(fill=X, pady=(5, 0))
        
        ttk.Label(advanced_frame, text='Diarization Model:', font=("TkDefaultFont", 10)).pack(fill=X, pady=(0, 5))
        diarization_models = [
            'speaker-diarization-3.1',
            'speaker-diarization-3.0',
            'speech-separation-ami-1.0',
            'segmentation',
            'wespeaker-voxceleb-resnet34-LM'
        ]
        ttk.Combobox(advanced_frame, textvariable=self.diarization_model, 
                    values=diarization_models, state="readonly").pack(fill=X)
        
        # Processing Settings
        proc_frame = ttk.LabelFrame(self, text="Processing Settings", padding="10", bootstyle="primary")
        proc_frame.pack(fill=X)
        
        # Transcription Method
        ttk.Label(proc_frame, text='Transcription Method:', font=("TkDefaultFont", 10)).pack(fill=X, pady=(0, 5))
        transcription_methods = ['local', 'groq']
        ttk.Combobox(proc_frame, textvariable=self.transcription_method,
                    values=transcription_methods, state="readonly").pack(fill=X, pady=(0, 10))

        # Language. The combobox shows names but self.language holds the code.
        ttk.Label(proc_frame, text='Language:', font=("TkDefaultFont", 10)).pack(fill=X, pady=(0, 5))
        self.language_display = ttk.StringVar(value=label_for(self.language.get()))
        ttk.Combobox(proc_frame, textvariable=self.language_display,
                    values=[name for name, _ in SUPPORTED_LANGUAGES],
                    state="readonly").pack(fill=X, pady=(0, 10))
        self.language_display.trace_add('write', self._on_language_selected)

        # Combiner Method. Ordered by measured assignment accuracy: word_level
        # splits segments at mid-segment speaker changes (needs word timestamps,
        # falls back to weighted without them); weighted is the segment-level
        # default that passed every regression case.
        ttk.Label(proc_frame, text='Combiner Method:', font=("TkDefaultFont", 10)).pack(fill=X, pady=(0, 5))
        combiner_methods = [
            'word_level',
            'weighted',
            'simple',
            'semantic_flow',
            'semantic',
            'semantic_enhanced',
            'semantic_adaptive',
            'two_stage_llm',
            'groq_llm',
            'adaptive',
            'adaptive_rule',
            'local_llama_tiny'
        ]
        ttk.Combobox(proc_frame, textvariable=self.combiner_method, 
                    values=combiner_methods, state="readonly").pack(fill=X, pady=(0, 10))
        
        # Processing Location
        ttk.Label(proc_frame, text='Diarization Method:', font=("TkDefaultFont", 10)).pack(fill=X, pady=(0, 5))
        processing_locations = ['local', 'cloud']
        ttk.Combobox(proc_frame, textvariable=self.processing_location, 
                    values=processing_locations, state="readonly").pack(fill=X)
        
        # Start button
        self.start_button = ttk.Button(self, text='▶ Start Processing', 
                                      command=self.main_window.start_process, 
                                      style='success.TButton')
        self.start_button.pack(pady=(10, 0), fill=X)
        
        # Progress frame (hidden initially)
        self.progress_frame = ttk.Frame(self)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate', 
                                           bootstyle="success-striped")
        self.progress_bar.pack(fill=X, expand=YES, padx=(0, 10))
        
        self.progress_label = ttk.Label(self.progress_frame, text="0%", font=("TkDefaultFont", 10))
        self.progress_label.pack(side=RIGHT)
    
    def _browse_output_directory(self):
        """Browse for an output directory."""
        directory = filedialog.askdirectory(initialdir=self.output_directory.get())
        if directory:
            self.output_directory.set(directory)
    
    def _on_language_selected(self, *args):
        """Translate the selected display name back into a language code."""
        self.language.set(code_for(self.language_display.get()))

    def _update_config(self, *args):
        """Update the config with current settings."""
        # Update diarization settings
        if 'diarization' not in self.config:
            self.config['diarization'] = {}
        self.config['diarization']['default_num_speakers'] = self.num_speakers.get()
        self.config['diarization']['model'] = self.diarization_model.get()

        # Update other settings
        self.config['transcription_method'] = self.transcription_method.get()
        self.config['output_directory'] = self.output_directory.get()
        self.config['processing_location'] = self.processing_location.get()

        # Update transcription settings
        if 'transcription' not in self.config:
            self.config['transcription'] = {}
        self.config['transcription']['language'] = self.language.get()

        # Update combiner settings
        if 'combiner' not in self.config:
            self.config['combiner'] = {}
        self.config['combiner']['method'] = self.combiner_method.get()

        # Update output formats, keeping the supported order rather than tick
        # order so the saved config reads the same way every time.
        if 'output' not in self.config:
            self.config['output'] = {}
        self.config['output']['formats'] = [
            fmt for fmt in SUPPORTED_OUTPUT_FORMATS
            if fmt == PRIMARY_OUTPUT_FORMAT or self.output_formats[fmt].get()
        ]

        # Save config
        config_manager.save_config()
    
    def show_progress_bar(self):
        """Show the progress bar."""
        self.progress_frame.pack(fill=X, pady=10, before=self.start_button)
        self.progress_bar['value'] = 0
        self.progress_label['text'] = "0%"
        self.update_idletasks()
    
    def hide_progress_bar(self):
        """Hide the progress bar."""
        self.progress_frame.pack_forget()
    
    def update_progress(self, value: int):
        """Update the progress bar."""
        self.progress_bar['value'] = value
        self.progress_label['text'] = f"{value}%"
        self.update_idletasks()
        
    def enable_start_button(self):
        """Enable the start button."""
        self.start_button.config(state='normal')
        
    def disable_start_button(self):
        """Disable the start button."""
        self.start_button.config(state='disabled')
    
    def get_settings(self) -> Dict[str, Any]:
        """Get the current settings."""
        return {
            'num_speakers': self.num_speakers.get(),
            'diarization_model': self.diarization_model.get(),
            'transcription_method': self.transcription_method.get(),
            'output_directory': self.output_directory.get(),
            'processing_location': self.processing_location.get(),
            'combiner_method': self.combiner_method.get(),
            'language': self.language.get()
        }


class StatusBar(ttk.Frame):
    """Status bar for displaying application status."""
    
    def __init__(self, parent, *args, **kwargs):
        """Initialize the status bar."""
        super().__init__(parent, *args, **kwargs)
        
        # Create separator
        ttk.Separator(self).pack(fill=X, pady=(0, 5))
        
        # Create status label
        self.status_label = ttk.Label(self, text="Ready", font=("TkDefaultFont", 9),
                                     bootstyle="secondary")
        self.status_label.pack(side=LEFT)
        
        # Create memory usage label on the right
        self.memory_label = ttk.Label(self, text="", font=("TkDefaultFont", 9),
                                     bootstyle="secondary")
        self.memory_label.pack(side=RIGHT)
        
        # Update memory usage periodically
        self.update_memory_usage()
    
    def set_status(self, text: str):
        """Set the status text."""
        self.status_label.config(text=text)
        self.update_idletasks()
    
    def update_memory_usage(self):
        """Update the memory usage display."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / (1024 * 1024)  # Convert to MB
            self.memory_label.config(text=f"Memory: {memory_usage:.1f} MB")
        except ImportError:
            self.memory_label.config(text="")
        except Exception as e:
            logger.error(f"Error updating memory usage: {e}")
            self.memory_label.config(text="")
        
        # Schedule next update
        self.after(10000, self.update_memory_usage)  # Update every 10 seconds


class MainWindow:
    """Main application window."""
    
    def __init__(self, root):
        """Initialize the main window."""
        self.root = root
        self.root.title('Audio Transcription & Diarization')
        
        # Set the icon and configure the window
        icon_path = os.path.join(os.path.dirname(__file__), '../../Icon/MeetNote.ico')
        try:
            self.root.iconbitmap(icon_path)
        except Exception as e:
            logger.warning(f"Could not set icon: {e}")
        
        self.root.minsize(900, 650)
        
        # Initialize variables
        self.config = config_manager.config
        self.file_path = ttk.StringVar()
        self.theme_var = ttk.StringVar(value=self.config.get('gui_theme', 'darkly'))
        self.process_started = False
        self.process_result = None
        self.processing_thread = None
        
        # Create keyboard shortcuts
        self._create_keyboard_shortcuts()
        
        # Create widgets
        self._create_widgets()
        
        # Initialize the file browser
        self._populate_file_browser()
        
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _create_keyboard_shortcuts(self):
        """Create keyboard shortcuts."""
        self.root.bind("<Control-o>", lambda e: self.browse_directory())
        self.root.bind("<Control-q>", lambda e: self.root.quit())
        self.root.bind("<F5>", lambda e: self._refresh_file_browser())
        self.root.bind("<F1>", lambda e: self._show_help())
    
    def _create_widgets(self):
        """Create the widgets for the main window."""
        # Create main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=BOTH, expand=YES)
        
        # Top frame with header and theme selector
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=X, pady=(0, 20))
        
        # Left side: Title
        header_label = ttk.Label(header_frame, text="Audio Processing Center", 
                                font=("TkDefaultFont", 16, "bold"))
        header_label.pack(side=LEFT)
        
        # Right side: Theme selector
        theme_frame = ttk.Frame(header_frame)
        theme_frame.pack(side=RIGHT)
        ttk.Label(theme_frame, text='Theme:', font=("TkDefaultFont", 10)).pack(side=LEFT, padx=(0, 5))
        themes = ['darkly', 'superhero', 'solar', 'cyborg', 'vapor', 'litera']
        theme_menu = ttk.Combobox(theme_frame, textvariable=self.theme_var, 
                                 values=themes, state="readonly", width=12, 
                                 bootstyle="primary")
        theme_menu.pack(side=LEFT, padx=(0, 5))
        ttk.Button(theme_frame, text="🎨", command=self._change_theme, 
                  style='primary-outline.TButton', width=3).pack(side=LEFT)
        
        # Main content frame with two columns
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=BOTH, expand=YES)
        content_frame.columnconfigure(0, weight=3)  # File list gets more space
        content_frame.columnconfigure(1, weight=1)  # Settings get less space
        
        # Left column: File browsing and list
        file_frame = ttk.Frame(content_frame)
        file_frame.grid(row=0, column=0, sticky=NSEW, padx=(0, 10))
        file_frame.rowconfigure(1, weight=1)  # Make file list expandable
        file_frame.columnconfigure(0, weight=1)
        
        # Browse button with icon
        browse_frame = ttk.Frame(file_frame)
        browse_frame.grid(row=0, column=0, sticky=EW, pady=(0, 10))
        browse_button = ttk.Button(browse_frame, text='📂 Browse Directory', 
                                  command=self.browse_directory, 
                                  style='primary.TButton')
        browse_button.pack(side=LEFT)
        
        refresh_button = ttk.Button(browse_frame, text='🔄 Refresh',
                                   command=self._refresh_file_browser,
                                   style='secondary.TButton')
        refresh_button.pack(side=LEFT, padx=(10, 0))

        # Type filter. The browser lists every format the pipeline accepts,
        # which is a lot of rows in a mixed directory; this narrows the view
        # without ever hiding something that could have been processed.
        ttk.Label(browse_frame, text='Show:').pack(side=LEFT, padx=(15, 5))
        self.type_filter = ttk.StringVar(value=FILTER_ALL)
        filter_box = ttk.Combobox(browse_frame, textvariable=self.type_filter,
                                  values=list(FILE_TYPE_FILTERS), state='readonly',
                                  width=18)
        filter_box.pack(side=LEFT)
        filter_box.bind('<<ComboboxSelected>>', self._apply_type_filter)
        
        # File browser with modern styling
        browser_frame = ttk.LabelFrame(file_frame, text="Media Files", padding="10",
                                      bootstyle="primary")
        browser_frame.grid(row=1, column=0, sticky=NSEW)
        
        self.file_browser = FileBrowser(browser_frame, self)
        self.file_browser.pack(expand=YES, fill=BOTH)
        
        # File info below browser
        self.file_label = ttk.Label(file_frame, text="No file selected",
                                   font=("TkDefaultFont", 12, "bold"))
        self.file_label.grid(row=2, column=0, sticky=W, pady=(10, 0))
        self.file_info = ttk.Label(file_frame, text="")
        self.file_info.grid(row=3, column=0, sticky=W)
        
        # Right column: Settings
        settings_frame = ttk.Frame(content_frame)
        settings_frame.grid(row=0, column=1, sticky=NSEW)
        
        # Create settings panel
        self.settings_panel = SettingsPanel(settings_frame, self)
        self.settings_panel.pack(fill=BOTH, expand=YES)
        
        # Status bar
        self.status_bar = StatusBar(main_frame)
        self.status_bar.pack(fill=X, side=BOTTOM, pady=(10, 0))
    
    def _populate_file_browser(self):
        """Initialize the file browser with the last used directory."""
        if self.config.get("last_directory") and os.path.exists(self.config["last_directory"]):
            directory = self.config["last_directory"]
        else:
            directory = os.path.expanduser("~/Videos")
            self.config["last_directory"] = directory
            config_manager.save_config()
        
        self.file_browser.populate(directory)
    
    def _refresh_file_browser(self):
        """Refresh the file browser with the current directory."""
        if self.file_browser.directory_path:
            self.file_browser.populate(self.file_browser.directory_path)

    def _apply_type_filter(self, event=None):
        """Narrow the browser to the selected group of file types."""
        choice = self.type_filter.get()
        self.file_browser.set_extension_filter(FILE_TYPE_FILTERS.get(choice))
        self.set_status(f"Showing: {choice}")
    
    def browse_directory(self):
        """Browse for a directory containing media files."""
        directory = filedialog.askdirectory(initialdir=self.config.get("last_directory"))
        if directory:
            self.config["last_directory"] = directory
            config_manager.save_config()
            self.file_browser.populate(directory)
    
    def select_file(self):
        """Handle file selection in the browser."""
        selected_file = self.file_browser.get_selected_file()
        if selected_file:
            self.file_path.set(selected_file)
            media_file = self.file_browser.get_selected_media_file()
            
            self.file_label.config(text=f"{media_file.name}")
            self.file_info.config(
                text=f"Size: {media_file.size_formatted} | Modified: {media_file.modified_date} | Duration: {media_file.duration}"
            )
        else:
            self.file_label.config(text="No file selected")
            self.file_info.config(text="")
    
    def start_process(self):
        """Start the transcription and diarization process."""
        selected_file = self.file_browser.get_selected_file()
        if not selected_file:
            Messagebox.show_error('Please select an audio file.', 'Error')
            return
        
        # Get settings from the settings panel
        settings = self.settings_panel.get_settings()
        
        # Show progress bar
        self.settings_panel.show_progress_bar()
        self.settings_panel.disable_start_button()
        
        # Set status
        self.set_status(f"Processing {os.path.basename(selected_file)}...")
        
        # Store process information
        self.process_started = True
        self.process_result = {
            'file_path': selected_file,
            **settings
        }
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(
            target=self._process_file,
            args=(selected_file, settings),
            daemon=True
        )
        self.processing_thread.start()
    
    def _process_file(self, file_path, settings):
        """Run the real transcription pipeline on a worker thread."""
        # Imported here rather than at module scope so that merely opening the
        # GUI does not pull in torch and pyannote.
        from pipeline import run_pipeline

        try:
            result = run_pipeline(
                {**settings, 'file_path': file_path},
                progress_callback=self._update_progress,
            )
            self.root.after(0, self._on_processing_complete, file_path, result)
        except Exception as e:
            logger.exception("Error processing file: %s", file_path)
            self.root.after(0, self._on_processing_error, str(e))

    def _update_progress(self, value, message=''):
        """Update the progress bar from a worker thread."""
        self.root.after(0, self.settings_panel.update_progress, value)
        if message:
            self.root.after(0, self.set_status, message)
    
    def _on_processing_complete(self, file_path, result=None):
        """Handle completion of processing."""
        self.settings_panel.enable_start_button()
        self.settings_panel.hide_progress_bar()
        self.set_status(f"Processing complete: {os.path.basename(file_path)}")

        # Refresh the file browser to show updated status
        self._refresh_file_browser()

        details = f"Successfully processed {os.path.basename(file_path)}"
        if result:
            details += (
                f"\n\nSaved to: {result['output_pdf']}"
                f"\nElapsed: {result['elapsed']:.1f}s"
            )
        Messagebox.show_info(details, "Processing Complete")

    def _on_processing_error(self, error_message):
        """Handle processing error."""
        self.settings_panel.enable_start_button()
        self.settings_panel.hide_progress_bar()
        self.set_status("Error during processing")
        
        # Show error message
        Messagebox.show_error(
            f"An error occurred during processing: {error_message}",
            "Processing Error"
        )
    
    def _change_theme(self):
        """Change the application theme."""
        new_theme = self.theme_var.get()
        if new_theme != self.config.get('gui_theme'):
            self.config['gui_theme'] = new_theme
            config_manager.save_config()
            
            # Apply theme without restarting
            style = ttk.Style()
            style.theme_use(new_theme)
            
            # Show message about restart for full effect
            Messagebox.show_info(
                "Theme partially applied. Restart the application for full effect.",
                "Theme Changed"
            )
    
    def _show_help(self):
        """Show help information."""
        help_text = """
        Audio Transcription & Diarization Tool
        
        Keyboard Shortcuts:
        - Ctrl+O: Browse Directory
        - Ctrl+Q: Quit Application
        - F5: Refresh File List
        - F1: Show This Help
        
        For more information, visit the documentation.
        """
        
        Messagebox.show_info(help_text, "Help")
    
    def set_status(self, text):
        """Set the status bar text."""
        self.status_bar.set_status(text)
        
    def update_progress(self, progress):
        """Update the progress bar with the given value.
        
        This delegates to the settings_panel's update_progress method.
        
        Args:
            progress (int): Progress value between 0 and 100
        """
        if hasattr(self, 'settings_panel'):
            self.settings_panel.update_progress(progress)
        else:
            logging.warning("Cannot update progress: settings_panel not initialized")
    
    def adjust_column_widths(self):
        """Adjust column widths in the file browser based on content."""
        if not hasattr(self, 'file_browser') or not self.file_browser:
            return
            
        # Get all items in the file browser
        items = self.file_browser.get_children()
        if not items:
            return
            
        # Calculate max widths for each column
        max_widths = {
            "#0": 20,  # Name column
            "Date": 20,
            "Type": 10,
            "Size": 15,
            "Duration": 15,
            "Status": 20,
            "Count": 10
        }
        
        # Check each item to find the maximum width needed
        for item_id in items:
            item = self.file_browser.item(item_id)
            text = item['text']
            values = item['values']
            
            # Update max width for name column
            max_widths["#0"] = max(max_widths["#0"], min(len(text), 40))
            
            # Update max width for other columns
            for i, col in enumerate(self.file_browser["columns"]):
                if i < len(values):
                    max_widths[col] = max(max_widths[col], min(len(str(values[i])), 30))
        
        # Apply the calculated widths
        for col in ["#0"] + list(self.file_browser["columns"]):
            self.file_browser.column(col, width=max_widths[col] * 7)
            
        logger.info(f"Adjusted column widths for {len(items)} items")
    
    def _on_close(self):
        """Handle window close event."""
        # Stop any running threads
        if self.processing_thread and self.processing_thread.is_alive():
            # We can't directly stop a thread, but we can ask the user
            # show_question returns the *text* of the button pressed, not a
            # bool. Both "Yes" and "No" are truthy, so testing it directly
            # quit either way - discarding a run that could be an hour in.
            answer = Messagebox.show_question(
                "A transcription is still running.\n\n"
                "Quitting now discards it. Transcription and diarization "
                "results are saved, so re-running the same file will resume "
                "from them.\n\nQuit anyway?",
                "Confirm Exit",
                buttons=["Keep running:secondary", "Quit:danger"],
            )
            if answer == "Quit":
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Start the main loop of the application."""
        self.root.mainloop()
    
    def get_process_result(self):
        """Return the result of the processing if started, otherwise None."""
        if self.process_started:
            return self.process_result
        return None


def create_gui():
    """Create and return the main window and root objects.
    
    Returns:
        Tuple containing the MainWindow instance and the root window
    """
    # Set the theme from config
    theme = config_manager.config.get('gui_theme', 'darkly')
    root = ttk.Window(themename=theme)
    
    # Create the main window
    window = MainWindow(root)
    
    # Return both objects
    return window, root


if __name__ == '__main__':
    # Create the GUI
    window, root = create_gui()
    
    # Run the application
    window.run()
    
    # Get the process result if any
    result = window.get_process_result()
    if result:
        logger.info(f"Process result: {result}")
