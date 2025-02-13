import tkinter as tk
from tkinter import filedialog
import os
import time
import sys
import subprocess
from mutagen import File as MutagenFile
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import cv2
from utils.config_manager import ConfigManager
from utils.transcription_tracker import TranscriptionTracker

config_manager = ConfigManager()

class CustomFileBrowser(ttk.Treeview):
    def __init__(self, parent, main_window, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.main_window = main_window
        self.tracker = TranscriptionTracker()
        self["columns"] = ("Date", "Type", "Size", "Duration", "Status", "Count")
        self.heading("#0", text="Name", anchor=tk.W, command=lambda: self.sort_column("#0", False))
        self.heading("Date", text="Date Modified", anchor=tk.W, command=lambda: self.sort_column("Date", False))
        self.heading("Type", text="Type", anchor=tk.W, command=lambda: self.sort_column("Type", False))
        self.heading("Size", text="Size", anchor=tk.W, command=lambda: self.sort_column("Size", False))
        self.heading("Duration", text="Duration", anchor=tk.W, command=lambda: self.sort_column("Duration", False))
        self.heading("Status", text="Status", anchor=tk.W, command=lambda: self.sort_column("Status", False))
        self.heading("Count", text="Times Transcribed", anchor=tk.W)
        
        self.file_path = None
        self.bind("<Double-1>", self.on_double_click)

    def on_double_click(self, event):
        item = self.identify('item', event.x, event.y)
        if item:
            self.selection_set(item)
            self.main_window.select_file()

    def populate(self, path):
        self.delete(*self.get_children())
        max_widths = {"#0": 20, "Date": 20, "Type": 10, "Size": 15, "Duration": 15, "Status": 15, "Count": 15}
        
        def process_files():
            files_data = []
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                if os.path.isfile(full_path):
                    file_type = os.path.splitext(item)[1]
                    if file_type.lower() in ['.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mov', '.mkv', '.flv']:
                        stats = os.stat(full_path)
                        size = f"{stats.st_size / (1024 * 1024):.2f} MB"
                        date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.st_mtime))
                        duration = self.get_duration(full_path)
                        history = self.tracker.get_transcription_history(full_path)
                        count = len(history)
                        status = f"✓ Transcribed ({count}x)" if count > 0 else "Not Transcribed"
                        files_data.append((item, date, file_type, size, duration, status, count, bool(count)))
            return files_data
        
        files_data = process_files()
        
        for item, date, file_type, size, duration, status, count, _ in files_data:
            values = (date, file_type, size, duration, status, count)
            
            tags = ('transcribed',) if self.tracker.is_transcribed(os.path.join(path, item)) else ()
            self.insert("", tk.END, text=item, values=values, tags=tags)
            
            max_widths["#0"] = min(max(max_widths["#0"], len(item)), 40)
            for i, col in enumerate(self["columns"]):
                max_widths[col] = min(max(max_widths[col], len(str(values[i]))), 30)

        for col in ("#0",) + self["columns"]:
            self.column(col, width=max_widths[col]*7)

        self.tag_configure('transcribed', foreground='green')

    def get_duration(self, file_path):
        try:
            audio = MutagenFile(file_path)
            if hasattr(audio.info, 'length'):
                return self.format_duration(audio.info.length)
            
            video = cv2.VideoCapture(file_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            video.release()
            return self.format_duration(duration)
        except Exception:
            return "N/A"

    def format_duration(self, seconds):
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def sort_column(self, column, reverse):
        l = [(self.item(k)["text"] if column == "#0" else self.set(k, column), k) for k in self.get_children('')]
        l.sort(key=lambda t: t[0].lower(), reverse=reverse)
        for index, (_, k) in enumerate(l):
            self.move(k, '', index)
        self.heading(column, command=lambda: self.sort_column(column, not reverse))

    def get_selected_file(self):
        selected_item = self.selection()
        if selected_item:
            item = self.item(selected_item[0])
            return os.path.join(self.file_path, item['text'])
        return None

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title('Audio Transcription & Diarization')
        
        # Set the icon and configure the window
        icon_path = os.path.join(os.path.dirname(__file__), '../../Icon/MeetNote.ico')
        self.root.iconbitmap(icon_path)
        self.root.minsize(800, 600)  # Set minimum window size
        
        self.config = config_manager.config
        self.file_path = ttk.StringVar()
        self.num_speakers = ttk.IntVar(value=2)
        self.diarization_model = ttk.StringVar(value='speaker-diarization-3.1')
        self.transcription_method = ttk.StringVar(value='groq')
        self.theme_var = ttk.StringVar(value=self.config.get('gui_theme', 'darkly'))
        self.output_directory = ttk.StringVar(value=self.config.get('output_directory', 'transcriptions'))
        self.processing_location = ttk.StringVar(value=self.config.get('processing_location', 'local'))
        self.combiner_method = ttk.StringVar(value=self.config.get('combiner', {}).get('method', 'semantic_flow'))
        self.combiner_method.trace_add('write', self.update_combiner_method)  # Add trace to update config

        self.process_started = False
        self.process_result = None
        self.create_widgets()
        self.hide_progress_bar()

    def create_widgets(self):
        # Create main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=BOTH, expand=YES)

        # Top frame with modern header and theme selector
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=X, pady=(0, 20))
        
        # Left side: Title
        header_label = ttk.Label(header_frame, text="Audio Processing Center", font=("TkDefaultFont", 16, "bold"))
        header_label.pack(side=LEFT)

        # Right side: Theme selector with modern styling
        theme_frame = ttk.Frame(header_frame)
        theme_frame.pack(side=RIGHT)
        ttk.Label(theme_frame, text='Theme:', font=("TkDefaultFont", 10)).pack(side=LEFT, padx=(0, 5))
        themes = ['darkly', 'superhero', 'solar', 'cyborg', 'vapor', 'litera']
        theme_menu = ttk.Combobox(theme_frame, textvariable=self.theme_var, values=themes, state="readonly", width=12, bootstyle="primary")
        theme_menu.pack(side=LEFT, padx=(0, 5))
        ttk.Button(theme_frame, text="🎨", command=self.change_theme, style='primary-outline.TButton', width=3).pack(side=LEFT)

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
        browse_button = ttk.Button(file_frame, text='📂 Browse Directory', command=self.browse_directory, style='primary.TButton', width=20)
        browse_button.grid(row=0, column=0, sticky=W, pady=(0, 10))

        # File browser with modern styling
        browser_frame = ttk.LabelFrame(file_frame, text="Media Files", padding="10", bootstyle="primary")
        browser_frame.grid(row=1, column=0, sticky=NSEW)
        
        self.file_browser = CustomFileBrowser(browser_frame, self)
        self.file_browser.pack(expand=YES, fill=BOTH)

        # File info below browser
        self.file_label = ttk.Label(file_frame, text="No file selected", font=("TkDefaultFont", 12, "bold"))
        self.file_label.grid(row=2, column=0, sticky=W, pady=(10, 0))
        self.file_info = ttk.Label(file_frame, text="")
        self.file_info.grid(row=3, column=0, sticky=W)

        # Right column: Settings
        settings_frame = ttk.Frame(content_frame)
        settings_frame.grid(row=0, column=1, sticky=NSEW)

        # Output Settings at the top
        output_frame = ttk.LabelFrame(settings_frame, text="Output Settings", padding="10", bootstyle="primary")
        output_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(output_frame, text='Output Directory:', font=("TkDefaultFont", 10)).pack(fill=X, pady=(0, 5))
        dir_frame = ttk.Frame(output_frame)
        dir_frame.pack(fill=X)
        ttk.Entry(dir_frame, textvariable=self.output_directory).pack(side=LEFT, fill=X, expand=YES, padx=(0, 5))
        ttk.Button(dir_frame, text="📁", command=self.browse_output_directory, style='primary-outline.TButton', width=3).pack(side=RIGHT)

        # Diarization Settings with collapsible advanced options
        diar_frame = ttk.LabelFrame(settings_frame, text="Diarization Settings", padding="10", bootstyle="primary")
        diar_frame.pack(fill=X, pady=(0, 10))
        
        # Basic diarization settings
        speaker_frame = ttk.Frame(diar_frame)
        speaker_frame.pack(fill=X, pady=(0, 5))
        ttk.Label(speaker_frame, text='Number of Speakers:', font=("TkDefaultFont", 10)).pack(side=LEFT)
        ttk.Spinbox(speaker_frame, from_=1, to=10, textvariable=self.num_speakers, width=5).pack(side=LEFT, padx=(5, 0))
        
        # Advanced diarization settings in a collapsible frame
        advanced_frame = ttk.Labelframe(diar_frame, text="Advanced Options", padding="5", bootstyle="secondary")
        advanced_frame.pack(fill=X, pady=(5, 0))
        
        ttk.Label(advanced_frame, text='Diarization Model:', font=("TkDefaultFont", 10)).pack(fill=X, pady=(0, 5))
        diarization_models = [
            'speaker-diarization-3.1',  # Default/recommended first
            'speaker-diarization-3.0',
            'speech-separation-ami-1.0',
            'segmentation',
            'wespeaker-voxceleb-resnet34-LM'
        ]
        ttk.Combobox(advanced_frame, textvariable=self.diarization_model, values=diarization_models, state="readonly").pack(fill=X)

        # Processing Settings
        proc_frame = ttk.LabelFrame(settings_frame, text="Processing Settings", padding="10", bootstyle="primary")
        proc_frame.pack(fill=X)
        
        # Transcription Method
        ttk.Label(proc_frame, text='Transcription Method:', font=("TkDefaultFont", 10)).pack(fill=X, pady=(0, 5))
        transcription_methods = ['local', 'groq']
        ttk.Combobox(proc_frame, textvariable=self.transcription_method, values=transcription_methods, state="readonly").pack(fill=X, pady=(0, 10))
        
        # Combiner Method
        ttk.Label(proc_frame, text='Combiner Method:', font=("TkDefaultFont", 10)).pack(fill=X, pady=(0, 5))
        combiner_methods = [
            'semantic_flow',      # Current default
            'semantic',           # Basic semantic
            'semantic_enhanced',  # Enhanced version
            'semantic_adaptive',  # Adaptive version
            'two_stage_llm',     # LLM-based
            'groq_llm',          # Groq specific
            'adaptive',          # Basic adaptive
            'adaptive_rule',     # Rule-based adaptive
            'weighted',          # Weighted combination
            'simple'             # Simple combination
        ]
        ttk.Combobox(proc_frame, textvariable=self.combiner_method, values=combiner_methods, state="readonly").pack(fill=X, pady=(0, 10))
        
        # Processing Location
        ttk.Label(proc_frame, text='Diarization Method:', font=("TkDefaultFont", 10)).pack(fill=X, pady=(0, 5))
        processing_locations = ['local', 'cloud']
        ttk.Combobox(proc_frame, textvariable=self.processing_location, values=processing_locations, state="readonly").pack(fill=X)

        # Progress bar (hidden initially)
        self.progress_frame = ttk.Frame(settings_frame)
        self.progress_frame.pack(fill=X, pady=(10, 0))
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate', bootstyle="success-striped")
        self.progress_bar.pack(fill=X, expand=YES, padx=(0, 10))
        
        self.progress_label = ttk.Label(self.progress_frame, text="0%", font=("TkDefaultFont", 10))
        self.progress_label.pack(side=RIGHT)

        # Start button with modern styling
        self.start_button = ttk.Button(settings_frame, text='▶ Start Processing', command=self.start_process, style='success.TButton', width=20)
        self.start_button.pack(pady=(10, 0), fill=X)

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=X, side=BOTTOM, pady=(10, 0))
        ttk.Separator(status_frame).pack(fill=X, pady=(0, 5))
        ttk.Label(status_frame, text="Ready", font=("TkDefaultFont", 9), bootstyle="secondary").pack(side=LEFT)

        # Initial population of the file browser
        self.populate_file_browser()

    def browse_directory(self):
        directory = filedialog.askdirectory(initialdir=self.config.get("last_directory"))
        if directory:
            self.config["last_directory"] = directory
            config_manager.save_config()
            self.file_browser.file_path = directory
            self.file_browser.populate(directory)
            self.root.update_idletasks()
            self.adjust_window_size()
            
    def browse_output_directory(self):
        directory = filedialog.askdirectory(initialdir=self.output_directory.get())
        if directory:
            self.output_directory.set(directory)
            self.config["output_directory"] = directory
            config_manager.save_config()

    def select_file(self):
        selected_file = self.file_browser.get_selected_file()
        if selected_file:
            self.file_path.set(selected_file)
            file_name = os.path.basename(selected_file)
            file_size = os.path.getsize(selected_file) / (1024 * 1024)
            file_mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(selected_file)))
            self.file_label.config(text=f"{file_name}")
            self.file_info.config(text=f"Size: {file_size:.2f} MB | Modified: {file_mod_time}")
        else:
            self.file_label.config(text="No file selected")
            self.file_info.config(text="")

    def hide_progress_bar(self):
        self.progress_frame.pack_forget()

    def show_progress_bar(self):
        self.progress_frame.pack(fill=X, pady=10, before=self.start_button)
        self.progress_bar['value'] = 0
        self.progress_label['text'] = "0%"
        self.root.update_idletasks()

    def start_process(self):
        if not self.file_path.get():
            ttk.dialogs.Messagebox.show_error('Please select an audio file.', 'Error')
            return
        
        self.process_started = True
        self.show_progress_bar()  # Show progress bar when starting the process
        self.process_result = {
            'file_path': self.file_path.get(),
            'num_speakers': self.num_speakers.get(),
            'diarization_model': self.diarization_model.get(),
            'transcription_method': self.transcription_method.get(),
            'output_directory': self.output_directory.get(),
            'processing_location': self.processing_location.get()
        }
        self.start_button.config(state='disabled')

    def update_progress(self, value):
        self.progress_bar['value'] = value
        self.progress_label['text'] = f"{value}%"
        self.root.update_idletasks()

    def change_theme(self):
        new_theme = self.theme_var.get()
        if new_theme != self.config.get('gui_theme'):
            self.config['gui_theme'] = new_theme
            config_manager.save_config()
            self.restart_application()

    def restart_application(self):
        self.root.destroy()
        current_script = sys.argv[0]
        if sys.prefix != sys.base_prefix:
            python = os.path.join(sys.prefix, 'Scripts' if sys.platform == "win32" else 'bin', 'python')
        else:
            python = sys.executable
        
        subprocess.Popen([python, current_script])

    def adjust_window_size(self):
        self.root.update_idletasks()
        width = self.root.winfo_reqwidth() + 40
        height = self.root.winfo_reqheight() + 40
        self.root.geometry(f"{width}x{height}")

    def populate_file_browser(self):
        if self.config.get("last_directory") and os.path.exists(self.config["last_directory"]):
            self.file_browser.file_path = self.config["last_directory"]
        else:
            self.config["last_directory"] = os.path.expanduser("~/Videos")
            config_manager.save_config()
            self.file_browser.file_path = self.config["last_directory"]
        
        self.file_browser.populate(self.file_browser.file_path)
        self.root.update_idletasks()
        self.adjust_window_size()

    def run(self):
        """Start the main loop of the application."""
        self.root.mainloop()

    def get_process_result(self):
        """Return the result of the processing if started, otherwise None."""
        if self.process_started:
            return self.process_result
        return None

    def update_processing_location(self, *args):
        """Update config when processing location changes"""
        self.config['processing_location'] = self.processing_location.get()
        config_manager.save_config()

    def update_combiner_method(self, *args):
        """Update the combiner method in the config when changed"""
        if 'combiner' not in self.config:
            self.config['combiner'] = {}
        self.config['combiner']['method'] = self.combiner_method.get()
        config_manager.save_config()
        print(f"Updated combiner method to: {self.combiner_method.get()}")  # Debug print

def create_gui():
    """Create and return the main window and root objects."""
    root = ttk.Window(themename=config_manager.config.get('gui_theme', 'darkly'))
    window = MainWindow(root)
    return window, root

if __name__ == '__main__':
    window, root = create_gui()
    window.run()
    result = window.get_process_result()
    if result:
        print(result)