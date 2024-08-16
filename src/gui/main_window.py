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

config_manager = ConfigManager()

class CustomFileBrowser(ttk.Treeview):
    def __init__(self, parent, main_window, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.main_window = main_window
        self["columns"] = ("Date", "Type", "Size", "Duration")
        self.heading("#0", text="Name", anchor=tk.W, command=lambda: self.sort_column("#0", False))
        self.heading("Date", text="Date Modified", anchor=tk.W, command=lambda: self.sort_column("Date", False))
        self.heading("Type", text="Type", anchor=tk.W, command=lambda: self.sort_column("Type", False))
        self.heading("Size", text="Size", anchor=tk.W, command=lambda: self.sort_column("Size", False))
        self.heading("Duration", text="Duration", anchor=tk.W, command=lambda: self.sort_column("Duration", False))
        
        self.file_path = None
        self.bind("<Double-1>", self.on_double_click)

    def on_double_click(self, event):
        item = self.identify('item', event.x, event.y)
        if item:
            self.selection_set(item)
            self.main_window.select_file()

    def populate(self, path):
        self.delete(*self.get_children())
        max_widths = {"#0": 20, "Date": 20, "Type": 10, "Size": 15, "Duration": 15}
        
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isfile(full_path):
                file_type = os.path.splitext(item)[1]
                if file_type.lower() in ['.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mov', '.mkv', '.flv']:
                    stats = os.stat(full_path)
                    size = f"{stats.st_size / (1024 * 1024):.2f} MB"
                    date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.st_mtime))
                    duration = self.get_duration(full_path)
                    values = (date, file_type, size, duration)
                    self.insert("", tk.END, text=item, values=values)
                    
                    max_widths["#0"] = min(max(max_widths["#0"], len(item)), 40)
                    for i, col in enumerate(self["columns"]):
                        max_widths[col] = min(max(max_widths[col], len(str(values[i]))), 30)

        for col in ("#0",) + self["columns"]:
            self.column(col, width=max_widths[col]*7)

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
        self.root.title('Audio Transcription and Diarization')
        
        # Set the icon for the main window and taskbar
        icon_path = os.path.join(os.path.dirname(__file__), '../../Icon/MeetNote.ico')
        self.root.iconbitmap(icon_path)
        
        self.config = config_manager.config
        self.file_path = ttk.StringVar()
        self.num_speakers = ttk.IntVar(value=2)
        self.diarization_model = ttk.StringVar(value='speaker-diarization-3.0')
        self.transcription_method = ttk.StringVar(value='groq')
        self.theme_var = ttk.StringVar(value=self.config.get('gui_theme', 'darkly'))
        self.output_directory = ttk.StringVar(value=self.config.get('output_directory', 'transcriptions'))

        self.process_started = False
        self.process_result = None
        self.create_widgets()
        self.hide_progress_bar()  # Hide progress bar initially

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20 10 20 10")
        main_frame.pack(fill=BOTH, expand=YES)

        # Top frame for file browsing
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=X, pady=10)

        browse_button = ttk.Button(top_frame, text='Browse Directory', command=self.browse_directory, style='primary.TButton')
        browse_button.pack(side=LEFT, padx=(0, 10))

        select_button = ttk.Button(top_frame, text='Select File', command=self.select_file, style='info.TButton')
        select_button.pack(side=LEFT)

        # File browser
        self.file_browser = CustomFileBrowser(main_frame, self)
        self.file_browser.pack(expand=YES, fill=BOTH, padx=10, pady=10)

        # File info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=X, pady=10)

        self.file_label = ttk.Label(info_frame, text="No file selected", font=("TkDefaultFont", 12, "bold"))
        self.file_label.pack(side=LEFT)

        self.file_info = ttk.Label(info_frame, text="")
        self.file_info.pack(side=LEFT, padx=(10, 0))

        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10 5 10 5")
        settings_frame.pack(fill=X, pady=10)

        # Number of speakers
        ttk.Label(settings_frame, text='Number of Speakers:').grid(row=0, column=0, sticky=W, padx=(0, 10))
        ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.num_speakers, width=5).grid(row=0, column=1, sticky=W, padx=(0, 20))

        # Diarization model dropdown
        ttk.Label(settings_frame, text='Diarization Model:').grid(row=0, column=2, sticky=W, padx=(0, 10))
        diarization_models = [
            'speaker-diarization-3.0',
            'speaker-diarization-3.1',
            'speech-separation-ami-1.0',
            'segmentation',
            'wespeaker-voxceleb-resnet34-LM'
        ]
        ttk.Combobox(settings_frame, textvariable=self.diarization_model, values=diarization_models, state="readonly", width=30).grid(row=0, column=3, sticky=W, padx=(0, 20))

        # Transcription method dropdown
        ttk.Label(settings_frame, text='Transcription Method:').grid(row=1, column=0, sticky=W, padx=(0, 10), pady=(10, 0))
        transcription_methods = ['groq', 'local']
        ttk.Combobox(settings_frame, textvariable=self.transcription_method, values=transcription_methods, state="readonly", width=10).grid(row=1, column=1, sticky=W, padx=(0, 20), pady=(10, 0))

        # Theme selection
        ttk.Label(settings_frame, text='GUI Theme:').grid(row=1, column=2, sticky=W, padx=(0, 10), pady=(10, 0))
        themes = ['darkly', 'superhero', 'solar', 'cyborg', 'vapor', 'litera']
        theme_menu = ttk.Combobox(settings_frame, textvariable=self.theme_var, values=themes, state="readonly", width=15)
        theme_menu.grid(row=1, column=3, sticky=W, padx=(0, 10), pady=(10, 0))
        ttk.Button(settings_frame, text="Apply Theme", command=self.change_theme, style='secondary.TButton').grid(row=1, column=4, sticky=W, pady=(10, 0))
        
        ttk.Label(settings_frame, text='Output Directory:').grid(row=2, column=0, sticky=W, padx=(0, 10), pady=(10, 0))
        ttk.Entry(settings_frame, textvariable=self.output_directory, width=30).grid(row=2, column=1, columnspan=2, sticky=W+E, padx=(0, 10), pady=(10, 0))
        ttk.Button(settings_frame, text="Browse", command=self.browse_output_directory, style='secondary.TButton').grid(row=2, column=3, sticky=W, pady=(10, 0))

        # Progress bar
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_frame.pack(fill=X, pady=10)
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=300, mode='determinate')
        self.progress_bar.pack(fill=X, expand=YES, padx=(0, 10))
        self.progress_label = ttk.Label(self.progress_frame, text="0%")
        self.progress_label.pack(side=RIGHT)

        # Start button
        self.start_button = ttk.Button(main_frame, text='Start Processing', command=self.start_process, style='success.TButton')
        self.start_button.pack(pady=10)

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
            'output_directory': self.output_directory.get()
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