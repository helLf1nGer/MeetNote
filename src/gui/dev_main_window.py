import tkinter as tk
from tkinter import filedialog
import os
import time
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from utils.config_manager import ConfigManager

config_manager = ConfigManager()

class DevMainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title('Combiner Testing Tool')
        
        self.config = config_manager.config
        self.file_path = ttk.StringVar()
        self.num_speakers = ttk.IntVar(value=2)
        self.diarization_model = ttk.StringVar(value='speaker-diarization-3.0')
        self.transcription_method = ttk.StringVar(value='groq')
        self.output_directory = ttk.StringVar(value=self.config.get('output_directory', 'dev_results'))

        self.process_started = False
        self.process_result = None
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20 10 20 10")
        main_frame.pack(fill=BOTH, expand=YES)

        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Input File", padding="10 5 10 5")
        file_frame.pack(fill=X, pady=10)

        ttk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side=LEFT, expand=YES, fill=X, padx=(0, 10))
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=LEFT)

        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10 5 10 5")
        settings_frame.pack(fill=X, pady=10)

        ttk.Label(settings_frame, text='Number of Speakers:').grid(row=0, column=0, sticky=W, padx=(0, 10))
        ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.num_speakers, width=5).grid(row=0, column=1, sticky=W, padx=(0, 20))

        ttk.Label(settings_frame, text='Diarization Model:').grid(row=1, column=0, sticky=W, padx=(0, 10))
        diarization_models = ['speaker-diarization-3.0', 'speaker-diarization-3.1', 'segmentation']
        ttk.Combobox(settings_frame, textvariable=self.diarization_model, values=diarization_models, state="readonly", width=30).grid(row=1, column=1, sticky=W, padx=(0, 20))

        ttk.Label(settings_frame, text='Transcription Method:').grid(row=2, column=0, sticky=W, padx=(0, 10))
        transcription_methods = ['groq', 'local']
        ttk.Combobox(settings_frame, textvariable=self.transcription_method, values=transcription_methods, state="readonly", width=10).grid(row=2, column=1, sticky=W, padx=(0, 20))

        ttk.Label(settings_frame, text='Output Directory:').grid(row=3, column=0, sticky=W, padx=(0, 10))
        ttk.Entry(settings_frame, textvariable=self.output_directory, width=30).grid(row=3, column=1, sticky=W+E, padx=(0, 10))
        ttk.Button(settings_frame, text="Browse", command=self.browse_output_directory).grid(row=3, column=2, sticky=W)

        # Start button
        self.start_button = ttk.Button(main_frame, text='Start Testing', command=self.start_process, style='success.TButton')
        self.start_button.pack(pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.flac")])
        if file_path:
            self.file_path.set(file_path)

    def browse_output_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_directory.set(directory)

    def start_process(self):
        if not self.file_path.get():
            ttk.dialogs.Messagebox.show_error('Please select an audio file.', 'Error')
            return
        
        self.process_started = True
        self.process_result = {
            'file_path': self.file_path.get(),
            'num_speakers': self.num_speakers.get(),
            'diarization_model': self.diarization_model.get(),
            'transcription_method': self.transcription_method.get(),
            'output_directory': self.output_directory.get()
        }
        self.start_button.config(state='disabled')
        self.root.quit()

    def run(self):
        self.root.mainloop()

    def get_process_result(self):
        return self.process_result if self.process_started else None

def create_dev_gui():
    root = ttk.Window(themename=config_manager.config.get('gui_theme', 'darkly'))
    window = DevMainWindow(root)
    window.run()
    return window.get_process_result()

if __name__ == '__main__':
    result = create_dev_gui()
    if result:
        print(result)