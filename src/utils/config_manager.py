import json
import os
import shutil

class ConfigManager:
    def __init__(self, config_file='config.json'):
        self.config_dir = 'Config'
        self.config_file = os.path.join(self.config_dir, config_file)
        self.template_file = 'config.template.json'
        self.config = self._load_config()

    def _load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return self._create_config_from_template()

    def _create_config_from_template(self):
        if os.path.exists(self.template_file):
            os.makedirs(self.config_dir, exist_ok=True)
            shutil.copy(self.template_file, self.config_file)
            print(f"Created new configuration file: {self.config_file}")
            print("Please update the configuration with your specific settings.")
            
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return self._get_default_config()
    
    def _get_default_config(self):
        return {
            'misc': {
                'print_to_terminal': True
            },
            'model_options': {
                'local': {
                    'model': 'medium.en',
                    'device': 'cuda',
                    'compute_type': 'float16'
                },
                'groq': {
                    'model': 'whisper-large-v3'
                }
            },
            'use_cuda': True,
            'output_directory': 'transcriptions',
            'diarization': {
                'min_speakers': 1,
                'max_speakers': 10,
                'default_num_speakers': 2
            },
            'transcription': {
                'language': 'en',
                'task': 'transcribe'
            },
            'pdf_output': {
                'font_size': 12,
                'line_spacing': 1.2
            }
        }

    def save_config(self):
        # Ensure the Config directory exists
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

    def update(self, new_config):
        self.config.update(new_config)
        self.save_config()