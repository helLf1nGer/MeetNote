import json
import os
import shutil

class ConfigManager:
    def __init__(self, config_file='config.json'):
        # Get the project root directory (two levels up from utils)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.config_dir = os.path.join(self.project_root, 'Config')
        self.config_file = os.path.join(self.config_dir, config_file)
        self.template_file = os.path.join(self.project_root, 'config.template.json')
        print(f"Looking for config at: {self.config_file}")  # Debug print
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
            },
            'combiner': {
                'method': 'semantic',  # Options: 'simple', 'weighted', 'adaptive', 'adaptive_rule', 'semantic', 'semantic_adaptive', 'semantic_enhanced', 'two_stage_llm', 'groq_llm'
                'model': 'llama3-groq-70b-8192-tool-use-preview'  # Only used for 'two_stage_llm' method
            }
        }

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._create_config_from_template()
        return self.config

    def save_config(self):
        # Ensure the Config directory exists
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
        # Reload the config after saving
        self.load_config()

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

    def update(self, new_config):
        self.config.update(new_config)
        self.save_config()