# Config Manager Documentation

## Overview

The Config Manager is a crucial component of the application, responsible for loading, storing, and managing configuration settings. It provides a centralized way to handle application-wide settings, ensuring consistency and ease of configuration across the entire application.

## Key Features

1. **JSON-based Configuration**: Uses a JSON file for storing configuration, making it human-readable and easy to edit.
2. **Default Configuration**: Provides sensible default values for all settings.
3. **Dynamic Updates**: Allows runtime updates to configuration values.
4. **Persistent Storage**: Saves changes to the configuration file, ensuring settings persist between application runs.

## Implementation Details

### Class: ConfigManager

```python
class ConfigManager:
    def __init__(self, config_file='config.json'):
        self.config_file = os.path.join('Config', config_file)
        self.config = self._load_config()
```

The ConfigManager is initialized with a path to the configuration file. It automatically loads the configuration upon instantiation.

### Key Methods

1. **_load_config()**
   - Loads the configuration from the JSON file.
   - If the file doesn't exist, it returns a default configuration.

2. **_get_default_config()**
   - Returns a dictionary with default configuration values.
   - Includes settings for model options, CUDA usage, output directory, diarization parameters, etc.

3. **save_config()**
   - Saves the current configuration to the JSON file.
   - Ensures the Config directory exists before saving.

4. **get(key, default=None)**
   - Retrieves a configuration value by key.
   - Returns a default value if the key doesn't exist.

5. **set(key, value)**
   - Sets a configuration value and immediately saves it to the file.

6. **update(new_config)**
   - Updates multiple configuration values at once.

## Usage Example

```python
from utils.config_manager import ConfigManager

# Initialize the config manager
config_manager = ConfigManager()

# Get a configuration value
output_dir = config_manager.get('output_directory', 'default_output')

# Set a configuration value
config_manager.set('use_cuda', True)

# Update multiple values
config_manager.update({
    'model_options': {'local': {'model': 'large-v2'}},
    'diarization': {'min_speakers': 2}
})
```

## Best Practices

1. Use the ConfigManager as a singleton throughout your application to ensure consistency.
2. Access configuration values through the ConfigManager rather than hardcoding them in your application.
3. Use the `get` method with a default value to handle cases where a configuration key might not exist.
4. When making changes to the configuration, use the `set` or `update` methods to ensure changes are persisted.

## Extensibility

The ConfigManager can be easily extended to include additional features such as:
- Configuration validation
- Environment-specific configurations (e.g., development, production)
- Encrypted storage for sensitive configuration values

By centralizing configuration management, the ConfigManager enhances the maintainability and flexibility of the entire application.