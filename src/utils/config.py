"""Configuration management module."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages application configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            # Look for config in multiple locations
            possible_paths = [
                "config/settings.yaml",
                "settings.yaml",
                os.path.expanduser("~/.ocr_app/settings.yaml")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            else:
                raise FileNotFoundError("Configuration file not found in any expected location")
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Override with environment variables
            self._apply_env_overrides(config)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> None:
        """Apply environment variable overrides."""
        # LLM API keys
        if 'OPENAI_API_KEY' in os.environ:
            config['llm']['openai']['api_key'] = os.environ['OPENAI_API_KEY']
        
        if 'ANTHROPIC_API_KEY' in os.environ:
            config['llm']['anthropic']['api_key'] = os.environ['ANTHROPIC_API_KEY']
        
        # GPU setting
        if 'OCR_USE_GPU' in os.environ:
            config['ocr']['engines']['easyocr']['gpu'] = os.environ['OCR_USE_GPU'].lower() == 'true'
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.get('logging.file', '').rsplit('/', 1)[0] if '/' in self.get('logging.file', '') else 'logs',
            self.get('export.output_directory', 'output'),
            'temp',
            'cache'
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'ocr.engines.easyocr.gpu')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """Get OCR-specific configuration."""
        return self.get('ocr', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM-specific configuration."""
        return self.get('llm', {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI-specific configuration."""
        return self.get('ui', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing-specific configuration."""
        return self.get('preprocessing', {})
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU is enabled for OCR."""
        return self.get('ocr.engines.easyocr.gpu', False)
    
    def is_llm_enabled(self) -> bool:
        """Check if LLM correction is enabled."""
        return self.get('llm.correction.enabled', False) and self.get('llm.provider', 'disabled') != 'disabled'
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return self.get('languages.supported', ['en'])
    
    def get_current_language(self) -> str:
        """Get current application language."""
        return self.get('app.language', 'en')


# Global configuration instance
config = None

def get_config() -> ConfigManager:
    """Get global configuration instance."""
    global config
    if config is None:
        config = ConfigManager()
    return config

def reload_config() -> ConfigManager:
    """Reload configuration from file."""
    global config
    config = ConfigManager()
    return config
