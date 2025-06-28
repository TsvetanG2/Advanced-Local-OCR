"""Logging configuration module."""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

from .config import get_config


def setup_logging(log_level: Optional[str] = None) -> None:
    """Setup application logging configuration.
    
    Args:
        log_level: Override log level from config
    """
    config = get_config()
    
    # Get logging configuration
    log_file = config.get('logging.file', 'logs/ocr_app.log')
    level = log_level or config.get('logging.level', 'INFO')
    max_size = config.get('logging.max_size', '10MB')
    backup_count = config.get('logging.backup_count', 5)
    log_format = config.get('logging.format', 
                           '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert max_size to bytes
    size_multipliers = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    if isinstance(max_size, str):
        for suffix, multiplier in size_multipliers.items():
            if max_size.upper().endswith(suffix):
                max_bytes = int(max_size[:-2]) * multiplier
                break
        else:
            max_bytes = int(max_size)
    else:
        max_bytes = max_size
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('easyocr').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logging.info("Logging system initialized")


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"performance.{name}")
    
    def log_ocr_performance(self, image_path: str, processing_time: float, 
                          memory_used: int, text_length: int) -> None:
        """Log OCR performance metrics."""
        self.logger.info(
            f"OCR Performance - Image: {image_path}, "
            f"Time: {processing_time:.2f}s, "
            f"Memory: {memory_used / 1024 / 1024:.1f}MB, "
            f"Text Length: {text_length}"
        )
    
    def log_llm_performance(self, provider: str, model: str, 
                          processing_time: float, tokens_used: int) -> None:
        """Log LLM performance metrics."""
        self.logger.info(
            f"LLM Performance - Provider: {provider}, "
            f"Model: {model}, "
            f"Time: {processing_time:.2f}s, "
            f"Tokens: {tokens_used}"
        )


class ErrorLogger:
    """Logger for error tracking and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger("errors")
    
    def log_ocr_error(self, error: Exception, image_path: str, 
                     engine: str, context: str = "") -> None:
        """Log OCR-related errors."""
        self.logger.error(
            f"OCR Error - Engine: {engine}, "
            f"Image: {image_path}, "
            f"Context: {context}, "
            f"Error: {str(error)}",
            exc_info=True
        )
    
    def log_llm_error(self, error: Exception, provider: str, 
                     text_length: int, context: str = "") -> None:
        """Log LLM-related errors."""
        self.logger.error(
            f"LLM Error - Provider: {provider}, "
            f"Text Length: {text_length}, "
            f"Context: {context}, "
            f"Error: {str(error)}",
            exc_info=True
        )
    
    def log_ui_error(self, error: Exception, component: str, 
                    action: str = "") -> None:
        """Log UI-related errors."""
        self.logger.error(
            f"UI Error - Component: {component}, "
            f"Action: {action}, "
            f"Error: {str(error)}",
            exc_info=True
        )
