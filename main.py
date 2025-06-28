#!/usr/bin/env python3
"""
OCR Document Processor - Main Application Entry Point

Advanced OCR processing application with LLM correction capabilities.
Supports multiple OCR engines, batch processing, and intelligent text comparison.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt

from src.utils.config import get_config, reload_config
from src.utils.logging_config import setup_logging
from src.gui.main_window import MainWindow


def setup_application() -> QApplication:
    """Setup and configure the Qt application.
    
    Returns:
        Configured QApplication instance
    """
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("OCR Document Processor")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("OCR Solutions")
    
    return app


def check_dependencies() -> bool:
    """Check if all required dependencies are available.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    missing_deps = []
    
    # Check core dependencies
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import easyocr
    except ImportError:
        missing_deps.append("easyocr")
    
    try:
        import pytesseract
    except ImportError:
        missing_deps.append("pytesseract")
    
    try:
        from Levenshtein import distance
    except ImportError:
        missing_deps.append("python-Levenshtein")
    
    try:
        import yaml
    except ImportError:
        missing_deps.append("PyYAML")
    
    # Check optional dependencies
    optional_missing = []
    
    try:
        import openai
    except ImportError:
        optional_missing.append("openai (for OpenAI LLM support)")
    
    try:
        import anthropic
    except ImportError:
        optional_missing.append("anthropic (for Anthropic LLM support)")
    
    try:
        import pandas
    except ImportError:
        optional_missing.append("pandas (for Excel export)")
    
    try:
        from reportlab.platypus import SimpleDocTemplate
    except ImportError:
        optional_missing.append("reportlab (for PDF export)")
    
    if missing_deps:
        print("ERROR: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies using:")
        print(f"  pip install {' '.join(missing_deps)}")
        return False
    
    if optional_missing:
        print("WARNING: Missing optional dependencies:")
        for dep in optional_missing:
            print(f"  - {dep}")
        print("\nSome features may not be available.")
        print("Install optional dependencies using:")
        print("  pip install openai anthropic pandas reportlab")
        print()
    
    return True


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="OCR Document Processor - Advanced OCR with LLM correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start GUI application
  python main.py --config custom.yaml  # Use custom configuration
  python main.py --log-level DEBUG     # Enable debug logging
  python main.py --check-deps          # Check dependencies only
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (default: config/settings.yaml)'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Override log level from configuration'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='OCR Document Processor 2.0.0'
    )
    
    return parser.parse_args()


def main() -> int:
    """Main application entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    if args.check_deps:
        print("All required dependencies are available.")
        return 0
    
    try:
        # Initialize configuration
        if args.config:
            # Use custom config file
            os.environ['OCR_CONFIG_PATH'] = args.config
            config = reload_config()
        else:
            config = get_config()
        
        # Setup logging
        setup_logging(args.log_level)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting OCR Document Processor v2.0.0")
        logger.info(f"Configuration loaded from: {config.config_path}")
        
        # Setup Qt application
        app = setup_application()
        
        # Create and show main window
        main_window = MainWindow()
        main_window.show()
        
        logger.info("Application started successfully")
        
        # Run application
        exit_code = app.exec_()
        
        logger.info(f"Application exited with code: {exit_code}")
        return exit_code
        
    except FileNotFoundError as e:
        print(f"ERROR: Configuration file not found: {e}")
        print("Please ensure config/settings.yaml exists or specify a valid config file.")
        return 1
        
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        print("Please check that all dependencies are installed correctly.")
        return 1
        
    except Exception as e:
        # Try to show error in GUI if possible
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            QMessageBox.critical(
                None, "Application Error",
                f"An unexpected error occurred:\n\n{str(e)}\n\n"
                "Please check the log file for more details."
            )
        except:
            # Fallback to console output
            print(f"FATAL ERROR: {e}")
        
        # Log the error if logging is available
        try:
            logger = logging.getLogger(__name__)
            logger.critical(f"Fatal application error: {e}", exc_info=True)
        except:
            pass
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
