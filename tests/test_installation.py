#!/usr/bin/env python3
"""
Installation verification tests for Advanced Local OCR Studio.
"""

import sys
import os
import unittest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestInstallation(unittest.TestCase):
    """Test that all required components are properly installed."""
    
    def test_core_imports(self):
        """Test that core modules can be imported."""
        try:
            import cv2
            import numpy as np
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import QThread
            import yaml
            import Levenshtein
        except ImportError as e:
            self.fail(f"Core import failed: {e}")
    
    def test_ocr_engines(self):
        """Test OCR engine availability."""
        engines_available = []
        
        # Test EasyOCR
        try:
            import easyocr
            engines_available.append("EasyOCR")
        except ImportError:
            pass
        
        # Test Tesseract
        try:
            import pytesseract
            engines_available.append("Tesseract")
        except ImportError:
            pass
        
        self.assertGreater(len(engines_available), 0, 
                          "At least one OCR engine must be available")
    
    def test_local_ocr_import(self):
        """Test that local OCR modules can be imported."""
        try:
            from src.core.local_ocr import LocalOCREngine
            from src.core.text_cleaner import OCRTextCleaner
            from src.core.smart_text_cleaner import SmartTextCleaner
        except ImportError as e:
            self.fail(f"Local OCR import failed: {e}")
    
    def test_config_loading(self):
        """Test configuration loading."""
        try:
            from src.utils.config import get_config
            config = get_config()
            self.assertIsNotNone(config)
        except Exception as e:
            self.fail(f"Config loading failed: {e}")


if __name__ == '__main__':
    unittest.main()
