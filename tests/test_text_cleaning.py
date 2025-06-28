#!/usr/bin/env python3
"""
Tests for text cleaning functionality.
"""

import sys
import os
import unittest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.smart_text_cleaner import SmartTextCleaner


class MockOCRResult:
    """Mock OCR result for testing."""
    def __init__(self, text):
        self.text = text


class TestTextCleaning(unittest.TestCase):
    """Test text cleaning functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cleaner = SmartTextCleaner()
        self.expected_text = "Trailing only Facebook Messenger, WeChat is now the second most popular messaging platform in Bhutan and Mongolia."
    
    def test_perfect_cleaning(self):
        """Test that the cleaner produces perfect results on known input."""
        # Problematic OCR fragments
        raw_fragments = [
            "Trailing only Facebook Messenger,",
            "WeChat is now the second most",
            "popular messaging platform in",
            "Bhutan and Mongolia.",
            "Bhutan and Mongolia.",  # Duplicate
            "popular messaging platform in",  # Duplicate
            "Trailing only Facebook Messenger,",  # Duplicate
            "WeChat is now the second most",  # Duplicate
            "Bhutanland",  # Error
            "€b:",  # Artifact
            "Més:",  # Artifact
            "WeChat'is:now:the:second most:",  # Spacing/punctuation error
            "popuilar:mess",  # Spelling error
            "lattormin",  # Word error
            "Mongoliax",  # Spelling error
            "WeChatis now [he second mosti",  # Multiple errors
            "popular messaging platform;jn"  # Punctuation error
        ]
        
        # Create mock OCR results
        mock_results = [MockOCRResult(fragment) for fragment in raw_fragments]
        
        # Clean the text
        cleaned_text = self.cleaner.clean_ocr_results(mock_results)
        
        # Should produce perfect result
        self.assertEqual(cleaned_text.strip().rstrip('.'), 
                        self.expected_text.strip().rstrip('.'))
    
    def test_empty_input(self):
        """Test handling of empty input."""
        result = self.cleaner.clean_ocr_results([])
        self.assertEqual(result, "")
    
    def test_single_fragment(self):
        """Test handling of single fragment."""
        mock_results = [MockOCRResult(self.expected_text)]
        result = self.cleaner.clean_ocr_results(mock_results)
        self.assertIn("Trailing only Facebook Messenger", result)
    
    def test_artifact_removal(self):
        """Test removal of OCR artifacts."""
        mock_results = [MockOCRResult("€b: Més: Hello World")]
        result = self.cleaner.clean_ocr_results(mock_results)
        self.assertNotIn("€b:", result)
        self.assertNotIn("Més:", result)
        self.assertIn("Hello World", result)
    
    def test_word_corrections(self):
        """Test word-level corrections."""
        test_cases = [
            ("WeChatis", "WeChat is"),
            ("popuilar", "popular"),
            ("Bhutanland", "Bhutan and"),
            ("lattormin", "platform"),
        ]
        
        for wrong, correct in test_cases:
            mock_results = [MockOCRResult(f"This is {wrong} text")]
            result = self.cleaner.clean_ocr_results(mock_results)
            self.assertIn(correct, result)
            self.assertNotIn(wrong, result)


if __name__ == '__main__':
    unittest.main()
