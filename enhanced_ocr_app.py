#!/usr/bin/env python3
"""
Advanced Local OCR Studio - Professional OCR Application

A modern, local-first OCR application with smart text cleaning and beautiful UI.
Features dual OCR engines (EasyOCR + Tesseract) with intelligent text reconstruction.

Author: OCR Solutions
License: MIT
Version: 2.0.0
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFileDialog, QMessageBox,
    QProgressBar, QGroupBox, QCheckBox, QComboBox, QSplitter,
    QFrame, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from src.core.local_ocr import LocalOCREngine
from src.core.text_processors import TextComparator


class OCRWorkerEnhanced(QThread):
    """Enhanced OCR worker thread with proper cleanup."""

    finished = pyqtSignal(list, str)  # OCR results, cleaned text
    error_occurred = pyqtSignal(str)  # Error message

    def __init__(self, image_path: str, preprocess: bool = True):
        super().__init__()
        self.image_path = image_path
        self.preprocess = preprocess
        self.ocr_engine = LocalOCREngine()
        self._is_running = True

    def run(self):
        try:
            if not self._is_running:
                return

            # Load image
            data = np.fromfile(self.image_path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)

            if image is None:
                if self._is_running:
                    self.error_occurred.emit("Failed to load image")
                return

            if not self._is_running:
                return

            # Extract raw text results
            results = self.ocr_engine.extract_text(image, self.preprocess)

            if not self._is_running:
                return

            # Get cleaned text
            cleaned_text = self.ocr_engine.extract_text_cleaned(image, self.preprocess)

            if self._is_running:
                self.finished.emit(results, cleaned_text)

        except Exception as e:
            if self._is_running:
                self.error_occurred.emit(str(e))

    def stop(self):
        """Stop the worker thread gracefully."""
        self._is_running = False


class EnhancedOCRApp(QMainWindow):
    """Enhanced OCR application with modern visual design."""

    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.current_results = []
        self.cleaned_text = ""
        self.ocr_worker = None

        self.setWindowTitle("🔍 Advanced Local OCR Studio")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(900, 600)

        # Apply modern styling
        self._apply_modern_style()
        self._setup_ui()
        self._check_ocr_availability()

    def closeEvent(self, event):
        """Handle application close event to properly cleanup threads."""
        if self.ocr_worker and self.ocr_worker.isRunning():
            # Stop the worker thread gracefully
            self.ocr_worker.stop()
            self.ocr_worker.quit()
            self.ocr_worker.wait(3000)  # Wait up to 3 seconds

        event.accept()
    
    def _apply_modern_style(self):
        """Apply modern visual styling."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                color: #333333;
            }
            
            QLabel#title {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                padding: 15px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border-radius: 8px;
                margin: 10px;
            }
            
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin: 10px 5px;
                padding-top: 15px;
                background-color: white;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 5px 10px;
                background-color: #ecf0f1;
                border-radius: 4px;
                color: #2c3e50;
            }
            
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                min-height: 20px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5dade2, stop:1 #3498db);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2980b9, stop:1 #1f618d);
            }
            
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
            
            QPushButton#success {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #27ae60, stop:1 #229954);
            }
            
            QPushButton#success:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #58d68d, stop:1 #27ae60);
            }
            
            QPushButton#warning {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f39c12, stop:1 #e67e22);
            }
            
            QPushButton#danger {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e74c3c, stop:1 #c0392b);
            }
            
            QTextEdit {
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                background-color: white;
                selection-background-color: #3498db;
            }
            
            QTextEdit:focus {
                border-color: #3498db;
            }
            
            QComboBox, QCheckBox {
                font-size: 12px;
                padding: 5px;
            }
            
            QComboBox {
                border: 2px solid #bdc3c7;
                border-radius: 4px;
                padding: 5px 10px;
                background-color: white;
                min-width: 120px;
            }
            
            QComboBox:focus {
                border-color: #3498db;
            }
            
            QCheckBox {
                color: #2c3e50;
                font-weight: normal;
            }
            
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 2px solid #bdc3c7;
                background-color: white;
            }
            
            QCheckBox::indicator:checked {
                background-color: #3498db;
                border-color: #2980b9;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xIDQuNUw0LjUgOEwxMSAxIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K);
            }
            
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                background-color: #ecf0f1;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                border-radius: 6px;
            }
            
            QLabel#status {
                color: #2c3e50;
                font-size: 13px;
                padding: 8px 15px;
                background-color: #ecf0f1;
                border-radius: 4px;
                border: 1px solid #bdc3c7;
            }
            
            QSplitter::handle {
                background-color: #bdc3c7;
                width: 3px;
                border-radius: 1px;
            }
            
            QSplitter::handle:hover {
                background-color: #3498db;
            }
        """)
    
    def _setup_ui(self):
        """Setup the enhanced user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Modern title bar
        title = QLabel("🔍 Advanced Local OCR Studio")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Main content with splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel - Controls
        left_panel = self._create_enhanced_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = self._create_enhanced_right_panel()
        main_splitter.addWidget(right_panel)
        
        main_splitter.setSizes([350, 850])
        
        # Enhanced bottom panel
        bottom_panel = self._create_enhanced_bottom_panel()
        layout.addWidget(bottom_panel)
    
    def _create_enhanced_left_panel(self):
        """Create enhanced left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Image loading section
        image_group = QGroupBox("📁 Image Input")
        image_layout = QVBoxLayout(image_group)
        
        self.load_btn = QPushButton("📂 Load Image")
        self.load_btn.clicked.connect(self.load_image)
        image_layout.addWidget(self.load_btn)
        
        self.image_info = QLabel("No image loaded")
        self.image_info.setWordWrap(True)
        self.image_info.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 10px;")
        image_layout.addWidget(self.image_info)
        
        layout.addWidget(image_group)
        
        # OCR settings section
        ocr_group = QGroupBox("⚙️ OCR Settings")
        ocr_layout = QVBoxLayout(ocr_group)
        
        self.preprocess_check = QCheckBox("🔧 Enable smart preprocessing")
        self.preprocess_check.setChecked(True)
        ocr_layout.addWidget(self.preprocess_check)
        
        ocr_layout.addWidget(QLabel("Engine Selection:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["🤖 Auto (Best Available)", "🔍 EasyOCR Only", "📝 Tesseract Only"])
        ocr_layout.addWidget(self.engine_combo)
        
        layout.addWidget(ocr_group)
        
        # Processing section
        process_group = QGroupBox("🚀 Processing")
        process_layout = QVBoxLayout(process_group)
        
        self.process_btn = QPushButton("✨ Extract Text")
        self.process_btn.setObjectName("success")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        process_layout.addWidget(self.process_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear Results")
        self.clear_btn.setObjectName("warning")
        self.clear_btn.clicked.connect(self.clear_results)
        process_layout.addWidget(self.clear_btn)
        
        layout.addWidget(process_group)
        
        # Text comparison section
        comparison_group = QGroupBox("📊 Text Comparison")
        comparison_layout = QVBoxLayout(comparison_group)
        
        comparison_layout.addWidget(QLabel("Expected text:"))
        self.expected_text = QTextEdit()
        self.expected_text.setMaximumHeight(80)
        self.expected_text.setPlaceholderText("Enter expected text to compare accuracy...")
        comparison_layout.addWidget(self.expected_text)
        
        self.compare_btn = QPushButton("📈 Compare Text")
        self.compare_btn.clicked.connect(self.compare_text)
        self.compare_btn.setEnabled(False)
        comparison_layout.addWidget(self.compare_btn)
        
        layout.addWidget(comparison_group)
        
        layout.addStretch()
        return panel
    
    def _create_enhanced_right_panel(self):
        """Create enhanced right results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Results header
        results_header = QLabel("📄 Extraction Results")
        results_header.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 6px;
            border: 1px solid #bdc3c7;
        """)
        layout.addWidget(results_header)
        
        # Cleaned extracted text
        cleaned_group = QGroupBox("✨ Smart Cleaned Text")
        cleaned_layout = QVBoxLayout(cleaned_group)
        
        self.extracted_text = QTextEdit()
        self.extracted_text.setReadOnly(True)
        self.extracted_text.setPlaceholderText("🎯 Cleaned and optimized text will appear here...")
        self.extracted_text.setMinimumHeight(150)
        cleaned_layout.addWidget(self.extracted_text)
        
        layout.addWidget(cleaned_group)
        
        # Raw OCR details
        raw_group = QGroupBox("🔍 Raw OCR Analysis")
        raw_layout = QVBoxLayout(raw_group)
        
        self.results_details = QTextEdit()
        self.results_details.setReadOnly(True)
        self.results_details.setMaximumHeight(120)
        self.results_details.setPlaceholderText("📊 Detailed OCR engine results will appear here...")
        raw_layout.addWidget(self.results_details)
        
        layout.addWidget(raw_group)
        
        # Comparison results
        comparison_group = QGroupBox("📈 Accuracy Analysis")
        comparison_layout = QVBoxLayout(comparison_group)
        
        self.comparison_results = QTextEdit()
        self.comparison_results.setReadOnly(True)
        self.comparison_results.setMaximumHeight(100)
        self.comparison_results.setPlaceholderText("📊 Text comparison results will appear here...")
        comparison_layout.addWidget(self.comparison_results)
        
        layout.addWidget(comparison_group)
        
        return panel
    
    def _create_enhanced_bottom_panel(self):
        """Create enhanced bottom status panel."""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border-top: 3px solid #3498db;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Status label
        self.status_label = QLabel("🟢 Ready - Load an image to start")
        self.status_label.setObjectName("status")
        self.status_label.setStyleSheet("""
            color: white;
            font-size: 14px;
            font-weight: bold;
            background-color: transparent;
            border: none;
            padding: 5px;
        """)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2c3e50;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                background-color: #2c3e50;
                color: white;
                height: 25px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f39c12, stop:1 #e67e22);
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        return panel

    def _check_ocr_availability(self):
        """Check which OCR engines are available."""
        try:
            ocr_engine = LocalOCREngine()
            available_engines = ocr_engine.get_available_engines()

            if available_engines:
                self.status_label.setText(f"🟢 Ready - OCR engines: {', '.join(available_engines)}")
            else:
                self.status_label.setText("🔴 No OCR engines available! Please install EasyOCR or Tesseract.")
        except Exception as e:
            self.status_label.setText(f"🔴 Error checking OCR engines: {e}")

    @pyqtSlot()
    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
        )

        if file_path:
            self.current_image_path = file_path
            filename = os.path.basename(file_path)
            self.image_info.setText(f"📁 {filename}")
            self.process_btn.setEnabled(True)
            self.status_label.setText(f"🟡 Image loaded: {filename}")

    @pyqtSlot()
    def process_image(self):
        """Process the current image with OCR."""
        if not self.current_image_path:
            return

        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("🔄 Processing image...")

        preprocess = self.preprocess_check.isChecked()
        self.ocr_worker = OCRWorkerEnhanced(self.current_image_path, preprocess)
        self.ocr_worker.finished.connect(self.on_ocr_finished)
        self.ocr_worker.error_occurred.connect(self.on_ocr_error)
        self.ocr_worker.start()

    @pyqtSlot(list, str)
    def on_ocr_finished(self, results, cleaned_text):
        """Handle OCR completion."""
        self.current_results = results
        self.cleaned_text = cleaned_text

        if cleaned_text:
            self.extracted_text.setPlainText(cleaned_text)
            self.status_label.setText(f"✅ OCR completed - {len(results)} regions processed")
            self.compare_btn.setEnabled(True)
        else:
            self.extracted_text.setPlainText("❌ No text detected")
            self.status_label.setText("🔴 No text detected")

        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Properly cleanup the worker thread
        if self.ocr_worker:
            self.ocr_worker.deleteLater()
            self.ocr_worker = None

    @pyqtSlot(str)
    def on_ocr_error(self, error_message):
        """Handle OCR error."""
        QMessageBox.critical(self, "OCR Error", f"❌ OCR failed:\n{error_message}")
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("🔴 OCR failed")

        # Properly cleanup the worker thread
        if self.ocr_worker:
            self.ocr_worker.deleteLater()
            self.ocr_worker = None

    @pyqtSlot()
    def compare_text(self):
        """Compare extracted text with expected text."""
        expected = self.expected_text.toPlainText().strip()
        if not expected or not self.current_results:
            return

        try:
            extracted = self.cleaned_text or ' '.join([r.text for r in self.current_results])
            comparator = TextComparator()
            comparison = comparator.compare_texts(expected, extracted)
            similarity = comparison.get('similarity', 0)

            emoji = "🎉" if similarity > 0.95 else "👍" if similarity > 0.85 else "👌" if similarity > 0.7 else "⚠️"
            result_text = f"{emoji} SIMILARITY: {similarity:.1%}"
            self.comparison_results.setPlainText(result_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Comparison failed: {e}")

    @pyqtSlot()
    def clear_results(self):
        """Clear all results."""
        self.extracted_text.clear()
        self.results_details.clear()
        self.comparison_results.clear()
        self.current_results = []
        self.cleaned_text = ""
        self.compare_btn.setEnabled(False)
        self.status_label.setText("🟡 Results cleared")


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Advanced Local OCR Studio")

    window = EnhancedOCRApp()
    window.show()

    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
