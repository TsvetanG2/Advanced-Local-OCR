"""Main application window."""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QTextEdit, QListWidget, QLabel, QFileDialog, QMessageBox,
    QProgressBar, QSlider, QTabWidget, QCheckBox, QMenuBar,
    QMenu, QAction, QStatusBar, QSplitter, QGroupBox, QComboBox,
    QSpinBox, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, pyqtSlot, QTimer

from ..core.ocr_worker import OCRWorker
from ..core.batch_processor import BatchProcessor
from ..utils.config import get_config
from ..utils.export import ExportManager
from ..utils.logging_config import ErrorLogger
from .widgets.image_viewer import ImageViewer
from .widgets.settings_dialog import SettingsDialog
from .widgets.batch_dialog import BatchDialog

logger = logging.getLogger(__name__)
error_logger = ErrorLogger()


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        
        # Configuration and managers
        self.config = get_config()
        self.export_manager = ExportManager()
        self.batch_processor = BatchProcessor()
        
        # State variables
        self.current_image_path = None
        self.current_worker = None
        self.recent_files = []
        self.current_results = None
        
        # UI components
        self.image_viewer = None
        self.text_expected = None
        self.text_extracted = None
        self.text_corrected = None
        self.error_list = None
        self.progress_bar = None
        self.status_bar = None
        
        # Initialize UI
        self._setup_ui()
        self._setup_menu_bar()
        self._setup_status_bar()
        self._load_settings()
        
        logger.info("Main window initialized")
    
    def _setup_ui(self) -> None:
        """Setup the main user interface."""
        self.setWindowTitle(f"{self.config.get('app.name', 'OCR Processor')} v{self.config.get('app.version', '2.0.0')}")
        
        # Window geometry
        ui_config = self.config.get_ui_config()
        window_config = ui_config.get('window', {})
        
        self.setGeometry(
            100, 100,
            window_config.get('width', 1200),
            window_config.get('height', 800)
        )
        self.setMinimumSize(
            window_config.get('min_width', 800),
            window_config.get('min_height', 600)
        )
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Image and controls
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Text and results
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 800])
        
        # Bottom panel - Controls and progress
        bottom_panel = self._create_bottom_panel()
        main_layout.addWidget(bottom_panel)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def _create_left_panel(self) -> QWidget:
        """Create left panel with image viewer and controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Image viewer group
        image_group = QGroupBox("Image")
        image_layout = QVBoxLayout(image_group)
        
        self.image_viewer = ImageViewer()
        image_layout.addWidget(self.image_viewer)
        
        # Image controls
        image_controls = QHBoxLayout()
        
        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self.load_image)
        image_controls.addWidget(btn_load)
        
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.clear_image)
        image_controls.addWidget(btn_clear)
        
        image_layout.addLayout(image_controls)
        layout.addWidget(image_group)
        
        # Processing settings group
        settings_group = QGroupBox("Processing Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self._on_confidence_changed)
        conf_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("0.50")
        conf_layout.addWidget(self.confidence_label)
        settings_layout.addLayout(conf_layout)
        
        # Processing mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Text Mode:"))
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Standard", "Legal", "Technical"])
        mode_layout.addWidget(self.mode_combo)
        settings_layout.addLayout(mode_layout)
        
        # LLM correction
        self.llm_checkbox = QCheckBox("Enable LLM Correction")
        self.llm_checkbox.setChecked(self.config.is_llm_enabled())
        settings_layout.addWidget(self.llm_checkbox)
        
        layout.addWidget(settings_group)
        
        # Stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create right panel with text areas and results."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Expected text input
        expected_group = QGroupBox("Expected Text")
        expected_layout = QVBoxLayout(expected_group)
        
        self.text_expected = QTextEdit()
        self.text_expected.setPlaceholderText("Enter expected text for comparison...")
        self.text_expected.setMaximumHeight(100)
        expected_layout.addWidget(self.text_expected)
        
        layout.addWidget(expected_group)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Extracted text tab
        self.text_extracted = QTextEdit()
        self.text_extracted.setReadOnly(True)
        self.text_extracted.setPlaceholderText("Extracted text will appear here...")
        self.results_tabs.addTab(self.text_extracted, "Extracted Text")
        
        # Corrected text tab
        self.text_corrected = QTextEdit()
        self.text_corrected.setReadOnly(True)
        self.text_corrected.setPlaceholderText("LLM-corrected text will appear here...")
        self.results_tabs.addTab(self.text_corrected, "Corrected Text")
        
        # Errors tab
        self.error_list = QListWidget()
        self.results_tabs.addTab(self.error_list, "Errors")
        
        layout.addWidget(self.results_tabs)
        
        return panel
    
    def _create_bottom_panel(self) -> QWidget:
        """Create bottom panel with action buttons and progress."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.btn_process = QPushButton("Process Image")
        self.btn_process.clicked.connect(self.process_image)
        self.btn_process.setEnabled(False)
        button_layout.addWidget(self.btn_process)
        
        btn_batch = QPushButton("Batch Process")
        btn_batch.clicked.connect(self.show_batch_dialog)
        button_layout.addWidget(btn_batch)
        
        btn_export = QPushButton("Export Results")
        btn_export.clicked.connect(self.export_results)
        btn_export.setEnabled(False)
        self.btn_export = btn_export
        button_layout.addWidget(btn_export)
        
        button_layout.addStretch()
        
        btn_settings = QPushButton("Settings")
        btn_settings.clicked.connect(self.show_settings)
        button_layout.addWidget(btn_settings)
        
        layout.addLayout(button_layout)
        
        return panel
    
    def _setup_menu_bar(self) -> None:
        """Setup application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Open action
        open_action = QAction('Open Image...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)
        
        # Recent files submenu
        self.recent_menu = file_menu.addMenu('Recent Files')
        self._update_recent_menu()
        
        file_menu.addSeparator()
        
        # Export actions
        export_action = QAction('Export Results...', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        batch_action = QAction('Batch Process...', self)
        batch_action.triggered.connect(self.show_batch_dialog)
        tools_menu.addAction(batch_action)
        
        settings_action = QAction('Settings...', self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def _setup_status_bar(self) -> None:
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _load_settings(self) -> None:
        """Load application settings."""
        try:
            # Load recent files
            # This would typically be loaded from a settings file
            self.recent_files = []
            
            # Apply theme if configured
            theme = self.config.get('ui.theme', 'light')
            self._apply_theme(theme)
            
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")
    
    def _apply_theme(self, theme: str) -> None:
        """Apply UI theme."""
        # Basic theme application - can be expanded
        if theme == 'dark':
            self.setStyleSheet("""
                QMainWindow { background-color: #2b2b2b; color: #ffffff; }
                QTextEdit { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555; }
                QListWidget { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555; }
                QPushButton { background-color: #4a4a4a; color: #ffffff; border: 1px solid #666; padding: 5px; }
                QPushButton:hover { background-color: #5a5a5a; }
                QGroupBox { color: #ffffff; border: 1px solid #666; margin: 5px; padding-top: 10px; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            """)
    
    @pyqtSlot()
    def load_image(self) -> None:
        """Load an image file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image File",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
            )
            
            if file_path:
                self._load_image_file(file_path)
                
        except Exception as e:
            error_logger.log_ui_error(e, "MainWindow", "load_image")
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def _load_image_file(self, file_path: str) -> None:
        """Load image file and update UI."""
        try:
            self.current_image_path = file_path
            self.image_viewer.load_image(file_path)
            
            # Update UI state
            self.btn_process.setEnabled(True)
            self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")
            
            # Add to recent files
            self._add_recent_file(file_path)
            
            # Clear previous results
            self._clear_results()
            
            logger.info(f"Image loaded: {file_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load image file: {e}")
    
    @pyqtSlot()
    def clear_image(self) -> None:
        """Clear current image and results."""
        self.current_image_path = None
        self.image_viewer.clear()
        self.btn_process.setEnabled(False)
        self.btn_export.setEnabled(False)
        self._clear_results()
        self.status_bar.showMessage("Ready")
    
    def _clear_results(self) -> None:
        """Clear all result displays."""
        self.text_extracted.clear()
        self.text_corrected.clear()
        self.error_list.clear()
        self.current_results = None
        self.btn_export.setEnabled(False)
    
    @pyqtSlot()
    def process_image(self) -> None:
        """Process current image with OCR."""
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        
        try:
            # Get expected text
            expected_text = self.text_expected.toPlainText().strip()
            
            # Update UI for processing
            self.btn_process.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("Processing...")
            
            # Create and start worker
            self.current_worker = OCRWorker(self.current_image_path, expected_text)
            self.current_worker.finished.connect(self._on_processing_finished)
            self.current_worker.progress_update.connect(self._on_progress_update)
            self.current_worker.error_occurred.connect(self._on_processing_error)
            self.current_worker.start()
            
            logger.info(f"Started processing: {self.current_image_path}")
            
        except Exception as e:
            error_logger.log_ui_error(e, "MainWindow", "process_image")
            self._reset_processing_ui()
            QMessageBox.critical(self, "Error", f"Failed to start processing: {str(e)}")
    
    @pyqtSlot(dict)
    def _on_processing_finished(self, results: Dict[str, Any]) -> None:
        """Handle processing completion."""
        try:
            self.current_results = results
            
            # Update text displays
            self.text_extracted.setPlainText(results.get('extracted_text', ''))
            self.text_corrected.setPlainText(results.get('corrected_text', ''))
            
            # Update error list
            self.error_list.clear()
            errors = results.get('errors', [])
            
            if errors:
                for error in errors:
                    if isinstance(error, dict):
                        self.error_list.addItem(error.get('message', str(error)))
                    else:
                        self.error_list.addItem(str(error))
                self.results_tabs.setCurrentIndex(2)  # Show errors tab
            else:
                self.error_list.addItem("No errors found - text matches perfectly!")
                self.results_tabs.setCurrentIndex(1)  # Show corrected text tab
            
            # Update status
            similarity = results.get('similarity', 0.0)
            processing_time = results.get('processing_time', 0.0)
            
            self.status_bar.showMessage(
                f"Processing completed - Similarity: {similarity:.1%}, "
                f"Time: {processing_time:.1f}s"
            )
            
            # Enable export
            self.btn_export.setEnabled(True)
            
            logger.info(f"Processing completed: similarity={similarity:.3f}")
            
        except Exception as e:
            error_logger.log_ui_error(e, "MainWindow", "processing_finished")
            QMessageBox.warning(self, "Warning", f"Error displaying results: {str(e)}")
        
        finally:
            self._reset_processing_ui()
    
    @pyqtSlot(int, int, str)
    def _on_progress_update(self, current: int, total: int, status: str) -> None:
        """Handle progress updates."""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
        
        self.status_bar.showMessage(status)
    
    @pyqtSlot(str)
    def _on_processing_error(self, error_message: str) -> None:
        """Handle processing errors."""
        QMessageBox.critical(self, "Processing Error", error_message)
        self._reset_processing_ui()
    
    def _reset_processing_ui(self) -> None:
        """Reset UI after processing."""
        self.btn_process.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.current_worker = None

    @pyqtSlot(int)
    def _on_confidence_changed(self, value: int) -> None:
        """Handle confidence threshold change."""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")

    @pyqtSlot()
    def export_results(self) -> None:
        """Export current results."""
        if not self.current_results:
            QMessageBox.warning(self, "Warning", "No results to export.")
            return

        try:
            # Get export format
            formats = self.export_manager.get_supported_formats()
            format_filter = ";;".join([
                f"{fmt.upper()} Files (*.{fmt})" for fmt in formats
            ])

            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Export Results",
                f"ocr_results.{formats[0]}",
                format_filter
            )

            if file_path:
                # Determine format from file extension or filter
                format_type = Path(file_path).suffix[1:].lower()
                if format_type not in formats:
                    format_type = formats[0]

                # Create batch result for export
                from ..core.batch_processor import BatchResult
                result = BatchResult(self.current_results['image_path'])
                result.expected_text = self.current_results['expected_text']
                result.extracted_text = self.current_results['extracted_text']
                result.corrected_text = self.current_results['corrected_text']
                result.errors = [error['message'] if isinstance(error, dict) else str(error)
                               for error in self.current_results.get('errors', [])]
                result.similarity = self.current_results['similarity']
                result.processing_time = self.current_results['processing_time']
                result.ocr_confidence = self.current_results['ocr_confidence']
                result.metadata = self.current_results['metadata']
                result.success = self.current_results['success']

                # Export
                exported_path = self.export_manager.export_single_result(
                    result, format_type, os.path.basename(file_path)
                )

                QMessageBox.information(
                    self, "Export Complete",
                    f"Results exported to:\n{exported_path}"
                )

                logger.info(f"Results exported to: {exported_path}")

        except Exception as e:
            error_logger.log_ui_error(e, "MainWindow", "export_results")
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")

    @pyqtSlot()
    def show_batch_dialog(self) -> None:
        """Show batch processing dialog."""
        try:
            dialog = BatchDialog(self)
            if dialog.exec_() == dialog.Accepted:
                # Batch processing was completed in dialog
                pass
        except Exception as e:
            error_logger.log_ui_error(e, "MainWindow", "show_batch_dialog")
            QMessageBox.critical(self, "Error", f"Failed to open batch dialog: {str(e)}")

    @pyqtSlot()
    def show_settings(self) -> None:
        """Show settings dialog."""
        try:
            dialog = SettingsDialog(self)
            if dialog.exec_() == dialog.Accepted:
                # Reload configuration
                self.config = get_config()
                logger.info("Settings updated")
        except Exception as e:
            error_logger.log_ui_error(e, "MainWindow", "show_settings")
            QMessageBox.critical(self, "Error", f"Failed to open settings: {str(e)}")

    @pyqtSlot()
    def show_about(self) -> None:
        """Show about dialog."""
        about_text = f"""
        <h2>{self.config.get('app.name', 'OCR Processor')}</h2>
        <p>Version {self.config.get('app.version', '2.0.0')}</p>
        <p>Advanced OCR processing with LLM correction capabilities.</p>
        <p>Supports multiple OCR engines and intelligent text comparison.</p>
        <br>
        <p><b>Features:</b></p>
        <ul>
        <li>EasyOCR and Tesseract integration</li>
        <li>LLM-powered text correction</li>
        <li>Batch processing</li>
        <li>Multiple export formats</li>
        <li>Advanced text comparison</li>
        </ul>
        """

        QMessageBox.about(self, "About", about_text)

    def _add_recent_file(self, file_path: str) -> None:
        """Add file to recent files list."""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)

        self.recent_files.insert(0, file_path)

        # Limit recent files
        max_recent = self.config.get('app.max_recent_files', 10)
        self.recent_files = self.recent_files[:max_recent]

        self._update_recent_menu()

    def _update_recent_menu(self) -> None:
        """Update recent files menu."""
        self.recent_menu.clear()

        for file_path in self.recent_files:
            if os.path.exists(file_path):
                action = QAction(os.path.basename(file_path), self)
                action.setToolTip(file_path)
                action.triggered.connect(lambda checked, path=file_path: self._load_image_file(path))
                self.recent_menu.addAction(action)

        if not self.recent_files:
            action = QAction("No recent files", self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)

    def dragEnterEvent(self, event) -> None:
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                if any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']):
                    event.accept()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        """Handle drop event."""
        try:
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                self._load_image_file(file_path)
        except Exception as e:
            error_logger.log_ui_error(e, "MainWindow", "drop_event")
            QMessageBox.warning(self, "Error", f"Failed to load dropped file: {str(e)}")

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        try:
            # Stop any running worker
            if self.current_worker and self.current_worker.isRunning():
                self.current_worker.stop()
                self.current_worker.wait(3000)  # Wait up to 3 seconds

            # Save settings
            # This would typically save window geometry, recent files, etc.

            event.accept()
            logger.info("Application closed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            event.accept()
