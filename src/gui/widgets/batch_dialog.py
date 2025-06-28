"""Batch processing dialog."""

import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QTextEdit, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QComboBox, QCheckBox, QDialogButtonBox, QTabWidget, QWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from ...core.batch_processor import BatchProcessor, BatchResult
from ...utils.export import ExportManager
from ...utils.logging_config import ErrorLogger

logger = logging.getLogger(__name__)
error_logger = ErrorLogger()


class BatchWorker(QThread):
    """Worker thread for batch processing."""
    
    # Signals
    finished = pyqtSignal(list, dict)  # results, summary
    progress_update = pyqtSignal(int, int)  # current, total
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, processor: BatchProcessor, folder_path: str, 
                 expected_texts: Optional[Dict[str, str]] = None):
        """Initialize batch worker.
        
        Args:
            processor: Batch processor instance
            folder_path: Path to folder containing images
            expected_texts: Dictionary mapping filenames to expected texts
        """
        super().__init__()
        self.processor = processor
        self.folder_path = folder_path
        self.expected_texts = expected_texts or {}
    
    def run(self) -> None:
        """Run batch processing."""
        try:
            # Process folder
            results = self.processor.process_folder(
                self.folder_path,
                self.expected_texts,
                self.progress_update.emit
            )
            
            # Generate summary
            summary = self.processor.generate_summary_report(results)
            
            self.finished.emit(results, summary)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.error_occurred.emit(str(e))


class BatchDialog(QDialog):
    """Dialog for batch processing configuration and execution."""
    
    def __init__(self, parent=None):
        """Initialize batch dialog."""
        super().__init__(parent)
        
        self.batch_processor = BatchProcessor()
        self.export_manager = ExportManager()
        self.batch_worker = None
        self.results = []
        self.summary = {}
        
        self.setWindowTitle("Batch Processing")
        self.setModal(True)
        self.resize(800, 600)
        
        self._setup_ui()
        
        logger.debug("Batch dialog initialized")
    
    def _setup_ui(self) -> None:
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Setup tabs
        self._create_input_tab()
        self._create_results_tab()
        
        # Button box
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Close
        )
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
    
    def _create_input_tab(self) -> None:
        """Create input configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Folder selection
        folder_group = QGroupBox("Image Folder")
        folder_layout = QVBoxLayout(folder_group)
        
        folder_select_layout = QHBoxLayout()
        self.folder_path_edit = QTextEdit()
        self.folder_path_edit.setMaximumHeight(60)
        self.folder_path_edit.setPlaceholderText("Select folder containing images...")
        folder_select_layout.addWidget(self.folder_path_edit)
        
        folder_browse_btn = QPushButton("Browse...")
        folder_browse_btn.clicked.connect(self._browse_folder)
        folder_select_layout.addWidget(folder_browse_btn)
        
        folder_layout.addLayout(folder_select_layout)
        layout.addWidget(folder_group)
        
        # Expected texts
        expected_group = QGroupBox("Expected Texts (Optional)")
        expected_layout = QVBoxLayout(expected_group)
        
        expected_info = QLabel(
            "You can provide expected texts for comparison. "
            "Format: filename.jpg=Expected text for this image"
        )
        expected_info.setWordWrap(True)
        expected_layout.addWidget(expected_info)
        
        self.expected_texts_edit = QTextEdit()
        self.expected_texts_edit.setPlaceholderText(
            "image1.jpg=This is the expected text for image1\n"
            "image2.png=Expected text for image2\n"
            "..."
        )
        expected_layout.addWidget(self.expected_texts_edit)
        
        expected_file_layout = QHBoxLayout()
        expected_load_btn = QPushButton("Load from File...")
        expected_load_btn.clicked.connect(self._load_expected_texts)
        expected_file_layout.addWidget(expected_load_btn)
        
        expected_save_btn = QPushButton("Save to File...")
        expected_save_btn.clicked.connect(self._save_expected_texts)
        expected_file_layout.addWidget(expected_save_btn)
        
        expected_file_layout.addStretch()
        expected_layout.addLayout(expected_file_layout)
        
        layout.addWidget(expected_group)
        
        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout(options_group)
        
        self.parallel_check = QCheckBox("Enable parallel processing")
        self.parallel_check.setChecked(self.batch_processor.parallel_processing)
        options_layout.addWidget(self.parallel_check)
        
        layout.addWidget(options_group)
        
        # Progress and controls
        progress_group = QGroupBox("Processing")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self._start_processing)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        control_layout.addStretch()
        progress_layout.addLayout(control_layout)
        
        layout.addWidget(progress_group)
        
        self.tabs.addTab(tab, "Input")
    
    def _create_results_tab(self) -> None:
        """Create results display tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Summary
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        
        # Results table
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Image", "Success", "Similarity", "Confidence", "Errors", "Time (s)"
        ])
        
        # Configure table
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Image column
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Success
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Similarity
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Confidence
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Errors
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Time
        
        results_layout.addWidget(self.results_table)
        
        # Export controls
        export_layout = QHBoxLayout()
        
        export_format_label = QLabel("Export format:")
        export_layout.addWidget(export_format_label)
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(self.export_manager.get_supported_formats())
        export_layout.addWidget(self.export_format_combo)
        
        export_btn = QPushButton("Export Results...")
        export_btn.clicked.connect(self._export_results)
        export_layout.addWidget(export_btn)
        
        export_layout.addStretch()
        results_layout.addLayout(export_layout)
        
        layout.addWidget(results_group)
        
        self.tabs.addTab(tab, "Results")
    
    @pyqtSlot()
    def _browse_folder(self) -> None:
        """Browse for image folder."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", ""
        )
        if folder_path:
            self.folder_path_edit.setPlainText(folder_path)
    
    @pyqtSlot()
    def _load_expected_texts(self) -> None:
        """Load expected texts from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Expected Texts", "", 
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.expected_texts_edit.setPlainText(content)
                
            except Exception as e:
                error_logger.log_ui_error(e, "BatchDialog", "load_expected_texts")
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    @pyqtSlot()
    def _save_expected_texts(self) -> None:
        """Save expected texts to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Expected Texts", "expected_texts.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.expected_texts_edit.toPlainText())
                
                QMessageBox.information(self, "Success", "Expected texts saved successfully.")
                
            except Exception as e:
                error_logger.log_ui_error(e, "BatchDialog", "save_expected_texts")
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
    
    def _parse_expected_texts(self) -> Dict[str, str]:
        """Parse expected texts from the text edit."""
        expected_texts = {}
        content = self.expected_texts_edit.toPlainText().strip()
        
        if not content:
            return expected_texts
        
        try:
            for line in content.split('\n'):
                line = line.strip()
                if '=' in line:
                    filename, text = line.split('=', 1)
                    expected_texts[filename.strip()] = text.strip()
            
        except Exception as e:
            logger.warning(f"Error parsing expected texts: {e}")
        
        return expected_texts
    
    @pyqtSlot()
    def _start_processing(self) -> None:
        """Start batch processing."""
        folder_path = self.folder_path_edit.toPlainText().strip()
        
        if not folder_path:
            QMessageBox.warning(self, "Warning", "Please select an image folder.")
            return
        
        if not os.path.exists(folder_path):
            QMessageBox.warning(self, "Warning", "Selected folder does not exist.")
            return
        
        try:
            # Update processor settings
            self.batch_processor.parallel_processing = self.parallel_check.isChecked()
            
            # Parse expected texts
            expected_texts = self._parse_expected_texts()
            
            # Update UI
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Start worker
            self.batch_worker = BatchWorker(
                self.batch_processor, folder_path, expected_texts
            )
            self.batch_worker.finished.connect(self._on_processing_finished)
            self.batch_worker.progress_update.connect(self._on_progress_update)
            self.batch_worker.error_occurred.connect(self._on_processing_error)
            self.batch_worker.start()
            
            logger.info(f"Started batch processing: {folder_path}")
            
        except Exception as e:
            error_logger.log_ui_error(e, "BatchDialog", "start_processing")
            self._reset_processing_ui()
            QMessageBox.critical(self, "Error", f"Failed to start processing: {str(e)}")
    
    @pyqtSlot()
    def _stop_processing(self) -> None:
        """Stop batch processing."""
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.terminate()
            self.batch_worker.wait(3000)
        
        self._reset_processing_ui()
        logger.info("Batch processing stopped")
    
    @pyqtSlot(list, dict)
    def _on_processing_finished(self, results: List[BatchResult], summary: Dict) -> None:
        """Handle processing completion."""
        try:
            self.results = results
            self.summary = summary
            
            # Update summary display
            self._update_summary_display()
            
            # Update results table
            self._update_results_table()
            
            # Switch to results tab
            self.tabs.setCurrentIndex(1)
            
            QMessageBox.information(
                self, "Processing Complete", 
                f"Processed {len(results)} images successfully."
            )
            
            logger.info(f"Batch processing completed: {len(results)} results")
            
        except Exception as e:
            error_logger.log_ui_error(e, "BatchDialog", "processing_finished")
            QMessageBox.warning(self, "Warning", f"Error displaying results: {str(e)}")
        
        finally:
            self._reset_processing_ui()
    
    @pyqtSlot(int, int)
    def _on_progress_update(self, current: int, total: int) -> None:
        """Handle progress updates."""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
    
    @pyqtSlot(str)
    def _on_processing_error(self, error_message: str) -> None:
        """Handle processing errors."""
        QMessageBox.critical(self, "Processing Error", error_message)
        self._reset_processing_ui()
    
    def _reset_processing_ui(self) -> None:
        """Reset UI after processing."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.batch_worker = None
    
    def _update_summary_display(self) -> None:
        """Update summary text display."""
        if not self.summary:
            return
        
        summary_text = []
        
        if 'summary' in self.summary:
            s = self.summary['summary']
            summary_text.extend([
                f"Total Images: {s.get('total_images', 0)}",
                f"Successful Extractions: {s.get('successful_extractions', 0)}",
                f"Success Rate: {s.get('success_rate', 0):.1%}",
                f"Total Errors: {s.get('total_errors', 0)}",
                f"Average Processing Time: {s.get('average_processing_time', 0):.2f}s",
                f"Average Confidence: {s.get('average_confidence', 0):.3f}",
                f"Average Similarity: {s.get('average_similarity', 0):.3f}",
                ""
            ])
        
        if 'error_analysis' in self.summary:
            summary_text.append("Error Analysis:")
            for error_type, count in self.summary['error_analysis'].items():
                summary_text.append(f"  {error_type.replace('_', ' ').title()}: {count}")
        
        self.summary_text.setPlainText('\n'.join(summary_text))
    
    def _update_results_table(self) -> None:
        """Update results table."""
        self.results_table.setRowCount(len(self.results))
        
        for row, result in enumerate(self.results):
            # Image name
            image_name = os.path.basename(result.image_path)
            self.results_table.setItem(row, 0, QTableWidgetItem(image_name))
            
            # Success
            success_text = "✓" if result.success else "✗"
            success_item = QTableWidgetItem(success_text)
            success_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 1, success_item)
            
            # Similarity
            similarity_text = f"{result.similarity:.3f}" if result.similarity > 0 else "N/A"
            similarity_item = QTableWidgetItem(similarity_text)
            similarity_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 2, similarity_item)
            
            # Confidence
            confidence_text = f"{result.ocr_confidence:.3f}" if result.ocr_confidence > 0 else "N/A"
            confidence_item = QTableWidgetItem(confidence_text)
            confidence_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 3, confidence_item)
            
            # Error count
            error_count = len(result.errors)
            error_item = QTableWidgetItem(str(error_count))
            error_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 4, error_item)
            
            # Processing time
            time_text = f"{result.processing_time:.2f}"
            time_item = QTableWidgetItem(time_text)
            time_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 5, time_item)
    
    @pyqtSlot()
    def _export_results(self) -> None:
        """Export batch results."""
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to export.")
            return
        
        try:
            format_type = self.export_format_combo.currentText()
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Results", f"batch_results.{format_type}",
                f"{format_type.upper()} Files (*.{format_type});;All Files (*)"
            )
            
            if file_path:
                exported_path = self.export_manager.export_results(
                    self.results, format_type, os.path.basename(file_path), self.summary
                )
                
                QMessageBox.information(
                    self, "Export Complete",
                    f"Results exported to:\n{exported_path}"
                )
                
                logger.info(f"Batch results exported to: {exported_path}")
                
        except Exception as e:
            error_logger.log_ui_error(e, "BatchDialog", "export_results")
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
