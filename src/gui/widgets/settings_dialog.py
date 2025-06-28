"""Settings dialog for application configuration."""

import os
from typing import Dict, Any
import logging

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QTextEdit,
    QFileDialog, QMessageBox, QFormLayout, QDialogButtonBox
)
from PyQt5.QtCore import Qt

from ...utils.config import get_config
from ...utils.logging_config import ErrorLogger

logger = logging.getLogger(__name__)
error_logger = ErrorLogger()


class SettingsDialog(QDialog):
    """Settings configuration dialog."""
    
    def __init__(self, parent=None):
        """Initialize settings dialog."""
        super().__init__(parent)
        
        self.config = get_config()
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(600, 500)
        
        # Setup UI
        self._setup_ui()
        self._load_current_settings()
        
        logger.debug("Settings dialog initialized")
    
    def _setup_ui(self) -> None:
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tabs
        self._create_general_tab()
        self._create_ocr_tab()
        self._create_llm_tab()
        self._create_ui_tab()
        self._create_performance_tab()
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self._apply_settings)
        layout.addWidget(button_box)
    
    def _create_general_tab(self) -> None:
        """Create general settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Application settings
        app_group = QGroupBox("Application")
        app_layout = QFormLayout(app_group)
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English"])  # Extensible for future languages
        app_layout.addRow("Language:", self.language_combo)
        
        self.auto_save_check = QCheckBox("Auto-save settings")
        app_layout.addRow(self.auto_save_check)
        
        self.max_recent_spin = QSpinBox()
        self.max_recent_spin.setRange(1, 50)
        app_layout.addRow("Max recent files:", self.max_recent_spin)
        
        layout.addWidget(app_group)
        
        # Logging settings
        log_group = QGroupBox("Logging")
        log_layout = QFormLayout(log_group)
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        log_layout.addRow("Log level:", self.log_level_combo)
        
        self.log_file_edit = QLineEdit()
        log_file_layout = QHBoxLayout()
        log_file_layout.addWidget(self.log_file_edit)
        
        log_browse_btn = QPushButton("Browse...")
        log_browse_btn.clicked.connect(self._browse_log_file)
        log_file_layout.addWidget(log_browse_btn)
        
        log_layout.addRow("Log file:", log_file_layout)
        
        layout.addWidget(log_group)
        
        layout.addStretch()
        self.tabs.addTab(tab, "General")
    
    def _create_ocr_tab(self) -> None:
        """Create OCR settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # EasyOCR settings
        easyocr_group = QGroupBox("EasyOCR")
        easyocr_layout = QFormLayout(easyocr_group)
        
        self.easyocr_enabled_check = QCheckBox("Enable EasyOCR")
        easyocr_layout.addRow(self.easyocr_enabled_check)
        
        self.easyocr_gpu_check = QCheckBox("Use GPU acceleration")
        easyocr_layout.addRow(self.easyocr_gpu_check)
        
        self.text_threshold_spin = QDoubleSpinBox()
        self.text_threshold_spin.setRange(0.1, 1.0)
        self.text_threshold_spin.setSingleStep(0.1)
        self.text_threshold_spin.setDecimals(1)
        easyocr_layout.addRow("Text threshold:", self.text_threshold_spin)
        
        self.contrast_threshold_spin = QDoubleSpinBox()
        self.contrast_threshold_spin.setRange(0.1, 1.0)
        self.contrast_threshold_spin.setSingleStep(0.1)
        self.contrast_threshold_spin.setDecimals(1)
        easyocr_layout.addRow("Contrast threshold:", self.contrast_threshold_spin)
        
        layout.addWidget(easyocr_group)
        
        # Tesseract settings
        tesseract_group = QGroupBox("Tesseract")
        tesseract_layout = QFormLayout(tesseract_group)
        
        self.tesseract_enabled_check = QCheckBox("Enable Tesseract")
        tesseract_layout.addRow(self.tesseract_enabled_check)
        
        self.tesseract_oem_spin = QSpinBox()
        self.tesseract_oem_spin.setRange(0, 3)
        tesseract_layout.addRow("OEM mode:", self.tesseract_oem_spin)
        
        self.tesseract_psm_spin = QSpinBox()
        self.tesseract_psm_spin.setRange(0, 13)
        tesseract_layout.addRow("PSM mode:", self.tesseract_psm_spin)
        
        layout.addWidget(tesseract_group)
        
        # General OCR settings
        general_group = QGroupBox("General OCR")
        general_layout = QFormLayout(general_group)
        
        self.confidence_threshold_spin = QDoubleSpinBox()
        self.confidence_threshold_spin.setRange(0.0, 1.0)
        self.confidence_threshold_spin.setSingleStep(0.05)
        self.confidence_threshold_spin.setDecimals(2)
        general_layout.addRow("Confidence threshold:", self.confidence_threshold_spin)
        
        self.fallback_tesseract_check = QCheckBox("Use Tesseract for low-confidence regions")
        general_layout.addRow(self.fallback_tesseract_check)
        
        layout.addWidget(general_group)
        
        layout.addStretch()
        self.tabs.addTab(tab, "OCR")
    
    def _create_llm_tab(self) -> None:
        """Create LLM settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # LLM provider settings
        provider_group = QGroupBox("LLM Provider")
        provider_layout = QFormLayout(provider_group)
        
        self.llm_provider_combo = QComboBox()
        self.llm_provider_combo.addItems(["disabled", "openai", "anthropic", "local"])
        self.llm_provider_combo.currentTextChanged.connect(self._on_llm_provider_changed)
        provider_layout.addRow("Provider:", self.llm_provider_combo)
        
        self.llm_enabled_check = QCheckBox("Enable LLM correction")
        provider_layout.addRow(self.llm_enabled_check)
        
        layout.addWidget(provider_group)
        
        # OpenAI settings
        self.openai_group = QGroupBox("OpenAI Settings")
        openai_layout = QFormLayout(self.openai_group)
        
        self.openai_model_combo = QComboBox()
        self.openai_model_combo.addItems([
            "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"
        ])
        openai_layout.addRow("Model:", self.openai_model_combo)
        
        self.openai_api_key_edit = QLineEdit()
        self.openai_api_key_edit.setEchoMode(QLineEdit.Password)
        self.openai_api_key_edit.setPlaceholderText("Set via OPENAI_API_KEY environment variable")
        openai_layout.addRow("API Key:", self.openai_api_key_edit)
        
        self.openai_max_tokens_spin = QSpinBox()
        self.openai_max_tokens_spin.setRange(100, 4000)
        openai_layout.addRow("Max tokens:", self.openai_max_tokens_spin)
        
        layout.addWidget(self.openai_group)
        
        # Anthropic settings
        self.anthropic_group = QGroupBox("Anthropic Settings")
        anthropic_layout = QFormLayout(self.anthropic_group)
        
        self.anthropic_model_combo = QComboBox()
        self.anthropic_model_combo.addItems([
            "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"
        ])
        anthropic_layout.addRow("Model:", self.anthropic_model_combo)
        
        self.anthropic_api_key_edit = QLineEdit()
        self.anthropic_api_key_edit.setEchoMode(QLineEdit.Password)
        self.anthropic_api_key_edit.setPlaceholderText("Set via ANTHROPIC_API_KEY environment variable")
        anthropic_layout.addRow("API Key:", self.anthropic_api_key_edit)
        
        self.anthropic_max_tokens_spin = QSpinBox()
        self.anthropic_max_tokens_spin.setRange(100, 4000)
        anthropic_layout.addRow("Max tokens:", self.anthropic_max_tokens_spin)
        
        layout.addWidget(self.anthropic_group)
        
        # Local LLM settings
        self.local_group = QGroupBox("Local LLM Settings")
        local_layout = QFormLayout(self.local_group)
        
        self.local_endpoint_edit = QLineEdit()
        self.local_endpoint_edit.setPlaceholderText("http://localhost:11434")
        local_layout.addRow("Endpoint:", self.local_endpoint_edit)
        
        self.local_model_edit = QLineEdit()
        self.local_model_edit.setPlaceholderText("llama2")
        local_layout.addRow("Model:", self.local_model_edit)
        
        layout.addWidget(self.local_group)
        
        # Correction settings
        correction_group = QGroupBox("Correction Settings")
        correction_layout = QFormLayout(correction_group)
        
        self.llm_confidence_threshold_spin = QDoubleSpinBox()
        self.llm_confidence_threshold_spin.setRange(0.0, 1.0)
        self.llm_confidence_threshold_spin.setSingleStep(0.05)
        self.llm_confidence_threshold_spin.setDecimals(2)
        correction_layout.addRow("Confidence threshold:", self.llm_confidence_threshold_spin)
        
        self.context_window_spin = QSpinBox()
        self.context_window_spin.setRange(50, 1000)
        correction_layout.addRow("Context window:", self.context_window_spin)
        
        layout.addWidget(correction_group)
        
        layout.addStretch()
        self.tabs.addTab(tab, "LLM")
    
    def _create_ui_tab(self) -> None:
        """Create UI settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Theme settings
        theme_group = QGroupBox("Appearance")
        theme_layout = QFormLayout(theme_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["light", "dark"])
        theme_layout.addRow("Theme:", self.theme_combo)
        
        layout.addWidget(theme_group)
        
        # Window settings
        window_group = QGroupBox("Window")
        window_layout = QFormLayout(window_group)
        
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(800, 2000)
        window_layout.addRow("Default width:", self.window_width_spin)
        
        self.window_height_spin = QSpinBox()
        self.window_height_spin.setRange(600, 1500)
        window_layout.addRow("Default height:", self.window_height_spin)
        
        layout.addWidget(window_group)
        
        layout.addStretch()
        self.tabs.addTab(tab, "UI")
    
    def _create_performance_tab(self) -> None:
        """Create performance settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Processing settings
        processing_group = QGroupBox("Processing")
        processing_layout = QFormLayout(processing_group)
        
        self.max_image_size_spin = QSpinBox()
        self.max_image_size_spin.setRange(1024, 8192)
        self.max_image_size_spin.setSuffix(" px")
        processing_layout.addRow("Max image size:", self.max_image_size_spin)
        
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(30, 600)
        self.timeout_spin.setSuffix(" seconds")
        processing_layout.addRow("Processing timeout:", self.timeout_spin)
        
        self.parallel_processing_check = QCheckBox("Enable parallel processing")
        processing_layout.addRow(self.parallel_processing_check)
        
        layout.addWidget(processing_group)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Performance")
    
    def _load_current_settings(self) -> None:
        """Load current settings into the dialog."""
        try:
            # General settings
            self.language_combo.setCurrentText(self.config.get('app.language', 'en'))
            self.auto_save_check.setChecked(self.config.get('app.auto_save', True))
            self.max_recent_spin.setValue(self.config.get('app.max_recent_files', 10))
            
            self.log_level_combo.setCurrentText(self.config.get('logging.level', 'INFO'))
            self.log_file_edit.setText(self.config.get('logging.file', 'logs/ocr_app.log'))
            
            # OCR settings
            self.easyocr_enabled_check.setChecked(self.config.get('ocr.engines.easyocr.enabled', True))
            self.easyocr_gpu_check.setChecked(self.config.get('ocr.engines.easyocr.gpu', False))
            self.text_threshold_spin.setValue(self.config.get('ocr.engines.easyocr.text_threshold', 0.7))
            self.contrast_threshold_spin.setValue(self.config.get('ocr.engines.easyocr.contrast_threshold', 0.1))
            
            self.tesseract_enabled_check.setChecked(self.config.get('ocr.engines.tesseract.enabled', True))
            self.tesseract_oem_spin.setValue(self.config.get('ocr.engines.tesseract.oem', 1))
            self.tesseract_psm_spin.setValue(self.config.get('ocr.engines.tesseract.psm', 7))
            
            self.confidence_threshold_spin.setValue(self.config.get('ocr.confidence_threshold', 0.5))
            self.fallback_tesseract_check.setChecked(self.config.get('ocr.fallback_to_tesseract', True))
            
            # LLM settings
            self.llm_provider_combo.setCurrentText(self.config.get('llm.provider', 'disabled'))
            self.llm_enabled_check.setChecked(self.config.get('llm.correction.enabled', False))
            
            self.openai_model_combo.setCurrentText(self.config.get('llm.openai.model', 'gpt-3.5-turbo'))
            self.openai_max_tokens_spin.setValue(self.config.get('llm.openai.max_tokens', 1000))
            
            self.anthropic_model_combo.setCurrentText(self.config.get('llm.anthropic.model', 'claude-3-haiku-20240307'))
            self.anthropic_max_tokens_spin.setValue(self.config.get('llm.anthropic.max_tokens', 1000))
            
            self.local_endpoint_edit.setText(self.config.get('llm.local.endpoint', 'http://localhost:11434'))
            self.local_model_edit.setText(self.config.get('llm.local.model', 'llama2'))
            
            self.llm_confidence_threshold_spin.setValue(self.config.get('llm.correction.confidence_threshold', 0.6))
            self.context_window_spin.setValue(self.config.get('llm.correction.context_window', 200))
            
            # UI settings
            self.theme_combo.setCurrentText(self.config.get('ui.theme', 'light'))
            self.window_width_spin.setValue(self.config.get('ui.window.width', 1200))
            self.window_height_spin.setValue(self.config.get('ui.window.height', 800))
            
            # Performance settings
            self.max_image_size_spin.setValue(self.config.get('performance.max_image_size', 4096))
            self.timeout_spin.setValue(self.config.get('performance.timeout_seconds', 300))
            self.parallel_processing_check.setChecked(self.config.get('performance.parallel_processing', False))
            
            # Update LLM group visibility
            self._on_llm_provider_changed(self.llm_provider_combo.currentText())
            
        except Exception as e:
            error_logger.log_ui_error(e, "SettingsDialog", "load_settings")
            logger.error(f"Failed to load settings: {e}")
    
    def _on_llm_provider_changed(self, provider: str) -> None:
        """Handle LLM provider change."""
        self.openai_group.setVisible(provider == "openai")
        self.anthropic_group.setVisible(provider == "anthropic")
        self.local_group.setVisible(provider == "local")
    
    def _browse_log_file(self) -> None:
        """Browse for log file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Log File", self.log_file_edit.text(), "Log Files (*.log);;All Files (*)"
        )
        if file_path:
            self.log_file_edit.setText(file_path)
    
    def _apply_settings(self) -> None:
        """Apply current settings."""
        try:
            self._save_settings()
            QMessageBox.information(self, "Settings", "Settings applied successfully.")
        except Exception as e:
            error_logger.log_ui_error(e, "SettingsDialog", "apply_settings")
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {str(e)}")
    
    def _save_settings(self) -> None:
        """Save settings to configuration."""
        # General settings
        self.config.set('app.language', self.language_combo.currentText())
        self.config.set('app.auto_save', self.auto_save_check.isChecked())
        self.config.set('app.max_recent_files', self.max_recent_spin.value())
        
        self.config.set('logging.level', self.log_level_combo.currentText())
        self.config.set('logging.file', self.log_file_edit.text())
        
        # OCR settings
        self.config.set('ocr.engines.easyocr.enabled', self.easyocr_enabled_check.isChecked())
        self.config.set('ocr.engines.easyocr.gpu', self.easyocr_gpu_check.isChecked())
        self.config.set('ocr.engines.easyocr.text_threshold', self.text_threshold_spin.value())
        self.config.set('ocr.engines.easyocr.contrast_threshold', self.contrast_threshold_spin.value())
        
        self.config.set('ocr.engines.tesseract.enabled', self.tesseract_enabled_check.isChecked())
        self.config.set('ocr.engines.tesseract.oem', self.tesseract_oem_spin.value())
        self.config.set('ocr.engines.tesseract.psm', self.tesseract_psm_spin.value())
        
        self.config.set('ocr.confidence_threshold', self.confidence_threshold_spin.value())
        self.config.set('ocr.fallback_to_tesseract', self.fallback_tesseract_check.isChecked())
        
        # LLM settings
        self.config.set('llm.provider', self.llm_provider_combo.currentText())
        self.config.set('llm.correction.enabled', self.llm_enabled_check.isChecked())
        
        self.config.set('llm.openai.model', self.openai_model_combo.currentText())
        self.config.set('llm.openai.max_tokens', self.openai_max_tokens_spin.value())
        
        self.config.set('llm.anthropic.model', self.anthropic_model_combo.currentText())
        self.config.set('llm.anthropic.max_tokens', self.anthropic_max_tokens_spin.value())
        
        self.config.set('llm.local.endpoint', self.local_endpoint_edit.text())
        self.config.set('llm.local.model', self.local_model_edit.text())
        
        self.config.set('llm.correction.confidence_threshold', self.llm_confidence_threshold_spin.value())
        self.config.set('llm.correction.context_window', self.context_window_spin.value())
        
        # UI settings
        self.config.set('ui.theme', self.theme_combo.currentText())
        self.config.set('ui.window.width', self.window_width_spin.value())
        self.config.set('ui.window.height', self.window_height_spin.value())
        
        # Performance settings
        self.config.set('performance.max_image_size', self.max_image_size_spin.value())
        self.config.set('performance.timeout_seconds', self.timeout_spin.value())
        self.config.set('performance.parallel_processing', self.parallel_processing_check.isChecked())
        
        # Save to file
        self.config.save()
        
        logger.info("Settings saved successfully")
    
    def accept(self) -> None:
        """Accept dialog and save settings."""
        try:
            self._save_settings()
            super().accept()
        except Exception as e:
            error_logger.log_ui_error(e, "SettingsDialog", "accept")
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")
