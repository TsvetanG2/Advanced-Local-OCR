"""OCR processing worker for asynchronous operations."""

import time
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
import logging
from PyQt5.QtCore import QThread, pyqtSignal, QMutex

from .ocr_engines import OCRManager
from .llm_corrector import LLMCorrector
from .text_processors import TextComparator
from .image_processors import ImageProcessor
from ..utils.config import get_config
from ..utils.logging_config import ErrorLogger, PerformanceLogger

logger = logging.getLogger(__name__)
error_logger = ErrorLogger()
performance_logger = PerformanceLogger("ocr_worker")


class OCRWorker(QThread):
    """Worker thread for asynchronous OCR processing."""
    
    # Signals
    finished = pyqtSignal(dict)  # Processing results
    progress_update = pyqtSignal(int, int, str)  # current, total, status
    error_occurred = pyqtSignal(str)  # Error message
    
    def __init__(self, image_path: str, expected_text: str = "", 
                 config: Optional[Dict[str, Any]] = None):
        """Initialize OCR worker.
        
        Args:
            image_path: Path to image file
            expected_text: Expected text for comparison
            config: Processing configuration
        """
        super().__init__()
        self.image_path = image_path
        self.expected_text = expected_text
        self.config = config or get_config()
        
        # Processing components
        self.ocr_manager = None
        self.llm_corrector = None
        self.text_comparator = None
        self.image_processor = None
        
        # Thread safety
        self.mutex = QMutex()
        self._should_stop = False
    
    def stop(self) -> None:
        """Request worker to stop processing."""
        self.mutex.lock()
        self._should_stop = True
        self.mutex.unlock()
    
    def should_stop(self) -> bool:
        """Check if worker should stop."""
        self.mutex.lock()
        should_stop = self._should_stop
        self.mutex.unlock()
        return should_stop
    
    def run(self) -> None:
        """Main processing method."""
        start_time = time.time()
        
        try:
            # Initialize processing components
            self.progress_update.emit(0, 6, "Initializing components...")
            self._initialize_components()
            
            if self.should_stop():
                return
            
            # Load and validate image
            self.progress_update.emit(1, 6, "Loading image...")
            image = self._load_image()
            
            if self.should_stop():
                return
            
            # Preprocess image
            self.progress_update.emit(2, 6, "Preprocessing image...")
            processed_image = self._preprocess_image(image)
            
            if self.should_stop():
                return
            
            # Extract text using OCR
            self.progress_update.emit(3, 6, "Extracting text...")
            ocr_results = self._extract_text(processed_image)
            
            if self.should_stop():
                return
            
            # Apply LLM correction if enabled
            self.progress_update.emit(4, 6, "Applying corrections...")
            corrected_text = self._apply_llm_correction(ocr_results)
            
            if self.should_stop():
                return
            
            # Compare with expected text
            self.progress_update.emit(5, 6, "Comparing texts...")
            comparison_results = self._compare_texts(corrected_text)
            
            # Prepare final results
            self.progress_update.emit(6, 6, "Finalizing results...")
            results = self._prepare_results(
                image, ocr_results, corrected_text, 
                comparison_results, time.time() - start_time
            )
            
            # Log performance
            performance_logger.log_ocr_performance(
                self.image_path,
                results['processing_time'],
                results['metadata'].get('memory_used', 0),
                len(results['extracted_text'])
            )
            
            self.finished.emit(results)
            
        except Exception as e:
            error_message = f"OCR processing failed: {str(e)}"
            logger.error(error_message, exc_info=True)
            error_logger.log_ocr_error(e, self.image_path, "OCRWorker", "main processing")
            self.error_occurred.emit(error_message)
    
    def _initialize_components(self) -> None:
        """Initialize processing components."""
        try:
            self.ocr_manager = OCRManager()
            self.llm_corrector = LLMCorrector()
            self.text_comparator = TextComparator()
            self.image_processor = ImageProcessor()
            
            logger.debug("OCR worker components initialized")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize components: {e}")
    
    def _load_image(self) -> np.ndarray:
        """Load and validate image file."""
        try:
            # Handle Unicode file paths
            data = np.fromfile(self.image_path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image file")
            
            # Validate image size
            max_size = self.config.get('performance', {}).get('max_image_size', 4096)
            h, w = image.shape[:2]
            
            if max(h, w) > max_size:
                logger.warning(f"Image size ({w}x{h}) exceeds maximum ({max_size})")
                # Resize to fit within limits
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.info(f"Image resized to {new_w}x{new_h}")
            
            logger.debug(f"Image loaded: {w}x{h} -> {image.shape[1]}x{image.shape[0]}")
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for OCR."""
        try:
            processed = self.image_processor.preprocess_image(image)
            logger.debug("Image preprocessing completed")
            return processed
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}, using original")
            return image
    
    def _extract_text(self, image: np.ndarray) -> List:
        """Extract text using OCR engines."""
        try:
            ocr_results = self.ocr_manager.extract_text(image)
            
            if not ocr_results:
                logger.warning("No text extracted from image")
                return []
            
            logger.debug(f"OCR extracted {len(ocr_results)} text regions")
            return ocr_results
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []
    
    def _apply_llm_correction(self, ocr_results: List) -> str:
        """Apply LLM correction to extracted text."""
        if not ocr_results:
            return ""
        
        try:
            # Combine all extracted text
            extracted_text = ' '.join([result.text for result in ocr_results])
            
            # Calculate average confidence
            confidences = [result.confidence for result in ocr_results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Apply LLM correction if enabled
            if self.llm_corrector.is_enabled():
                corrected_text = self.llm_corrector.correct_text(
                    extracted_text,
                    avg_confidence,
                    self.expected_text[:200]  # Use expected text as context
                )
                logger.debug("LLM correction applied")
                return corrected_text
            else:
                logger.debug("LLM correction disabled, using original text")
                return extracted_text
                
        except Exception as e:
            logger.error(f"LLM correction failed: {e}")
            # Return original text if correction fails
            return ' '.join([result.text for result in ocr_results]) if ocr_results else ""
    
    def _compare_texts(self, extracted_text: str) -> Dict[str, Any]:
        """Compare extracted text with expected text."""
        if not self.expected_text.strip():
            return {
                'errors': [],
                'similarity': 1.0,
                'metrics': {}
            }
        
        try:
            comparison = self.text_comparator.compare_texts(
                self.expected_text,
                extracted_text,
                tokenization_mode='standard'  # Can be made configurable
            )
            
            logger.debug(f"Text comparison completed: similarity={comparison['similarity']:.3f}")
            return comparison
            
        except Exception as e:
            logger.error(f"Text comparison failed: {e}")
            return {
                'errors': [f"Comparison failed: {str(e)}"],
                'similarity': 0.0,
                'metrics': {}
            }
    
    def _prepare_results(self, original_image: np.ndarray, ocr_results: List,
                        corrected_text: str, comparison_results: Dict[str, Any],
                        processing_time: float) -> Dict[str, Any]:
        """Prepare final processing results."""
        try:
            # Calculate metrics
            avg_confidence = 0.0
            if ocr_results:
                confidences = [result.confidence for result in ocr_results]
                avg_confidence = sum(confidences) / len(confidences)
            
            # Get image quality metrics
            quality_metrics = self.image_processor.get_image_quality_metrics(original_image)
            
            # Memory usage (approximate)
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            results = {
                'success': bool(ocr_results),
                'image_path': self.image_path,
                'expected_text': self.expected_text,
                'extracted_text': ' '.join([r.text for r in ocr_results]) if ocr_results else "",
                'corrected_text': corrected_text,
                'ocr_confidence': avg_confidence,
                'similarity': comparison_results.get('similarity', 0.0),
                'errors': comparison_results.get('errors', []),
                'processing_time': processing_time,
                'metadata': {
                    'ocr_engines': self.ocr_manager.get_available_engines() if self.ocr_manager else [],
                    'llm_enabled': self.llm_corrector.is_enabled() if self.llm_corrector else False,
                    'image_size': f"{original_image.shape[1]}x{original_image.shape[0]}",
                    'ocr_regions': len(ocr_results),
                    'comparison_metrics': comparison_results.get('metrics', {}),
                    'quality_metrics': quality_metrics,
                    'memory_used': memory_info.rss,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error preparing results: {e}")
            # Return minimal results
            return {
                'success': False,
                'image_path': self.image_path,
                'expected_text': self.expected_text,
                'extracted_text': "",
                'corrected_text': "",
                'ocr_confidence': 0.0,
                'similarity': 0.0,
                'errors': [f"Result preparation failed: {str(e)}"],
                'processing_time': processing_time,
                'metadata': {}
            }
