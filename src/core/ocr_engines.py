"""OCR engines module."""

import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

from ..utils.config import get_config
from ..utils.logging_config import ErrorLogger
from .image_processors import ImageProcessor

logger = logging.getLogger(__name__)
error_logger = ErrorLogger()


class OCRResult:
    """Represents OCR extraction result."""
    
    def __init__(self, text: str, confidence: float, bbox: Optional[List] = None, 
                 engine: str = "", metadata: Optional[Dict] = None):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox or []
        self.engine = engine
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return f"OCRResult(text='{self.text[:50]}...', confidence={self.confidence:.3f}, engine={self.engine})"


class OCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    @abstractmethod
    def extract_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Extract text from image.
        
        Args:
            image: Input image as numpy array
            **kwargs: Engine-specific parameters
            
        Returns:
            List of OCR results
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the OCR engine is available."""
        pass


class EasyOCREngine(OCREngine):
    """EasyOCR implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize EasyOCR engine.
        
        Args:
            config: Engine configuration. If None, uses global config.
        """
        self.config = config or get_config().get_ocr_config().get('engines', {}).get('easyocr', {})
        self.reader = None
        self._initialize_reader()
    
    def _initialize_reader(self) -> None:
        """Initialize EasyOCR reader."""
        try:
            languages = self.config.get('languages', ['en'])
            gpu = self.config.get('gpu', False)
            
            self.reader = easyocr.Reader(languages, gpu=gpu)
            logger.info(f"EasyOCR initialized with languages: {languages}, GPU: {gpu}")
            
        except Exception as e:
            error_logger.log_ocr_error(e, "", "EasyOCR", "initialization")
            self.reader = None
    
    def extract_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Extract text using EasyOCR.

        Args:
            image: Input image
            **kwargs: Additional parameters

        Returns:
            List of OCR results
        """
        if not self.is_available():
            return []

        try:
            # Convert BGR to RGB for EasyOCR
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Debug: Log image properties
            logger.debug(f"EasyOCR input image shape: {rgb_image.shape}")
            logger.debug(f"Image dtype: {rgb_image.dtype}, min: {rgb_image.min()}, max: {rgb_image.max()}")

            # EasyOCR parameters - more conservative settings for better accuracy
            detail = kwargs.get('detail', 1)
            paragraph = kwargs.get('paragraph', False)
            batch_size = kwargs.get('batch_size', 1)  # Reduced for stability
            min_size = kwargs.get('min_size', 20)  # Increased to filter small noise
            text_threshold = kwargs.get('text_threshold', 0.8)  # Higher threshold
            contrast_threshold = kwargs.get('contrast_threshold', 0.2)  # Higher contrast
            width_ths = kwargs.get('width_ths', 0.8)  # Add width threshold
            height_ths = kwargs.get('height_ths', 0.8)  # Add height threshold

            logger.debug(f"EasyOCR parameters: text_threshold={text_threshold}, contrast_threshold={contrast_threshold}")

            # Perform OCR with improved parameters
            results = self.reader.readtext(
                rgb_image,
                detail=detail,
                paragraph=paragraph,
                batch_size=batch_size,
                min_size=min_size,
                text_threshold=text_threshold,
                contrast_ths=contrast_threshold,
                width_ths=width_ths,
                height_ths=height_ths,
                decoder='greedy',  # Use greedy decoder for better accuracy
                beamWidth=5,  # Beam search width
                allowlist=None  # Allow all characters initially
            )

            # Convert to OCRResult objects with better filtering
            ocr_results = []
            for bbox, text, confidence in results:
                # Clean and validate text
                cleaned_text = text.strip()

                # Filter out very short or suspicious results
                if (len(cleaned_text) >= 2 and  # At least 2 characters
                    confidence > 0.3 and  # Minimum confidence
                    not all(c in '.,!?;:' for c in cleaned_text)):  # Not just punctuation

                    ocr_results.append(OCRResult(
                        text=cleaned_text,
                        confidence=float(confidence),
                        bbox=bbox,
                        engine="EasyOCR",
                        metadata={
                            'text_threshold': text_threshold,
                            'contrast_threshold': contrast_threshold,
                            'width_ths': width_ths,
                            'height_ths': height_ths,
                            'original_text': text  # Keep original for debugging
                        }
                    ))

                    logger.debug(f"EasyOCR result: '{cleaned_text}' (confidence: {confidence:.3f})")
                else:
                    logger.debug(f"Filtered out: '{text}' (confidence: {confidence:.3f})")

            logger.info(f"EasyOCR extracted {len(ocr_results)} valid text regions from {len(results)} total detections")
            return ocr_results

        except Exception as e:
            error_logger.log_ocr_error(e, "", "EasyOCR", "text extraction")
            logger.error(f"EasyOCR extraction failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available."""
        return self.reader is not None


class TesseractEngine(OCREngine):
    """Tesseract OCR implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Tesseract engine.
        
        Args:
            config: Engine configuration. If None, uses global config.
        """
        self.config = config or get_config().get_ocr_config().get('engines', {}).get('tesseract', {})
    
    def extract_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Extract text using Tesseract.

        Args:
            image: Input image
            **kwargs: Additional parameters

        Returns:
            List of OCR results
        """
        if not self.is_available():
            return []

        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = PILImage.fromarray(image)

            # Debug: Log image properties
            logger.debug(f"Tesseract input image size: {pil_image.size}")

            # Tesseract configuration with improved defaults
            language = kwargs.get('lang', self.config.get('language', 'eng'))
            oem = kwargs.get('oem', self.config.get('oem', 1))  # LSTM OCR Engine
            psm = kwargs.get('psm', self.config.get('psm', 6))  # Changed to 6 (uniform block)
            char_whitelist = kwargs.get('char_whitelist', self.config.get('char_whitelist', ''))

            # Build config string with improved settings
            config_parts = [
                f'--oem {oem}',
                f'--psm {psm}',
                '-c tessedit_char_blacklist=',  # Remove character blacklist
                '-c preserve_interword_spaces=1',  # Preserve spaces
                '-c tessedit_do_invert=0',  # Don't auto-invert
            ]

            # Add character whitelist only if specified and reasonable
            if char_whitelist and len(char_whitelist) > 10:
                config_parts.append(f'-c tessedit_char_whitelist={char_whitelist}')

            config_string = ' '.join(config_parts)

            logger.debug(f"Tesseract config: {config_string}")

            # Try multiple approaches for better results
            approaches = [
                # Standard approach
                {'config': config_string, 'name': 'standard'},
                # More aggressive approach
                {'config': f'--oem {oem} --psm 8', 'name': 'single_word'},
                # Line-based approach
                {'config': f'--oem {oem} --psm 7', 'name': 'single_line'},
                # Sparse text approach
                {'config': f'--oem {oem} --psm 11', 'name': 'sparse'},
            ]

            best_result = None
            best_confidence = 0

            for approach in approaches:
                try:
                    # Extract text
                    text = pytesseract.image_to_string(
                        pil_image,
                        lang=language,
                        config=approach['config']
                    ).strip()

                    if text and len(text) > 1:  # At least 2 characters
                        # Get confidence data
                        try:
                            data = pytesseract.image_to_data(
                                pil_image,
                                lang=language,
                                config=approach['config'],
                                output_type=pytesseract.Output.DICT
                            )

                            # Calculate average confidence of words with confidence > 0
                            word_confidences = []
                            for i, conf in enumerate(data['conf']):
                                if int(conf) > 0 and data['text'][i].strip():
                                    word_confidences.append(int(conf))

                            avg_confidence = sum(word_confidences) / len(word_confidences) / 100 if word_confidences else 0.0

                        except Exception:
                            avg_confidence = 0.3  # Default confidence

                        # Check if this is the best result so far
                        if avg_confidence > best_confidence or best_result is None:
                            best_result = OCRResult(
                                text=text,
                                confidence=avg_confidence,
                                bbox=[],
                                engine="Tesseract",
                                metadata={
                                    'language': language,
                                    'oem': oem,
                                    'psm': psm,
                                    'config': approach['config'],
                                    'approach': approach['name'],
                                    'word_count': len(word_confidences) if 'word_confidences' in locals() else 0
                                }
                            )
                            best_confidence = avg_confidence

                        logger.debug(f"Tesseract {approach['name']}: '{text[:50]}...' (conf: {avg_confidence:.3f})")

                except Exception as e:
                    logger.debug(f"Tesseract {approach['name']} failed: {e}")
                    continue

            if best_result:
                logger.info(f"Tesseract best result: '{best_result.text[:50]}...' (approach: {best_result.metadata.get('approach')}, conf: {best_result.confidence:.3f})")
                return [best_result]
            else:
                logger.warning("Tesseract: No valid text extracted with any approach")
                return []

        except Exception as e:
            error_logger.log_ocr_error(e, "", "Tesseract", "text extraction")
            logger.error(f"Tesseract extraction failed: {e}")
            return []
    
    def extract_from_region(self, image: np.ndarray, bbox: List, **kwargs) -> OCRResult:
        """Extract text from a specific region using Tesseract.
        
        Args:
            image: Full image
            bbox: Bounding box coordinates
            **kwargs: Additional parameters
            
        Returns:
            OCR result for the region
        """
        try:
            # Crop the region
            processor = ImageProcessor()
            cropped = processor.crop_region(image, bbox, margin=kwargs.get('margin', 5))
            
            # Enhance the cropped region
            enhanced = processor.enhance_contrast(cropped)
            
            # Scale up for better recognition
            scale_factor = kwargs.get('scale_factor', 2)
            if scale_factor > 1:
                enhanced = cv2.resize(
                    enhanced, None, 
                    fx=scale_factor, fy=scale_factor, 
                    interpolation=cv2.INTER_CUBIC
                )
            
            # Extract text
            results = self.extract_text(enhanced, **kwargs)
            
            if results:
                result = results[0]
                result.bbox = bbox
                result.metadata['region_enhanced'] = True
                result.metadata['scale_factor'] = scale_factor
                return result
            
            return OCRResult("", 0.0, bbox, "Tesseract")
            
        except Exception as e:
            error_logger.log_ocr_error(e, "", "Tesseract", "region extraction")
            return OCRResult("", 0.0, bbox, "Tesseract")
    
    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False


class OCRManager:
    """Manages multiple OCR engines and provides unified interface."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OCR manager.
        
        Args:
            config: OCR configuration. If None, uses global config.
        """
        self.config = config or get_config().get_ocr_config()
        self.engines = {}
        self.image_processor = ImageProcessor()
        self._initialize_engines()
    
    def _initialize_engines(self) -> None:
        """Initialize available OCR engines."""
        engine_configs = self.config.get('engines', {})
        
        # Initialize EasyOCR
        if engine_configs.get('easyocr', {}).get('enabled', True):
            easyocr_engine = EasyOCREngine(engine_configs.get('easyocr'))
            if easyocr_engine.is_available():
                self.engines['easyocr'] = easyocr_engine
                logger.info("EasyOCR engine initialized")
        
        # Initialize Tesseract
        if engine_configs.get('tesseract', {}).get('enabled', True):
            tesseract_engine = TesseractEngine(engine_configs.get('tesseract'))
            if tesseract_engine.is_available():
                self.engines['tesseract'] = tesseract_engine
                logger.info("Tesseract engine initialized")
        
        if not self.engines:
            logger.warning("No OCR engines available!")
    
    def extract_text(self, image: np.ndarray, preprocess: bool = True) -> List[OCRResult]:
        """Extract text using all available engines.
        
        Args:
            image: Input image
            preprocess: Whether to preprocess the image
            
        Returns:
            Combined results from all engines
        """
        if not self.engines:
            logger.error("No OCR engines available")
            return []
        
        try:
            # Preprocess image if requested
            processed_image = self.image_processor.preprocess_image(image) if preprocess else image
            
            all_results = []
            confidence_threshold = self.config.get('confidence_threshold', 0.5)
            
            # Primary extraction with EasyOCR
            if 'easyocr' in self.engines:
                easyocr_results = self.engines['easyocr'].extract_text(processed_image)
                all_results.extend(easyocr_results)
                
                # Use Tesseract for low-confidence regions if enabled
                if ('tesseract' in self.engines and 
                    self.config.get('fallback_to_tesseract', True)):
                    
                    for i, result in enumerate(easyocr_results):
                        if result.confidence < confidence_threshold and result.bbox:
                            try:
                                tesseract_result = self.engines['tesseract'].extract_from_region(
                                    processed_image, result.bbox
                                )
                                
                                if (tesseract_result.text and 
                                    len(tesseract_result.text) > len(result.text) * 0.5):
                                    # Replace with Tesseract result if it seems better
                                    all_results[i] = tesseract_result
                                    logger.debug(f"Replaced low-confidence result with Tesseract")
                                    
                            except Exception as e:
                                logger.warning(f"Tesseract fallback failed: {e}")
            
            # If no EasyOCR, use Tesseract only
            elif 'tesseract' in self.engines:
                tesseract_results = self.engines['tesseract'].extract_text(processed_image)
                all_results.extend(tesseract_results)
            
            logger.info(f"OCR extraction completed: {len(all_results)} results")
            return all_results
            
        except Exception as e:
            error_logger.log_ocr_error(e, "", "OCRManager", "text extraction")
            return []
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engine names."""
        return list(self.engines.keys())
    
    def get_engine_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available engines."""
        info = {}
        for name, engine in self.engines.items():
            info[name] = {
                'available': engine.is_available(),
                'config': getattr(engine, 'config', {})
            }
        return info
