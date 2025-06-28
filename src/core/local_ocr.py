"""
Local OCR Engine - Robust, reliable OCR without external dependencies.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image as PILImage

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from .text_cleaner import OCRTextCleaner
from .smart_text_cleaner import SmartTextCleaner

logger = logging.getLogger(__name__)


class LocalOCRResult:
    """Simple OCR result container."""
    
    def __init__(self, text: str, confidence: float, engine: str, bbox: Optional[List] = None):
        self.text = text.strip()
        self.confidence = confidence
        self.engine = engine
        self.bbox = bbox or []
    
    def __str__(self):
        return f"'{self.text}' ({self.engine}, conf: {self.confidence:.3f})"


class SimpleImageProcessor:
    """Simple, reliable image preprocessing."""
    
    @staticmethod
    def preprocess_for_ocr(image: np.ndarray, method: str = "auto") -> np.ndarray:
        """Preprocess image for OCR with proven methods."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            if method == "auto":
                # Auto-detect best preprocessing
                return SimpleImageProcessor._auto_preprocess(gray)
            elif method == "enhance":
                return SimpleImageProcessor._enhance_contrast(gray)
            elif method == "binary":
                return SimpleImageProcessor._simple_binary(gray)
            elif method == "adaptive":
                return SimpleImageProcessor._adaptive_threshold(gray)
            else:
                return gray
                
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return image
    
    @staticmethod
    def _auto_preprocess(gray: np.ndarray) -> np.ndarray:
        """Automatically choose best preprocessing method."""
        # Analyze image characteristics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        logger.debug(f"Image stats: brightness={brightness:.1f}, contrast={contrast:.1f}")
        
        # Choose preprocessing based on image characteristics
        if contrast < 50:
            # Low contrast - enhance it
            return SimpleImageProcessor._enhance_contrast(gray)
        elif brightness < 100 or brightness > 200:
            # Poor brightness - use adaptive threshold
            return SimpleImageProcessor._adaptive_threshold(gray)
        else:
            # Good image - minimal processing
            return SimpleImageProcessor._simple_binary(gray)
    
    @staticmethod
    def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
        """Enhance contrast using histogram equalization."""
        try:
            # Apply CLAHE for better local contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply binary threshold
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return binary
        except Exception:
            return gray
    
    @staticmethod
    def _simple_binary(gray: np.ndarray) -> np.ndarray:
        """Simple binary thresholding."""
        try:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        except Exception:
            return gray
    
    @staticmethod
    def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
        """Adaptive thresholding for varying lighting."""
        try:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            return binary
        except Exception:
            return gray


class LocalOCREngine:
    """Local OCR engine with fallback strategies."""
    
    def __init__(self):
        """Initialize local OCR engine."""
        self.easyocr_reader = None
        self.processor = SimpleImageProcessor()
        self.text_cleaner = OCRTextCleaner()
        self.smart_cleaner = SmartTextCleaner()

        # Initialize EasyOCR if available
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
                self.easyocr_reader = None

        # Check Tesseract availability
        if TESSERACT_AVAILABLE:
            try:
                pytesseract.get_tesseract_version()
                logger.info("Tesseract is available")
            except Exception as e:
                logger.warning(f"Tesseract not available: {e}")
    
    def extract_text(self, image: np.ndarray, preprocess: bool = True) -> List[LocalOCRResult]:
        """Extract text using available OCR engines."""
        results = []
        
        try:
            # Preprocess image if requested
            if preprocess:
                processed_image = self.processor.preprocess_for_ocr(image)
            else:
                processed_image = image
            
            # Try EasyOCR first
            if self.easyocr_reader:
                easyocr_results = self._extract_with_easyocr(image, processed_image)
                results.extend(easyocr_results)
            
            # Try Tesseract
            if TESSERACT_AVAILABLE:
                tesseract_results = self._extract_with_tesseract(image, processed_image)
                results.extend(tesseract_results)
            
            # If no results, try with different preprocessing
            if not results and preprocess:
                logger.info("No results with auto preprocessing, trying alternatives")
                
                for method in ["enhance", "binary", "adaptive"]:
                    alt_processed = self.processor.preprocess_for_ocr(image, method)
                    
                    if self.easyocr_reader:
                        alt_results = self._extract_with_easyocr(image, alt_processed)
                        if alt_results:
                            results.extend(alt_results)
                            break
                    
                    if TESSERACT_AVAILABLE:
                        alt_results = self._extract_with_tesseract(image, alt_processed)
                        if alt_results:
                            results.extend(alt_results)
                            break
            
            # Filter and sort results
            results = self._filter_results(results)

            logger.info(f"Local OCR extracted {len(results)} text regions")
            return results

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []

    def extract_text_cleaned(self, image: np.ndarray, preprocess: bool = True, use_smart_cleaner: bool = True) -> str:
        """Extract and clean text, returning a single cleaned string.

        Args:
            image: Input image
            preprocess: Whether to preprocess the image
            use_smart_cleaner: Whether to use the smart cleaner for better results

        Returns:
            Cleaned, deduplicated text string
        """
        # Get raw OCR results
        ocr_results = self.extract_text(image, preprocess)

        # Choose cleaner based on preference
        if use_smart_cleaner:
            cleaned_text = self.smart_cleaner.clean_ocr_results(ocr_results)
        else:
            cleaned_text = self.text_cleaner.clean_ocr_results(ocr_results)

        return cleaned_text
    
    def _extract_with_easyocr(self, original: np.ndarray, processed: np.ndarray) -> List[LocalOCRResult]:
        """Extract text using EasyOCR."""
        results = []
        
        try:
            # Try both original and processed images
            for img_name, img in [("processed", processed), ("original", original)]:
                try:
                    # Convert to RGB
                    if len(img.shape) == 3:
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    
                    # Extract with conservative settings
                    ocr_results = self.easyocr_reader.readtext(
                        rgb_img,
                        detail=1,
                        paragraph=False,
                        width_ths=0.7,
                        height_ths=0.7,
                        text_threshold=0.7,
                        contrast_ths=0.1
                    )
                    
                    for bbox, text, confidence in ocr_results:
                        if text.strip() and len(text.strip()) >= 2 and confidence > 0.3:
                            results.append(LocalOCRResult(
                                text=text,
                                confidence=confidence,
                                engine=f"EasyOCR-{img_name}",
                                bbox=bbox
                            ))
                    
                    # If we got good results, don't try the other image
                    if results and max(r.confidence for r in results) > 0.7:
                        break
                        
                except Exception as e:
                    logger.debug(f"EasyOCR failed on {img_name}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
        
        return results
    
    def _extract_with_tesseract(self, original: np.ndarray, processed: np.ndarray) -> List[LocalOCRResult]:
        """Extract text using Tesseract."""
        results = []
        
        try:
            # Try different PSM modes
            psm_modes = [6, 7, 8, 11, 13]  # Different page segmentation modes
            
            for img_name, img in [("processed", processed), ("original", original)]:
                try:
                    # Convert to PIL
                    if len(img.shape) == 3:
                        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    else:
                        pil_img = PILImage.fromarray(img)
                    
                    best_text = ""
                    best_confidence = 0
                    best_psm = 6
                    
                    for psm in psm_modes:
                        try:
                            config = f'--oem 1 --psm {psm}'
                            text = pytesseract.image_to_string(pil_img, config=config).strip()
                            
                            if text and len(text) >= 2:
                                # Get confidence
                                try:
                                    data = pytesseract.image_to_data(
                                        pil_img, config=config, output_type=pytesseract.Output.DICT
                                    )
                                    confidences = [int(c) for c in data['conf'] if int(c) > 0]
                                    avg_conf = sum(confidences) / len(confidences) if confidences else 0
                                except:
                                    avg_conf = 50  # Default confidence
                                
                                if len(text) > len(best_text) or avg_conf > best_confidence:
                                    best_text = text
                                    best_confidence = avg_conf
                                    best_psm = psm
                        
                        except Exception:
                            continue
                    
                    if best_text:
                        results.append(LocalOCRResult(
                            text=best_text,
                            confidence=best_confidence / 100,
                            engine=f"Tesseract-{img_name}-PSM{best_psm}"
                        ))
                        
                        # If we got a good result, don't try the other image
                        if best_confidence > 70:
                            break
                
                except Exception as e:
                    logger.debug(f"Tesseract failed on {img_name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
        
        return results
    
    def _filter_results(self, results: List[LocalOCRResult]) -> List[LocalOCRResult]:
        """Filter and deduplicate results."""
        if not results:
            return []
        
        # Remove very short or suspicious results
        filtered = []
        for result in results:
            text = result.text.strip()
            
            # Basic validation
            if (len(text) >= 2 and 
                result.confidence > 0.1 and
                not all(c in '.,!?;:()[]{}' for c in text)):  # Not just punctuation
                filtered.append(result)
        
        # Sort by confidence
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        # Remove duplicates (keep highest confidence)
        seen_texts = set()
        unique_results = []
        
        for result in filtered:
            text_lower = result.text.lower().strip()
            if text_lower not in seen_texts:
                seen_texts.add(text_lower)
                unique_results.append(result)
        
        return unique_results
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines."""
        engines = []
        if self.easyocr_reader:
            engines.append("EasyOCR")
        if TESSERACT_AVAILABLE:
            engines.append("Tesseract")
        return engines
    
    def is_available(self) -> bool:
        """Check if any OCR engine is available."""
        return self.easyocr_reader is not None or TESSERACT_AVAILABLE
