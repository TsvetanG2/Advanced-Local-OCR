"""Image preprocessing module."""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

from ..utils.config import get_config

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image preprocessing for OCR optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize image processor.
        
        Args:
            config: Preprocessing configuration. If None, uses global config.
        """
        self.config = config or get_config().get_preprocessing_config()
        logger.debug("ImageProcessor initialized with config")
    
    def preprocess_image(self, image: np.ndarray, debug_mode: bool = False) -> np.ndarray:
        """Apply full preprocessing pipeline to image.

        Args:
            image: Input image as numpy array
            debug_mode: If True, saves intermediate steps for debugging

        Returns:
            Preprocessed image
        """
        try:
            logger.debug(f"Starting preprocessing for image shape: {image.shape}")

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            logger.debug(f"Grayscale image stats: mean={np.mean(gray):.1f}, std={np.std(gray):.1f}")

            # Save original for debugging
            if debug_mode:
                cv2.imwrite('debug_01_original.png', gray)

            # Enhance contrast first
            enhanced = self._enhance_contrast_adaptive(gray)
            if debug_mode:
                cv2.imwrite('debug_02_enhanced.png', enhanced)

            # Apply noise reduction
            denoised = self._apply_noise_reduction(enhanced)
            if debug_mode:
                cv2.imwrite('debug_03_denoised.png', denoised)

            # Apply Gaussian blur for smoothing
            if self.config.get('gaussian_blur', {}).get('kernel_size'):
                blurred = self._apply_gaussian_blur(denoised)
            else:
                blurred = denoised

            if debug_mode:
                cv2.imwrite('debug_04_blurred.png', blurred)

            # Apply adaptive thresholding with multiple methods
            binary = self._apply_adaptive_threshold_improved(blurred)
            if debug_mode:
                cv2.imwrite('debug_05_binary.png', binary)

            # Apply morphological operations to clean up
            if self.config.get('morphology', {}).get('operation'):
                cleaned = self._apply_morphology_improved(binary)
            else:
                cleaned = binary

            if debug_mode:
                cv2.imwrite('debug_06_cleaned.png', cleaned)

            # Deskew if enabled
            if self.config.get('deskew', {}).get('enabled', True):
                deskewed = self._deskew_image(cleaned)
            else:
                deskewed = cleaned

            if debug_mode:
                cv2.imwrite('debug_07_deskewed.png', deskewed)

            # Final inversion check (white text on black background)
            if np.mean(deskewed) < 127:
                final = cv2.bitwise_not(deskewed)
                logger.debug("Image inverted (white text on black background detected)")
            else:
                final = deskewed

            if debug_mode:
                cv2.imwrite('debug_08_final.png', final)

            logger.debug("Image preprocessing completed successfully")
            return final

        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            # Return enhanced grayscale as fallback
            try:
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                return self._enhance_contrast_adaptive(gray)
            except:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to reduce noise."""
        blur_config = self.config.get('gaussian_blur', {})
        kernel_size = blur_config.get('kernel_size', [5, 5])
        sigma = blur_config.get('sigma', 0)
        
        return cv2.GaussianBlur(image, tuple(kernel_size), sigma)
    
    def _apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for binarization."""
        threshold_config = self.config.get('adaptive_threshold', {})
        
        max_value = threshold_config.get('max_value', 255)
        adaptive_method = getattr(cv2, threshold_config.get('adaptive_method', 'ADAPTIVE_THRESH_GAUSSIAN_C'))
        threshold_type = getattr(cv2, threshold_config.get('threshold_type', 'THRESH_BINARY'))
        block_size = threshold_config.get('block_size', 11)
        c = threshold_config.get('c', 2)
        
        return cv2.adaptiveThreshold(
            image, max_value, adaptive_method, threshold_type, block_size, c
        )
    
    def _apply_morphology(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to improve text quality."""
        morph_config = self.config.get('morphology', {})
        
        kernel_size = morph_config.get('kernel_size', [1, 1])
        operation = morph_config.get('operation', 'MORPH_CLOSE')
        
        kernel = np.ones(tuple(kernel_size), np.uint8)
        morph_op = getattr(cv2, operation)
        
        return cv2.morphologyEx(image, morph_op, kernel)
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew/rotation."""
        try:
            # Find coordinates of all non-zero pixels
            coords = np.column_stack(np.where(image > 0))
            
            if len(coords) < 10:  # Not enough points for reliable deskewing
                return image
            
            # Find minimum area rectangle
            angle = cv2.minAreaRect(coords)[-1]
            
            # Correct angle calculation
            angle_threshold = self.config.get('deskew', {}).get('angle_threshold', 45)
            if angle < -angle_threshold:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Only apply correction if angle is significant
            if abs(angle) < 0.5:
                return image
            
            # Apply rotation
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            rotated = cv2.warpAffine(
                image, rotation_matrix, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            logger.debug(f"Image deskewed by {angle:.2f} degrees")
            return rotated
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}, returning original image")
            return image
    
    def resize_for_display(self, image: np.ndarray, max_size: Optional[int] = None) -> np.ndarray:
        """Resize image for display purposes.
        
        Args:
            image: Input image
            max_size: Maximum size for display. If None, uses config.
            
        Returns:
            Resized image
        """
        if max_size is None:
            max_size = self.config.get('resize', {}).get('max_display_size', 400)
        
        h, w = image.shape[:2]
        scale = min(max_size / w, max_size / h)
        
        if scale < 1:
            interpolation = getattr(cv2, self.config.get('resize', {}).get('interpolation', 'INTER_CUBIC'))
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using histogram equalization."""
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return cv2.equalizeHist(image)
    
    def crop_region(self, image: np.ndarray, bbox: list, margin: int = 5) -> np.ndarray:
        """Crop a specific region from the image with margin.
        
        Args:
            image: Input image
            bbox: Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            margin: Margin to add around the region
            
        Returns:
            Cropped image region
        """
        try:
            # Extract coordinates
            xs = [int(pt[0]) for pt in bbox]
            ys = [int(pt[1]) for pt in bbox]
            
            # Calculate bounding rectangle with margin
            x1 = max(min(xs) - margin, 0)
            x2 = min(max(xs) + margin, image.shape[1])
            y1 = max(min(ys) - margin, 0)
            y2 = min(max(ys) + margin, image.shape[0])
            
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid crop region")
            
            return image[y1:y2, x1:x2]
            
        except Exception as e:
            logger.error(f"Error cropping region: {e}")
            return image
    
    def _enhance_contrast_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive contrast enhancement."""
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)

            # Blend with original to avoid over-enhancement
            alpha = 0.7
            result = cv2.addWeighted(enhanced, alpha, image, 1 - alpha, 0)

            logger.debug("Applied adaptive contrast enhancement")
            return result

        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image

    def _apply_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction while preserving text."""
        try:
            # Use bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(image, 9, 75, 75)

            logger.debug("Applied noise reduction")
            return denoised

        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return image

    def _apply_adaptive_threshold_improved(self, image: np.ndarray) -> np.ndarray:
        """Apply improved adaptive thresholding with multiple methods."""
        try:
            # Try multiple thresholding methods and choose the best
            methods = [
                (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),
                (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 3),
                (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2),
                (cv2.ADAPTIVE_THRESH_MEAN_C, 15, 3)
            ]

            results = []
            for method, block_size, c in methods:
                try:
                    binary = cv2.adaptiveThreshold(
                        image, 255, method, cv2.THRESH_BINARY, block_size, c
                    )
                    # Calculate quality metric (variance of binary image)
                    quality = np.var(binary)
                    results.append((binary, quality))
                except:
                    continue

            if results:
                # Choose the result with highest variance (most detail preserved)
                best_binary = max(results, key=lambda x: x[1])[0]
                logger.debug("Applied improved adaptive thresholding")
                return best_binary
            else:
                # Fallback to simple thresholding
                _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                logger.debug("Applied fallback Otsu thresholding")
                return binary

        except Exception as e:
            logger.warning(f"Adaptive thresholding failed: {e}")
            # Simple fallback
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return binary

    def _apply_morphology_improved(self, image: np.ndarray) -> np.ndarray:
        """Apply improved morphological operations."""
        try:
            # Remove small noise with opening
            kernel_small = np.ones((2, 2), np.uint8)
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_small)

            # Fill small gaps with closing
            kernel_medium = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)

            # Connect broken characters
            kernel_connect = np.ones((1, 3), np.uint8)  # Horizontal connection
            connected = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_connect)

            logger.debug("Applied improved morphological operations")
            return connected

        except Exception as e:
            logger.warning(f"Morphological operations failed: {e}")
            return image

    def get_image_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate image quality metrics.

        Args:
            image: Input image

        Returns:
            Dictionary with quality metrics
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Calculate metrics
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()  # Sharpness
            mean_brightness = np.mean(gray)
            contrast = np.std(gray)

            # Additional quality metrics
            gradient_magnitude = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)))

            return {
                'sharpness': float(variance),
                'brightness': float(mean_brightness),
                'contrast': float(contrast),
                'gradient_magnitude': float(gradient_magnitude),
                'resolution': image.shape[1] * image.shape[0],
                'aspect_ratio': image.shape[1] / image.shape[0]
            }

        except Exception as e:
            logger.error(f"Error calculating image quality metrics: {e}")
            return {}
