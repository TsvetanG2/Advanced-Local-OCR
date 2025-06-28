"""Batch processing module for multiple images."""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .ocr_engines import OCRManager
from .llm_corrector import LLMCorrector
from .text_processors import TextComparator
from ..utils.config import get_config
from ..utils.logging_config import PerformanceLogger

logger = logging.getLogger(__name__)
performance_logger = PerformanceLogger("batch")


class BatchResult:
    """Represents the result of processing a single image in a batch."""
    
    def __init__(self, image_path: str, expected_text: str = ""):
        self.image_path = image_path
        self.expected_text = expected_text
        self.extracted_text = ""
        self.corrected_text = ""
        self.errors = []
        self.similarity = 0.0
        self.processing_time = 0.0
        self.ocr_confidence = 0.0
        self.metadata = {}
        self.success = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'image_path': self.image_path,
            'expected_text': self.expected_text,
            'extracted_text': self.extracted_text,
            'corrected_text': self.corrected_text,
            'errors': self.errors,
            'similarity': self.similarity,
            'processing_time': self.processing_time,
            'ocr_confidence': self.ocr_confidence,
            'metadata': self.metadata,
            'success': self.success
        }


class BatchProcessor:
    """Handles batch processing of multiple images."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize batch processor.
        
        Args:
            config: Configuration. If None, uses global config.
        """
        self.config = config or get_config()
        self.ocr_manager = OCRManager()
        self.llm_corrector = LLMCorrector()
        self.text_comparator = TextComparator()
        self.parallel_processing = self.config.get('performance', {}).get('parallel_processing', False)
        self.max_workers = min(4, os.cpu_count() or 1)  # Limit concurrent workers
    
    def process_folder(self, folder_path: str, expected_texts: Optional[Dict[str, str]] = None,
                      progress_callback: Optional[Callable[[int, int], None]] = None) -> List[BatchResult]:
        """Process all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            expected_texts: Dictionary mapping image filenames to expected texts
            progress_callback: Callback function for progress updates (current, total)
            
        Returns:
            List of batch results
        """
        try:
            # Find all image files
            image_files = self._find_image_files(folder_path)
            
            if not image_files:
                logger.warning(f"No image files found in {folder_path}")
                return []
            
            logger.info(f"Processing {len(image_files)} images from {folder_path}")
            
            # Prepare processing tasks
            tasks = []
            for image_file in image_files:
                filename = os.path.basename(image_file)
                expected_text = expected_texts.get(filename, "") if expected_texts else ""
                tasks.append((image_file, expected_text))
            
            # Process images
            if self.parallel_processing and len(tasks) > 1:
                results = self._process_parallel(tasks, progress_callback)
            else:
                results = self._process_sequential(tasks, progress_callback)
            
            logger.info(f"Batch processing completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return []
    
    def process_file_list(self, file_list: List[str], expected_texts: Optional[Dict[str, str]] = None,
                         progress_callback: Optional[Callable[[int, int], None]] = None) -> List[BatchResult]:
        """Process a specific list of image files.
        
        Args:
            file_list: List of image file paths
            expected_texts: Dictionary mapping image filenames to expected texts
            progress_callback: Callback function for progress updates
            
        Returns:
            List of batch results
        """
        try:
            # Filter existing files
            existing_files = [f for f in file_list if os.path.exists(f)]
            
            if not existing_files:
                logger.warning("No valid image files found in the provided list")
                return []
            
            logger.info(f"Processing {len(existing_files)} images from file list")
            
            # Prepare processing tasks
            tasks = []
            for image_file in existing_files:
                filename = os.path.basename(image_file)
                expected_text = expected_texts.get(filename, "") if expected_texts else ""
                tasks.append((image_file, expected_text))
            
            # Process images
            if self.parallel_processing and len(tasks) > 1:
                results = self._process_parallel(tasks, progress_callback)
            else:
                results = self._process_sequential(tasks, progress_callback)
            
            logger.info(f"File list processing completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error processing file list: {e}")
            return []
    
    def _find_image_files(self, folder_path: str) -> List[str]:
        """Find all image files in a folder."""
        supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        folder = Path(folder_path)
        if not folder.exists():
            return []
        
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def _process_sequential(self, tasks: List[tuple], 
                          progress_callback: Optional[Callable[[int, int], None]] = None) -> List[BatchResult]:
        """Process images sequentially."""
        results = []
        total_tasks = len(tasks)
        
        for i, (image_path, expected_text) in enumerate(tasks):
            try:
                result = self._process_single_image(image_path, expected_text)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total_tasks)
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                # Create failed result
                failed_result = BatchResult(image_path, expected_text)
                failed_result.errors = [f"Processing failed: {str(e)}"]
                results.append(failed_result)
        
        return results
    
    def _process_parallel(self, tasks: List[tuple],
                         progress_callback: Optional[Callable[[int, int], None]] = None) -> List[BatchResult]:
        """Process images in parallel."""
        results = []
        total_tasks = len(tasks)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_single_image, image_path, expected_text): (image_path, expected_text)
                for image_path, expected_text in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                image_path, expected_text = future_to_task[future]
                
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    # Create failed result
                    failed_result = BatchResult(image_path, expected_text)
                    failed_result.errors = [f"Processing failed: {str(e)}"]
                    results.append(failed_result)
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_tasks)
        
        # Sort results by image path to maintain order
        results.sort(key=lambda r: r.image_path)
        return results
    
    def _process_single_image(self, image_path: str, expected_text: str = "") -> BatchResult:
        """Process a single image.
        
        Args:
            image_path: Path to image file
            expected_text: Expected text for comparison
            
        Returns:
            Batch result for the image
        """
        start_time = time.time()
        result = BatchResult(image_path, expected_text)
        
        try:
            # Load image
            import cv2
            import numpy as np
            
            # Handle Unicode paths
            data = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to load image")
            
            # Extract text using OCR
            ocr_results = self.ocr_manager.extract_text(image)
            
            if ocr_results:
                # Combine all extracted text
                extracted_texts = [r.text for r in ocr_results]
                result.extracted_text = ' '.join(extracted_texts)
                
                # Calculate average confidence
                confidences = [r.confidence for r in ocr_results]
                result.ocr_confidence = sum(confidences) / len(confidences)
                
                # Apply LLM correction if available
                if self.llm_corrector.is_enabled():
                    result.corrected_text = self.llm_corrector.correct_text(
                        result.extracted_text, 
                        result.ocr_confidence,
                        expected_text[:200]  # Use expected text as context
                    )
                else:
                    result.corrected_text = result.extracted_text
                
                # Compare with expected text if provided
                if expected_text:
                    comparison = self.text_comparator.compare_texts(
                        expected_text, 
                        result.corrected_text
                    )
                    result.errors = [error['message'] for error in comparison['errors']]
                    result.similarity = comparison['similarity']
                    result.metadata['comparison_metrics'] = comparison['metrics']
                
                result.success = True
                
            else:
                result.errors = ["No text extracted from image"]
            
            # Record processing time
            result.processing_time = time.time() - start_time
            
            # Add metadata
            result.metadata.update({
                'image_size': f"{image.shape[1]}x{image.shape[0]}",
                'ocr_engines': self.ocr_manager.get_available_engines(),
                'llm_enabled': self.llm_corrector.is_enabled()
            })
            
            logger.debug(f"Processed {image_path} in {result.processing_time:.2f}s")
            
        except Exception as e:
            result.errors = [f"Processing error: {str(e)}"]
            result.processing_time = time.time() - start_time
            logger.error(f"Failed to process {image_path}: {e}")
        
        return result
    
    def generate_summary_report(self, results: List[BatchResult]) -> Dict[str, Any]:
        """Generate a summary report from batch results.
        
        Args:
            results: List of batch results
            
        Returns:
            Summary report dictionary
        """
        if not results:
            return {}
        
        total_images = len(results)
        successful_extractions = sum(1 for r in results if r.success)
        total_errors = sum(len(r.errors) for r in results)
        
        # Calculate average metrics
        processing_times = [r.processing_time for r in results if r.processing_time > 0]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        confidences = [r.ocr_confidence for r in results if r.ocr_confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        similarities = [r.similarity for r in results if r.similarity > 0]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Error analysis
        error_types = {}
        for result in results:
            for error in result.errors:
                # Simple error categorization
                if 'missing' in error.lower():
                    error_types['missing_words'] = error_types.get('missing_words', 0) + 1
                elif 'extra' in error.lower():
                    error_types['extra_words'] = error_types.get('extra_words', 0) + 1
                elif 'different' in error.lower():
                    error_types['different_words'] = error_types.get('different_words', 0) + 1
                else:
                    error_types['other'] = error_types.get('other', 0) + 1
        
        return {
            'summary': {
                'total_images': total_images,
                'successful_extractions': successful_extractions,
                'success_rate': successful_extractions / total_images,
                'total_errors': total_errors,
                'average_processing_time': avg_processing_time,
                'average_confidence': avg_confidence,
                'average_similarity': avg_similarity
            },
            'error_analysis': error_types,
            'performance': {
                'fastest_processing': min(processing_times) if processing_times else 0,
                'slowest_processing': max(processing_times) if processing_times else 0,
                'highest_confidence': max(confidences) if confidences else 0,
                'lowest_confidence': min(confidences) if confidences else 0
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'ocr_engines': self.ocr_manager.get_available_engines(),
                'llm_enabled': self.llm_corrector.is_enabled(),
                'parallel_processing': self.parallel_processing
            }
        }
