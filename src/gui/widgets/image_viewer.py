"""Custom image viewer widget."""

import cv2
import numpy as np
from typing import Optional
import logging

from PyQt5.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QRect

logger = logging.getLogger(__name__)


class ImageViewer(QScrollArea):
    """Custom image viewer with zoom and scroll capabilities."""
    
    # Signals
    image_clicked = pyqtSignal(int, int)  # x, y coordinates
    
    def __init__(self, parent=None):
        """Initialize image viewer."""
        super().__init__(parent)
        
        # Setup scroll area
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #777; background-color: #f0f0f0;")
        self.image_label.setText("Drop image here or click 'Load Image'")
        self.image_label.setMinimumSize(280, 280)
        self.image_label.mousePressEvent = self._on_image_click
        
        # Set the label as the scroll area widget
        self.setWidget(self.image_label)
        
        # State variables
        self.original_image = None
        self.current_pixmap = None
        self.scale_factor = 1.0
        self.max_display_size = 400
        
        logger.debug("ImageViewer initialized")
    
    def load_image(self, image_path: str) -> None:
        """Load and display an image.
        
        Args:
            image_path: Path to image file
        """
        try:
            # Load image using OpenCV for better Unicode support
            data = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to load image")
            
            # Store original image
            self.original_image = image.copy()
            
            # Convert and display
            self._display_image(image)
            
            # Update styling
            self.image_label.setStyleSheet("border: 1px solid #ccc;")
            
            logger.info(f"Image loaded: {image.shape[1]}x{image.shape[0]}")
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            self.clear()
            raise
    
    def load_image_array(self, image: np.ndarray) -> None:
        """Load and display an image from numpy array.
        
        Args:
            image: Image as numpy array
        """
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid image array")
            
            # Store original image
            self.original_image = image.copy()
            
            # Convert and display
            self._display_image(image)
            
            # Update styling
            self.image_label.setStyleSheet("border: 1px solid #ccc;")
            
            logger.debug(f"Image array loaded: {image.shape[1]}x{image.shape[0]}")
            
        except Exception as e:
            logger.error(f"Failed to load image array: {e}")
            self.clear()
            raise
    
    def _display_image(self, image: np.ndarray) -> None:
        """Convert and display OpenCV image.
        
        Args:
            image: OpenCV image (BGR format)
        """
        try:
            # Resize for display if needed
            display_image = self._resize_for_display(image)
            
            # Convert BGR to RGB
            if len(display_image.shape) == 3:
                rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = display_image
            
            # Convert to QImage
            h, w = rgb_image.shape[:2]
            
            if len(rgb_image.shape) == 3:
                bytes_per_line = 3 * w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                bytes_per_line = w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            
            # Convert to QPixmap and display
            pixmap = QPixmap.fromImage(q_image)
            self.current_pixmap = pixmap
            self.image_label.setPixmap(pixmap)
            
            # Adjust label size
            self.image_label.resize(pixmap.size())
            
        except Exception as e:
            logger.error(f"Failed to display image: {e}")
            raise
    
    def _resize_for_display(self, image: np.ndarray) -> np.ndarray:
        """Resize image for display purposes.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        # Calculate scale factor
        scale = min(self.max_display_size / w, self.max_display_size / h)
        
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.scale_factor = scale
            return resized
        else:
            self.scale_factor = 1.0
            return image
    
    def clear(self) -> None:
        """Clear the image viewer."""
        self.original_image = None
        self.current_pixmap = None
        self.scale_factor = 1.0
        
        self.image_label.clear()
        self.image_label.setText("Drop image here or click 'Load Image'")
        self.image_label.setStyleSheet("border: 2px dashed #777; background-color: #f0f0f0;")
        self.image_label.setMinimumSize(280, 280)
        
        logger.debug("Image viewer cleared")
    
    def zoom_in(self) -> None:
        """Zoom in on the image."""
        if self.current_pixmap:
            self.scale_factor *= 1.25
            self._update_display_scale()
    
    def zoom_out(self) -> None:
        """Zoom out on the image."""
        if self.current_pixmap:
            self.scale_factor /= 1.25
            self._update_display_scale()
    
    def reset_zoom(self) -> None:
        """Reset zoom to fit the display area."""
        if self.original_image is not None:
            self._display_image(self.original_image)
    
    def _update_display_scale(self) -> None:
        """Update display with current scale factor."""
        if self.current_pixmap:
            scaled_pixmap = self.current_pixmap.scaled(
                self.current_pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())
    
    def get_original_image(self) -> Optional[np.ndarray]:
        """Get the original loaded image.
        
        Returns:
            Original image as numpy array or None
        """
        return self.original_image.copy() if self.original_image is not None else None
    
    def get_display_scale(self) -> float:
        """Get current display scale factor.
        
        Returns:
            Current scale factor
        """
        return self.scale_factor
    
    def _on_image_click(self, event) -> None:
        """Handle mouse click on image.
        
        Args:
            event: Mouse event
        """
        if self.current_pixmap and event.button() == Qt.LeftButton:
            # Convert click coordinates to original image coordinates
            label_pos = event.pos()
            pixmap_rect = self.image_label.pixmap().rect()
            
            # Calculate relative position
            if pixmap_rect.contains(label_pos):
                rel_x = label_pos.x() / pixmap_rect.width()
                rel_y = label_pos.y() / pixmap_rect.height()
                
                # Convert to original image coordinates
                if self.original_image is not None:
                    orig_h, orig_w = self.original_image.shape[:2]
                    orig_x = int(rel_x * orig_w)
                    orig_y = int(rel_y * orig_h)
                    
                    self.image_clicked.emit(orig_x, orig_y)
                    logger.debug(f"Image clicked at: ({orig_x}, {orig_y})")
    
    def highlight_region(self, bbox: list, color: tuple = (255, 0, 0), thickness: int = 2) -> None:
        """Highlight a region on the image.
        
        Args:
            bbox: Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            color: Highlight color (R, G, B)
            thickness: Line thickness
        """
        if self.original_image is None or not bbox:
            return
        
        try:
            # Create a copy of the original image
            highlighted = self.original_image.copy()
            
            # Convert bbox to integer coordinates
            points = np.array([[int(pt[0]), int(pt[1])] for pt in bbox], np.int32)
            
            # Draw the highlight
            cv2.polylines(highlighted, [points], True, color[::-1], thickness)  # BGR format
            
            # Display the highlighted image
            self._display_image(highlighted)
            
            logger.debug(f"Region highlighted with {len(bbox)} points")
            
        except Exception as e:
            logger.error(f"Failed to highlight region: {e}")
    
    def remove_highlights(self) -> None:
        """Remove all highlights and show original image."""
        if self.original_image is not None:
            self._display_image(self.original_image)
            logger.debug("Highlights removed")
    
    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for zooming."""
        if self.current_pixmap:
            # Get wheel delta
            delta = event.angleDelta().y()
            
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            
            event.accept()
        else:
            super().wheelEvent(event)
