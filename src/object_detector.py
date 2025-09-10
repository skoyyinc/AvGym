import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DetectedObject:
    """Detected object information"""
    id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    center: Tuple[float, float]
    position_3d: Optional[np.ndarray] = None

class SimpleObjectDetector:
    """
    Simple object detector for active vision
    Uses color-based detection for rapid prototyping
    """
    
    def __init__(self):
        # Define color ranges for different objects (HSV)
        self.color_ranges = {
            'red_object': {
                'lower': np.array([0, 50, 50]),
                'upper': np.array([10, 255, 255])
            },
            'green_object': {
                'lower': np.array([50, 50, 50]),
                'upper': np.array([70, 255, 255])
            },
            'blue_object': {
                'lower': np.array([100, 50, 50]),
                'upper': np.array([120, 255, 255])
            }
        }
        
        self.min_contour_area = 500
        self.detected_objects = []
        
    def detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """Detect objects in image using color-based detection"""
        if image is None:
            return []
            
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detected_objects = []
        
        for obj_id, (class_name, color_range) in enumerate(self.color_ranges.items()):
            # Create mask for this color
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_contour_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate center
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Create detected object
                    detected_obj = DetectedObject(
                        id=obj_id,
                        class_name=class_name,
                        bbox=(x, y, w, h),
                        confidence=min(area / 5000.0, 1.0),  # Simple confidence based on size
                        center=(center_x, center_y)
                    )
                    
                    detected_objects.append(detected_obj)
        
        self.detected_objects = detected_objects
        return detected_objects
    
    def draw_detections(self, image: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        """Draw detection results on image"""
        result_img = image.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(result_img, (int(detection.center[0]), int(detection.center[1])), 5, (0, 0, 255), -1)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(result_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_img
    
    def get_target_position(self, detection: DetectedObject, camera_id: int) -> np.ndarray:
        """Estimate 3D position of detected object"""
        # Simple depth estimation based on object size
        # In real implementation, this would use stereo vision or depth camera
        
        _, _, w, h = detection.bbox
        object_size = max(w, h)
        
        # Estimate depth based on object size (very rough approximation)
        estimated_depth = 1000.0 / object_size if object_size > 0 else 0.5
        
        # Convert pixel coordinates to 3D (simplified)
        # This assumes known camera calibration parameters
        center_x, center_y = detection.center
        
        # Simple projection (would need proper camera matrix in real setup)
        world_x = (center_x - 320) * estimated_depth / 320
        world_y = (center_y - 240) * estimated_depth / 240
        world_z = estimated_depth
        
        return np.array([world_x, world_y, world_z])