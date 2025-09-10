import numpy as np
import cv2
import time
from typing import Dict, Tuple, Optional, List
import threading
from dataclasses import dataclass

@dataclass
class CameraConfig:
    """Configuration for camera setup"""
    id: int
    name: str
    position: Tuple[float, float, float]  # x, y, z
    orientation: Tuple[float, float, float]  # roll, pitch, yaw
    fov: float = 60.0
    resolution: Tuple[int, int] = (640, 480)

class ActiveCameraController:
    """
    Active vision camera controller for SO-ARM101
    Supports both real cameras and simulated viewpoints
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.cameras = {}
        self.active_camera_id = 0
        self.camera_positions = {}
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.tracking_enabled = False
        
        # Camera movement parameters
        self.pan_angle = 0.0  # Current pan angle
        self.tilt_angle = 0.0  # Current tilt angle
        self.pan_limits = (-90, 90)  # Pan limits in degrees
        self.tilt_limits = (-45, 45)  # Tilt limits in degrees
        
        # Initialize cameras
        self._setup_cameras()
        
    def _setup_cameras(self):
        """Initialize camera system"""
        # For simulation, we'll use OpenCV to simulate different viewpoints
        # In real setup, this would initialize multiple USB cameras
        
        default_cameras = [
            CameraConfig(0, "overhead", (0, 0, 0.5), (0, 90, 0)),
            CameraConfig(1, "side", (0.3, 0, 0.2), (0, 0, 0)),
            CameraConfig(2, "wrist", (0, 0, 0.1), (0, 0, 0)),
        ]
        
        for cam_config in default_cameras:
            self.cameras[cam_config.id] = {
                'config': cam_config,
                'capture': None,  # Will be initialized when needed
                'active': False
            }
            
    def initialize_camera(self, camera_id: int) -> bool:
        """Initialize specific camera"""
        try:
            # For simulation, create a virtual camera
            # In real setup: cap = cv2.VideoCapture(camera_id)
            self.cameras[camera_id]['active'] = True
            print(f"✓ Camera {camera_id} initialized")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize camera {camera_id}: {e}")
            return False
    
    def get_camera_view(self, camera_id: int) -> Optional[np.ndarray]:
        """Get current view from specified camera"""
        if camera_id not in self.cameras:
            return None
            
        # For simulation, generate a synthetic view
        # In real setup, this would capture from actual camera
        img = self._generate_synthetic_view(camera_id)
        return img
    
    def _generate_synthetic_view(self, camera_id: int) -> np.ndarray:
        """Generate synthetic camera view for simulation"""
        config = self.cameras[camera_id]['config']
        
        # Create a simple synthetic scene
        img = np.zeros((config.resolution[1], config.resolution[0], 3), dtype=np.uint8)
        
        # Add some visual elements to simulate workspace
        cv2.rectangle(img, (50, 50), (590, 430), (100, 100, 100), 2)  # Workspace boundary
        cv2.circle(img, (320, 240), 30, (0, 255, 0), -1)  # Target object
        
        # Add camera info text
        cv2.putText(img, f"Camera {camera_id}: {config.name}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Pan: {self.pan_angle:.1f}° Tilt: {self.tilt_angle:.1f}°", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def set_active_camera(self, camera_id: int):
        """Switch to specified camera"""
        if camera_id in self.cameras:
            self.active_camera_id = camera_id
            print(f"Switched to camera {camera_id}")
    
    def move_camera(self, delta_pan: float, delta_tilt: float):
        """Move active camera by specified angles"""
        # Update pan angle
        new_pan = self.pan_angle + delta_pan
        self.pan_angle = np.clip(new_pan, self.pan_limits[0], self.pan_limits[1])
        
        # Update tilt angle
        new_tilt = self.tilt_angle + delta_tilt
        self.tilt_angle = np.clip(new_tilt, self.tilt_limits[0], self.tilt_limits[1])
        
        print(f"Camera moved to pan: {self.pan_angle:.1f}°, tilt: {self.tilt_angle:.1f}°")
    
    def track_target(self, target_pos: np.ndarray):
        """Point camera towards target position"""
        self.target_position = target_pos
        
        # Simple tracking: calculate angles needed to center target
        # This is a simplified version - real implementation would use camera matrix
        cam_config = self.cameras[self.active_camera_id]['config']
        cam_pos = np.array(cam_config.position)
        
        # Calculate direction vector
        direction = target_pos - cam_pos
        
        # Convert to pan/tilt angles (simplified)
        distance = np.linalg.norm(direction[:2])
        if distance > 0:
            pan_target = np.degrees(np.arctan2(direction[1], direction[0]))
            tilt_target = np.degrees(np.arctan2(direction[2], distance))
            
            # Smooth movement towards target
            pan_diff = pan_target - self.pan_angle
            tilt_diff = tilt_target - self.tilt_angle
            
            # Apply movement with smoothing
            self.move_camera(pan_diff * 0.1, tilt_diff * 0.1)
    
    def get_best_viewpoint(self, target_objects: List[np.ndarray]) -> int:
        """Select best camera viewpoint for given target objects"""
        best_camera = self.active_camera_id
        best_score = 0
        
        for camera_id in self.cameras.keys():
            score = self._evaluate_viewpoint(camera_id, target_objects)
            if score > best_score:
                best_score = score
                best_camera = camera_id
        
        return best_camera
    
    def _evaluate_viewpoint(self, camera_id: int, target_objects: List[np.ndarray]):
        """Evaluate viewpoint quality for target objects"""
        # Simple scoring based on camera position and target visibility
        # In real implementation, this would consider occlusion, distance, angle, etc.
        config = self.cameras[camera_id]['config']
        
        if not target_objects:
            return 0.0
        
        # Calculate average distance to targets
        cam_pos = np.array(config.position)
        total_score = 0
        
        for target_pos in target_objects:
            distance = np.linalg.norm(target_pos - cam_pos)
            # Prefer moderate distances (not too close, not too far)
            distance_score = 1.0 / (1.0 + (distance - 0.3)**2)
            total_score += distance_score
        
        return total_score / len(target_objects)