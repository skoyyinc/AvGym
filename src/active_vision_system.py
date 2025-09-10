import cv2
import numpy as np
import time
import threading
from typing import Dict, List, Optional
from src.camera_controller import ActiveCameraController, CameraConfig
from src.object_detector import SimpleObjectDetector, DetectedObject

class ActiveVisionSystem:
    """
    Main active vision system for SO-ARM101
    Integrates camera control with object detection and tracking
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.camera_controller = ActiveCameraController(config)
        self.object_detector = SimpleObjectDetector()
        
        # State variables
        self.current_targets = []
        self.tracking_active = False
        self.occlusion_detected = False
        
        # Performance metrics
        self.detection_count = 0
        self.successful_tracks = 0
        self.camera_switches = 0
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the active vision system"""
        print("🔄 Initializing Active Vision System...")
        
        # Initialize cameras
        for camera_id in self.camera_controller.cameras.keys():
            self.camera_controller.initialize_camera(camera_id)
        
        print("✓ Active Vision System ready!")
    
    def start_active_vision(self):
        """Start the active vision loop"""
        print("🚀 Starting active vision loop...")
        self.tracking_active = True
        
        try:
            while self.tracking_active:
                self._active_vision_step()
                time.sleep(0.1)  # 10 FPS update rate
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping active vision...")
            self.tracking_active = False
    
    def _active_vision_step(self):
        """Single step of active vision processing"""
        # Get current camera view
        current_view = self.camera_controller.get_camera_view(
            self.camera_controller.active_camera_id
        )
        
        if current_view is None:
            return
        
        # Detect objects in current view
        detections = self.object_detector.detect_objects(current_view)
        self.detection_count += len(detections)
        
        # Update target tracking
        self._update_target_tracking(detections)
        
        # Check for occlusions and handle them
        self._handle_occlusions(detections)
        
        # Visualize results
        self._visualize_results(current_view, detections)
    
    def _update_target_tracking(self, detections: List[DetectedObject]):
        """Update target tracking based on detections"""
        if not detections:
            return
        
        # For each detection, update tracking
        for detection in detections:
            # Estimate 3D position
            position_3d = self.object_detector.get_target_position(
                detection, self.camera_controller.active_camera_id
            )
            
            # Update camera to track this target
            self.camera_controller.track_target(position_3d)
            
            # Store target information
            self.current_targets.append({
                'detection': detection,
                'position': position_3d,
                'timestamp': time.time()
            })
        
        # Keep only recent targets
        current_time = time.time()
        self.current_targets = [
            target for target in self.current_targets
            if current_time - target['timestamp'] < 1.0
        ]
    
    def _handle_occlusions(self, detections: List[DetectedObject]):
        """Handle occlusions by switching camera viewpoints"""
        # Simple occlusion detection: if we lost track of objects
        if len(self.current_targets) > 0 and len(detections) == 0:
            if not self.occlusion_detected:
                print("⚠️  Occlusion detected! Switching camera viewpoint...")
                self.occlusion_detected = True
                self._switch_to_best_viewpoint()
        else:
            self.occlusion_detected = False
    
    def _switch_to_best_viewpoint(self):
        """Switch to the best camera viewpoint for current targets"""
        if not self.current_targets:
            return
        
        # Get target positions
        target_positions = [target['position'] for target in self.current_targets]
        
        # Find best viewpoint
        best_camera = self.camera_controller.get_best_viewpoint(target_positions)
        
        if best_camera != self.camera_controller.active_camera_id:
            self.camera_controller.set_active_camera(best_camera)
            self.camera_switches += 1
            print(f"📷 Switched to camera {best_camera}")
    
    def _visualize_results(self, image: np.ndarray, detections: List[DetectedObject]):
        """Visualize active vision results with fallback options"""
        try:
            # Draw detections
            result_img = self.object_detector.draw_detections(image, detections)
            
            # Add system info
            info_text = [
                f"Active Camera: {self.camera_controller.active_camera_id}",
                f"Detections: {len(detections)}",
                f"Targets: {len(self.current_targets)}",
                f"Camera Switches: {self.camera_switches}",
                f"Occlusion: {'Yes' if self.occlusion_detected else 'No'}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(result_img, text, (10, 100 + i * 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Try to display with fallback
            self._safe_imshow('Active Vision - SO-ARM101', result_img)
            
        except Exception as e:
            # Fallback: just print status instead of showing image
            print(f"📊 Camera: {self.camera_controller.active_camera_id} | "
                f"Objects: {len(detections)} | "
                f"Targets: {len(self.current_targets)} | "
                f"Switches: {self.camera_switches}")

    def _safe_imshow(self, window_name: str, image: np.ndarray):
        """Safely display image with fallback options"""
        try:
            # First attempt: normal imshow
            cv2.imshow(window_name, image)
            cv2.waitKey(1)
        except Exception as e:
            try:
                # Second attempt: create window first
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, image)
                cv2.waitKey(1)
            except Exception as e2:
                # Third attempt: save image instead of displaying
                timestamp = int(time.time() * 1000)
                filename = f"debug_frame_{timestamp}.jpg"
                cv2.imwrite(filename, image)
                if hasattr(self, '_last_save_print') and time.time() - self._last_save_print < 5:
                    return  # Don't spam prints
                print(f"💾 Saved frame to {filename} (display unavailable)")
                self._last_save_print = time.time()
        
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        return {
            'detection_count': self.detection_count,
            'successful_tracks': self.successful_tracks,
            'camera_switches': self.camera_switches,
            'occlusion_events': int(self.occlusion_detected)
        }
    
    def stop(self):
        """Stop the active vision system"""
        self.tracking_active = False
        cv2.destroyAllWindows()
        print("🛑 Active Vision System stopped")