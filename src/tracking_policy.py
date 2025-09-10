import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import math

class ImprovedTrackingPolicy:
    """
    Improved visual servoing policy for SO-ARM101 object tracking
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Control parameters
        self.p_gain_fast = config.get('p_gain_fast', 0.8)  # For large errors
        self.p_gain_slow = config.get('p_gain_slow', 0.3)  # For small errors
        self.max_velocity = config.get('max_velocity', 1.0)
        
        # Visual servoing parameters
        self.error_threshold_fast = 0.3  # Switch to slow gain below this
        self.deadzone_radius = 0.05      # Stop moving if error is very small
        
        # Tracking state
        self.last_target_position = None
        self.lost_target_steps = 0
        self.search_direction = 1
        self.search_joint = 0
        
        # Performance tracking
        self.tracking_errors = []
        self.control_actions = []
        
        print(f"🎯 Improved tracking policy initialized:")
        print(f"   Fast gain: {self.p_gain_fast}, Slow gain: {self.p_gain_slow}")
        print(f"   Max velocity: {self.max_velocity}")
        
    def predict(self, observation: Dict) -> np.ndarray:
        """Predict action based on observation"""
        camera_image = observation['camera_image']
        joint_positions = observation['joint_positions']
        target_in_view = observation['target_in_view'][0]
        center_distance = observation['target_center_distance'][0]
        
        # Enhanced target detection
        target_pixel_pos = self._detect_target_position(camera_image)
        
        if target_pixel_pos is not None:
            # Target visible - use improved visual servoing
            action = self._improved_visual_servoing(
                target_pixel_pos, camera_image.shape, joint_positions, center_distance
            )
            self.lost_target_steps = 0
            self.last_target_position = target_pixel_pos
            
        else:
            # Target lost - intelligent search
            action = self._intelligent_search(joint_positions)
            self.lost_target_steps += 1
            
        # Smooth and clip action
        action = self._smooth_action(action)
        action = np.clip(action, -1.0, 1.0)
        
        # Track performance
        self.tracking_errors.append(center_distance)
        self.control_actions.append(action.copy())
        
        return action
        
    def _detect_target_position(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """Enhanced target detection with better filtering"""
        if image is None or image.size == 0:
            return None
            
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Enhanced red detection with multiple ranges
            ranges = [
                (np.array([0, 120, 120]), np.array([10, 255, 255])),
                (np.array([160, 120, 120]), np.array([180, 255, 255]))
            ]
            
            masks = []
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                masks.append(mask)
            
            # Combine masks
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Filter contours by area and circularity
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 200:  # Minimum area
                        # Check circularity
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.3:  # Reasonably circular
                                valid_contours.append((contour, area))
                
                if valid_contours:
                    # Get largest valid contour
                    largest_contour = max(valid_contours, key=lambda x: x[1])[0]
                    
                    # Get center using moments
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        return (cx, cy)
                        
        except Exception as e:
            print(f"⚠️  Target detection error: {e}")
            
        return None
        
    def _improved_visual_servoing(self, target_pos: Tuple[int, int], 
                                image_shape: Tuple, joint_positions: np.ndarray,
                                center_distance: float) -> np.ndarray:
        """Improved visual servoing with adaptive gains"""
        height, width = image_shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Calculate pixel errors
        error_x = target_pos[0] - center_x
        error_y = target_pos[1] - center_y
        
        # Normalize errors to [-1, 1]
        norm_error_x = error_x / (width / 2)
        norm_error_y = error_y / (height / 2)
        
        # Apply deadzone
        if abs(norm_error_x) < self.deadzone_radius:
            norm_error_x = 0
        if abs(norm_error_y) < self.deadzone_radius:
            norm_error_y = 0
            
        # Adaptive gain based on error magnitude
        if center_distance > self.error_threshold_fast:
            gain = self.p_gain_fast
        else:
            gain = self.p_gain_slow
            
        # Initialize action
        action = np.zeros(len(joint_positions))
        
        # SO-ARM101 specific joint mapping
        if len(action) >= 6:
            # Joint 0: Base rotation (yaw) - for horizontal error (inverted due to camera orientation)
            action[0] = gain * norm_error_x * 0.8
            
            # Joint 1: Shoulder (pitch) - for vertical error  
            action[1] = gain * norm_error_y * 0.6
            
            # Joint 2: Elbow - assist with vertical movement
            action[2] = gain * norm_error_y * 0.4
            
            # Joint 3: Wrist pitch - fine vertical adjustment
            action[3] = gain * norm_error_y * 0.3
            
            # Joint 4: Wrist roll - fine horizontal adjustment (inverted due to camera orientation)
            action[4] = gain * norm_error_x * 0.2
            
            # Joint 5: Wrist yaw - minimal movement for stability
            action[5] = 0.0
            
        # Scale by max velocity
        action = action * self.max_velocity
        
        return action
        
    def _intelligent_search(self, joint_positions: np.ndarray) -> np.ndarray:
        """Intelligent search pattern when target is lost"""
        action = np.zeros(len(joint_positions))
        
        # Multi-phase search strategy
        search_phase = self.lost_target_steps // 30  # Change phase every 3 seconds
        
        if search_phase == 0:
            # Phase 1: Rotate base slowly
            action[0] = 0.4 * self.search_direction
            
        elif search_phase == 1:
            # Phase 2: Move shoulder up/down
            action[1] = 0.3 * self.search_direction
            
        elif search_phase == 2:
            # Phase 3: Combined base + shoulder movement
            action[0] = 0.3 * self.search_direction
            action[1] = 0.2 * (-self.search_direction)
            
        else:
            # Phase 4: Return to center and reverse direction
            # Move joints toward neutral positions
            neutral_positions = [0, -0.3, 0.5, 0, 0.2, 0]
            for i in range(min(len(action), len(neutral_positions))):
                if i < len(joint_positions):
                    error = neutral_positions[i] - joint_positions[i]
                    action[i] = 0.3 * np.sign(error) * min(abs(error), 1.0)
            
            # Reset search after returning to center
            if self.lost_target_steps > 150:  # 15 seconds
                self.lost_target_steps = 0
                self.search_direction *= -1  # Reverse search direction
                
        return action
        
    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        """Apply smoothing to reduce jittery movements"""
        # Simple exponential smoothing
        if hasattr(self, '_last_action'):
            alpha = 0.7  # Smoothing factor
            action = alpha * action + (1 - alpha) * self._last_action
        
        self._last_action = action.copy()
        return action
        
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.tracking_errors:
            return {'no_data': True}
            
        return {
            'average_tracking_error': np.mean(self.tracking_errors),
            'min_tracking_error': np.min(self.tracking_errors),
            'std_tracking_error': np.std(self.tracking_errors),
            'tracking_steps': len(self.tracking_errors),
            'lost_target_episodes': self.lost_target_steps,
            'recent_performance': np.mean(self.tracking_errors[-50:]) if len(self.tracking_errors) >= 50 else np.mean(self.tracking_errors)
        }