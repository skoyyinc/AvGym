"""
Hybrid Tracking Policy - Phase 3 Implementation

Combines visual servoing (tracking) with RL (search)
Automatically switches between modes based on target visibility
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
import time

try:
    from stable_baselines3 import PPO, SAC
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("⚠️  stable-baselines3 not available for RL functionality")

from .tracking_policy import ImprovedTrackingPolicy


class HybridTrackingPolicy:
    """
    Combines visual servoing (tracking) with RL (search)
    Automatically switches between modes based on target visibility
    """
    
    def __init__(self, 
                 visual_config: Dict,
                 rl_model_path: str,
                 rl_algorithm: str = "ppo",
                 target_lost_threshold: int = 5,
                 target_found_threshold: int = 2):
        """
        Initialize hybrid tracking policy
        
        Args:
            visual_config: Configuration for visual servoing policy
            rl_model_path: Path to trained RL search model
            rl_algorithm: RL algorithm type ('ppo' or 'sac')
            target_lost_threshold: Steps without target before switching to search
            target_found_threshold: Steps with target before switching to visual
        """
        
        # Visual servoing policy (unchanged)
        self.visual_policy = ImprovedTrackingPolicy(visual_config)
        
        # Load trained RL search policy
        self.rl_policy = None
        self.rl_algorithm = rl_algorithm.lower()
        if STABLE_BASELINES_AVAILABLE:
            try:
                if self.rl_algorithm == 'ppo':
                    self.rl_policy = PPO.load(rl_model_path)
                elif self.rl_algorithm == 'sac':
                    self.rl_policy = SAC.load(rl_model_path)
                else:
                    raise ValueError(f"Unsupported RL algorithm: {rl_algorithm}")
                    
                print(f"✅ {rl_algorithm.upper()} RL search policy loaded successfully")
                print(f"✅ Loaded RL search model from: {rl_model_path}")
            except Exception as e:
                print(f"❌ Failed to load RL model: {e}")
                self.rl_policy = None
        
        # State tracking
        self.current_mode = 'visual'  # 'visual' or 'search'
        self.search_start_time = None
        self.last_target_pos = None
        self.mode_switch_count = 0
        
        # Target visibility tracking
        self.target_lost_threshold = target_lost_threshold
        self.target_found_threshold = target_found_threshold
        self.target_lost_steps = 0
        self.target_found_steps = 0
        
        # Performance metrics
        self.visual_servoing_time = 0.0
        self.search_time = 0.0
        self.last_mode_switch_time = time.time()
        
        print(f"🔀 Hybrid Tracking Policy initialized:")
        print(f"   Visual servoing: {visual_config}")
        print(f"   RL model available: {self.rl_policy is not None}")
        print(f"   Lost threshold: {target_lost_threshold}, Found threshold: {target_found_threshold}")
    
    def predict(self, observation: Dict, deterministic: bool = True) -> np.ndarray:
        """
        Predict action using hybrid policy
        
        Args:
            observation: Environment observation (can be base_obs dict or hybrid dict)
            deterministic: Use deterministic RL policy
            
        Returns:
            Action array for robot joints (normalized [-1,1] for SearchRLEnv)
        """
        
        # Extract base observation for target visibility check
        if 'base_obs' in observation:
            base_obs = observation['base_obs']
            self._rl_state = observation.get('rl_state', None)  # Cache for RL use
        else:
            base_obs = observation
            self._rl_state = None
        
        # Check target visibility
        target_visible = self._is_target_visible(base_obs)
        
        # Update visibility counters
        if target_visible:
            self.target_found_steps += 1
            self.target_lost_steps = 0
        else:
            self.target_lost_steps += 1
            self.target_found_steps = 0
        
        # Decide on mode switch
        should_switch_to_search = (
            self.current_mode == 'visual' and 
            self.target_lost_steps >= self.target_lost_threshold
        )
        
        should_switch_to_visual = (
            self.current_mode == 'search' and 
            self.target_found_steps >= self.target_found_threshold
        )
        
        # Handle mode switching
        if should_switch_to_search:
            self._switch_to_search_mode(observation)
        elif should_switch_to_visual:
            self._switch_to_visual_mode(observation)
        
        # Get action based on current mode
        if self.current_mode == 'visual':
            return self._get_visual_servoing_action(base_obs)
        else:
            return self._get_search_action(base_obs, deterministic)
    
    def _is_target_visible(self, observation: Dict) -> bool:
        """Check if target is visible in current observation"""
        if 'target_in_view' in observation:
            return observation['target_in_view'][0] > 0.5
        elif 'target_center_distance' in observation:
            return observation['target_center_distance'][0] >= 0  # Negative if not visible
        else:
            # Fallback: try to detect target in camera image
            if 'camera_image' in observation:
                target_pos = self.visual_policy._detect_target_position(observation['camera_image'])
                return target_pos is not None
            return False
    
    def _switch_to_search_mode(self, observation: Dict):
        """Switch to RL search mode"""
        if self.current_mode != 'search':
            print("🔍 Target lost! Switching to RL search mode")
            self.current_mode = 'search'
            self.search_start_time = 0
            self.mode_switch_count += 1
            
            # Update timing
            current_time = time.time()
            self.visual_servoing_time += current_time - self.last_mode_switch_time
            self.last_mode_switch_time = current_time
            
            # Store last known target position
            if 'target_pixel_pos' in observation:
                self.last_target_pos = observation['target_pixel_pos'][:3]  # World coordinates
    
    def _switch_to_visual_mode(self, observation: Dict):
        """Switch to visual servoing mode"""
        if self.current_mode != 'visual':
            print("🎯 Target found! Switching to visual servoing mode")
            self.current_mode = 'visual'
            self.search_start_time = None
            self.mode_switch_count += 1
            
            # Update timing
            current_time = time.time()
            self.search_time += current_time - self.last_mode_switch_time
            self.last_mode_switch_time = current_time
    
    def _get_visual_servoing_action(self, base_obs: Dict) -> np.ndarray:
        """Get action from visual servoing policy"""
        # Use base observation directly (no conversion needed)
        visual_action = self.visual_policy.predict(base_obs)
        
        # Visual servoing returns denormalized actions, but SearchRLEnv expects normalized [-1,1]
        # SearchRLEnv now handles visual servoing scaling properly (1.3 rad/s max during visual phase)
        visual_max_velocity = 1.3  # Matches the visual_config max_velocity
        
        # Normalize the visual action to [-1, 1] for SearchRLEnv
        normalized_action = visual_action / visual_max_velocity
        
        # Clip to [-1, 1] range for safety
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        
        return normalized_action
    
    def _get_search_action(self, base_obs: Dict, deterministic: bool = True) -> np.ndarray:
        """Get action from RL search policy"""
        if self.rl_policy is None:
            # Fallback to visual policy's search behavior  
            print("⚠️  No RL model available, using visual policy search")
            visual_action = self.visual_policy.predict(base_obs)
            # Normalize the action for SearchRLEnv
            max_velocities = np.array([8.0, 7.0, 7.0, 5.6, 5.6, 8.0])
            normalized_action = np.clip(visual_action / max_velocities, -1.0, 1.0)
            return normalized_action
        
        # Use cached RL state if available, otherwise prepare from base observation
        if hasattr(self, '_rl_state') and self._rl_state is not None:
            rl_state = self._rl_state
        else:
            # Fallback to preparing state from base observation
            rl_state = self._prepare_rl_state(base_obs)
        
        # Get action from RL policy (already normalized [-1,1])
        action, _ = self.rl_policy.predict(rl_state, deterministic=deterministic)
        
        # Increment search time
        if self.search_start_time is not None:
            self.search_start_time += 1
        
        return action
    
    def _prepare_rl_state(self, observation: Dict) -> np.ndarray:
        """Convert environment observation to RL state vector"""
        
        # Extract joint positions (6D)
        if 'joint_positions' in observation:
            joint_positions = observation['joint_positions'][:6]
        else:
            joint_positions = np.zeros(6)  # Fallback
        
        # Time since search started (normalized) (1D)
        if self.search_start_time is not None:
            time_lost = min(self.search_start_time / 300.0, 1.0)  # Max 30 seconds at 10Hz
        else:
            time_lost = 0.0
        
        # Last known target position (3D)
        if self.last_target_pos is not None:
            last_pos = np.array(self.last_target_pos[:3])
        else:
            last_pos = np.array([0.5, 0.0, 0.3])  # Default workspace center
        
        # Distance from workspace center (1D)
        workspace_center = np.array([0.5, 0.0, 0.3])
        current_ee_pos = self._get_end_effector_position(joint_positions)
        workspace_dist = np.linalg.norm(current_ee_pos - workspace_center)
        workspace_dist = min(workspace_dist / 1.0, 1.0)  # Normalize to [0,1]
        
        # Combine into 11D state vector
        state = np.concatenate([
            joint_positions,      # 6D
            [time_lost],         # 1D  
            last_pos,            # 3D
            [workspace_dist]     # 1D
        ])
        
        return state.astype(np.float32)
    
    def _convert_to_base_observation(self, rl_observation: Dict) -> Dict:
        """
        Convert SearchRLEnv observation back to base environment format
        
        The visual servoing policy expects the base environment observation format,
        but we get SearchRLEnv format. Need to reconstruct the base format.
        """
        # SearchRLEnv state is 11D: [joint_pos(6), time_lost(1), last_target(3), workspace_dist(1)]
        # But we also get the full base observation in some cases
        
        if isinstance(rl_observation, dict) and 'camera_image' in rl_observation:
            # We have the full base observation, just return it
            return rl_observation
        elif isinstance(rl_observation, np.ndarray):
            # We have the 11D RL state vector, need to reconstruct
            joint_positions = rl_observation[:6]
            
            # Create a mock base observation (missing camera image)
            base_obs = {
                'joint_positions': joint_positions,
                'target_in_view': np.array([0.0]),  # Assume not visible in search mode
                'target_center_distance': np.array([-1.0]),  # Not visible
                'camera_image': np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy image
            }
            return base_obs
        else:
            # Fallback - create minimal observation
            return {
                'joint_positions': np.zeros(6),
                'target_in_view': np.array([0.0]),
                'target_center_distance': np.array([-1.0]),
                'camera_image': np.zeros((480, 640, 3), dtype=np.uint8)
            }

    def _get_end_effector_position(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Estimate end-effector position using simplified forward kinematics
        
        For now, this is a simplified approximation. In a full implementation,
        you would use the actual robot's forward kinematics.
        """
        # Simple approximation based on joint angles
        # This should be replaced with proper forward kinematics
        base_rotation = joint_positions[0]
        shoulder_angle = joint_positions[1]
        elbow_angle = joint_positions[2]
        
        # Rough FK approximation (replace with actual robot FK)
        link1_length = 0.3  # Approximate link lengths
        link2_length = 0.25
        
        x = (link1_length * np.cos(shoulder_angle) + 
             link2_length * np.cos(shoulder_angle + elbow_angle)) * np.cos(base_rotation)
        y = (link1_length * np.cos(shoulder_angle) + 
             link2_length * np.cos(shoulder_angle + elbow_angle)) * np.sin(base_rotation)
        z = (link1_length * np.sin(shoulder_angle) + 
             link2_length * np.sin(shoulder_angle + elbow_angle)) + 0.1  # Base height
        
        return np.array([x, y, z])
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the hybrid policy"""
        total_time = self.visual_servoing_time + self.search_time
        
        return {
            'mode_switches': self.mode_switch_count,
            'visual_servoing_time': self.visual_servoing_time,
            'search_time': self.search_time,
            'total_time': total_time,
            'visual_servoing_ratio': self.visual_servoing_time / max(total_time, 1e-6),
            'search_ratio': self.search_time / max(total_time, 1e-6),
            'current_mode': self.current_mode,
            'target_lost_steps': self.target_lost_steps,
            'target_found_steps': self.target_found_steps
        }
    
    def reset(self):
        """Reset policy state for new episode"""
        self.current_mode = 'visual'
        self.search_start_time = None
        self.last_target_pos = None
        self.target_lost_steps = 0
        self.target_found_steps = 0
        self.last_mode_switch_time = time.time()
        
        # Reset underlying policies
        if hasattr(self.visual_policy, 'reset'):
            self.visual_policy.reset()