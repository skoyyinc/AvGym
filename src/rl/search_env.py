"""
RL Environment wrapper for search behavior optimization

This module wraps the SO101CameraTrackingEnv to focus specifically on 
search episodes when the target object is lost from view.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import time

from ..so_arm_gym_env import SO101CameraTrackingEnv
from .utils import (
    normalize_joint_positions, 
    compute_workspace_distance,
    check_joint_limits,
    compute_exploration_bonus
)


class SearchRLEnv(gym.Env):
    """
    Gymnasium environment wrapper for SO101 search behavior training
    
    This environment focuses specifically on search episodes - when the target
    is lost and the robot must find it again using RL-optimized search patterns.
    """
    
    def __init__(self, 
                 base_env: Optional[SO101CameraTrackingEnv] = None,
                 max_search_steps: int = 300,  # 30 seconds at 10Hz
                 reward_config: Optional[Dict] = None,
                 max_joint_velocities: Optional[List[float]] = None,
                 static_target: bool = True,
                 target_position: Optional[List[float]] = None,
                 min_visual_servoing_steps: int = 100,  # 10 seconds at 10Hz
                 centering_threshold: float = 0.1,
                 training_mode: bool = False,
                 randomize_position: bool = False,
                 position_deviation: float = 0.15,
                 # Occlusion curriculum parameters
                 distractors: bool = False,
                 num_distractors: int = 2,
                 distractor_gap: float = 0.08,
                 occlusion_placement: str = 'between_robot_target'):
        """
        Initialize search RL environment
        
        Args:
            base_env: Base SO101 environment (creates new if None)
            max_search_steps: Maximum steps per search episode
            reward_config: Custom reward configuration
            max_joint_velocities: Max velocity for each joint [rad/s] (uses defaults if None)
        """
        super().__init__()
        
        # Create or use provided base environment
        if base_env is None:
            self.base_env = SO101CameraTrackingEnv(render_mode="rgb_array")
        else:
            self.base_env = base_env
            
        # Episode configuration
        self.max_search_steps = max_search_steps
        self.current_step = 0
        self.search_start_time = None
        
        # Robot configuration (from base environment)
        self.num_joints = 6
        self.joint_limits = [
            (-np.pi, np.pi),      # Base rotation
            (-np.pi/2, np.pi/2),  # Shoulder
            (-np.pi/2, np.pi/2),  # Elbow  
            (-np.pi/2, np.pi/2),  # Wrist 1
            (-np.pi/2, np.pi/2),  # Wrist 2
            (-np.pi, np.pi)       # Wrist rotation
        ]
        
        # Joint velocity limits - doubled for very aggressive search behavior
        if max_joint_velocities is None:
            self.max_joint_velocities = [
                8.0,  # Base rotation - extremely fast for wide search sweeps
                7.0,  # Shoulder - doubled for large arm movements
                7.0,  # Elbow - doubled for large arm movements
                5.6,  # Wrist 1 - doubled but controlled
                5.6,  # Wrist 2 - doubled but controlled
                8.0   # Wrist rotation - extremely fast for orientation changes
            ]
        else:
            if len(max_joint_velocities) != 6:
                raise ValueError("max_joint_velocities must have 6 values for 6-DOF robot")
            self.max_joint_velocities = max_joint_velocities
        
        # Minimum velocity limits - ensure robot always moves with significant speed
        self.min_joint_velocities = [
            1.0,  # Base rotation - minimum pan speed
            0.8,  # Shoulder - minimum arm movement
            0.8,  # Elbow - minimum arm movement  
            0.5,  # Wrist 1 - minimum wrist movement
            0.5,  # Wrist 2 - minimum wrist movement
            1.0   # Wrist rotation - minimum orientation change
        ]
        
        # Target positioning configuration
        self.static_target = static_target
        self.randomize_position = randomize_position
        self.position_deviation = position_deviation
        
        if target_position is not None:
            self.base_target_position = np.array(target_position)
        else:
            # Default static position: left side, further away
            angle = np.radians(70)   # 75° from front (proper left side)
            distance = 2     # 0.75m from robot
            height = 0.35            # 35cm height
            self.base_target_position = np.array([
                distance * np.cos(angle),
                distance * np.sin(angle),
                height
            ])
        
        # Current target position (will be randomized if enabled)
        self.custom_target_position = self.base_target_position.copy()
        
        # State tracking
        self.last_target_position = self.custom_target_position.copy()
        self.visited_positions = []
        self.previous_joint_positions = None  # For tracking movement
        self.pan_direction_history = []  # Track pan direction for consistency bonus
        self.episode_data = {}
        
        # Training vs evaluation mode
        self.training_mode = training_mode
        
        # Visual servoing phase tracking (only for evaluation/hybrid mode)
        self.min_visual_servoing_steps = min_visual_servoing_steps
        self.centering_threshold = centering_threshold
        self.visual_servoing_start_step = None
        self.visual_servoing_steps = 0
        self.target_centered = False
        
        # Occlusion curriculum parameters
        self.distractors = distractors
        self.num_distractors = num_distractors
        self.distractor_gap = distractor_gap
        self.occlusion_placement = occlusion_placement
        self.distractor_positions = []  # Will store distractor cube positions
        
        # Simplified reward configuration for focused search training
        default_rewards = {
            'target_found': 100.0,           # Large reward for success
            'active_search_penalty': -0.1,   # Small penalty for active movement (time cost)
            'idle_penalty': -1.0,            # Larger penalty for not moving
            'joint_violation_penalty': -5.0, # Safety constraint penalty
            'stuck_penalty': -2.0            # Penalty for being stuck
        }
        self.reward_config = reward_config or default_rewards
        
        # Define spaces
        self._define_spaces()
        
        print("🔍 SearchRLEnv initialized:")
        print(f"   Max search steps: {self.max_search_steps}")
        print(f"   State space: {self.observation_space.shape}")
        print(f"   Action space: {self.action_space.shape} (2 DOF: shoulder pan + lift only)")
        print(f"   Search joints: 0=pan, 1=lift | Other joints fixed at 0")
        print(f"   Max joint velocities: {self.max_joint_velocities} rad/s")
        print(f"   Min joint velocities: {self.min_joint_velocities} rad/s")
        if self.static_target:
            x, y, z = self.custom_target_position
            angle_deg = np.degrees(np.arctan2(y, x))
            distance = np.linalg.norm([x, y])
            print(f"   Static target: {angle_deg:.1f}°, {distance:.2f}m, height {z:.2f}m")
    
    def _define_spaces(self):
        """Define observation and action spaces"""
        
        # State space: [joint_pos(6), time_lost(1), last_target_pos(3), workspace_dist(1)] = 11D
        state_low = np.array([
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  # Normalized joint positions [-1, 1]
            0.0,                                   # Time since lost [0, 1] 
            0.0, -1.0, 0.0,                       # Last target position [m] (workspace bounds)
            0.0                                    # Workspace distance [0, 1]
        ], dtype=np.float32)
        
        state_high = np.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,        # Normalized joint positions
            1.0,                                   # Time since lost (normalized)
            1.0, 1.0, 1.0,                        # Last target position 
            1.0                                    # Workspace distance
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)
        
        # Action space: joint velocities [-1, 1] for only 2 search joints (shoulder pan + lift)
        # Joint 0: Base rotation (shoulder pan) - horizontal search
        # Joint 1: Shoulder lift - vertical search  
        self.search_joints = 2  # Only first 2 joints
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.search_joints,), dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to start of search episode
        
        Returns:
            observation: Initial state for search episode
            info: Episode information
        """
        super().reset(seed=seed)
        
        # Reset base environment
        base_obs, base_info = self.base_env.reset()
        
        # Force target to be lost (simulate search scenario)
        self._setup_search_scenario()
        
        # Reset episode tracking
        self.current_step = 0
        self.search_start_time = time.time()
        self.visited_positions = []
        self.previous_joint_positions = None
        
        # Reset visual servoing phase tracking
        self.visual_servoing_start_step = None
        self.visual_servoing_steps = 0
        self.target_centered = False
        self.pan_direction_history = []
        
        # Initialize episode data for metrics
        self.episode_data = {
            'start_joint_positions': self.base_env._get_joint_positions().copy(),
            'target_position': self.base_env.target_position.copy(),
            'search_start_time': self.search_start_time,
            'joint_violations': 0,
            'collisions': 0,
            'exploration_cells_visited': set(),
            'stuck_counter': 0
        }
        
        # Get initial RL observation
        observation = self._get_rl_observation()
        info = self._get_episode_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the search environment
        
        Args:
            action: Normalized joint velocity commands [-1, 1]
            
        Returns:
            observation: Next state
            reward: Step reward
            terminated: Episode terminated (target found)
            truncated: Episode truncated (timeout/failure)
            info: Step information
        """
        self.current_step += 1
        
        # Expand 2D action (shoulder pan + lift) to full 6D action space
        # Set other joints (elbow, wrists) to zero for pure search behavior
        full_action = np.zeros(self.num_joints, dtype=np.float32)
        full_action[0] = action[0]  # Shoulder pan (base rotation)
        full_action[1] = action[1]  # Shoulder lift
        # Joints 2-5 (elbow, wrist1, wrist2, wrist3) remain at 0
        
        # Denormalize action based on mode
        if self.training_mode:
            # Training mode: always use aggressive search scaling (no visual servoing)
            denormalized_action = full_action * np.array(self.max_joint_velocities)
            
            # Apply minimum velocity enforcement for guaranteed movement during training
            # Only apply to search joints (0 and 1), leave others at 0
            for i in range(self.search_joints):
                if abs(denormalized_action[i]) < self.min_joint_velocities[i]:
                    # If action is too small, boost it to minimum velocity while preserving direction
                    sign = np.sign(denormalized_action[i]) if denormalized_action[i] != 0 else np.random.choice([-1, 1])
                    denormalized_action[i] = sign * self.min_joint_velocities[i]
        else:
            # Evaluation/hybrid mode: use different scaling for visual servoing vs search
            if self.visual_servoing_start_step is not None:  # Visual servoing mode
                # During visual servoing, use gentler max velocities for smooth tracking
                visual_max_velocities = np.array([1.3, 1.3, 1.3, 1.3, 1.3, 1.3])  # Consistent with visual policy
                denormalized_action = full_action * visual_max_velocities
            else:  # Search mode
                # During search, use aggressive max velocities for fast exploration  
                denormalized_action = full_action * np.array(self.max_joint_velocities)
            
            # Apply minimum velocity enforcement ONLY during search phase (not visual servoing)
            # Only apply to search joints (0 and 1), leave others at 0
            if self.visual_servoing_start_step is None:  # Still in search mode
                for i in range(self.search_joints):
                    if abs(denormalized_action[i]) < self.min_joint_velocities[i]:
                        # If action is too small, boost it to minimum velocity while preserving direction
                        sign = np.sign(denormalized_action[i]) if denormalized_action[i] != 0 else np.random.choice([-1, 1])
                        denormalized_action[i] = sign * self.min_joint_velocities[i]
        
        base_obs, base_reward, base_terminated, base_truncated, base_info = self.base_env.step(denormalized_action)
        
        # Check if target was found
        target_found = base_obs['target_in_view'][0] > 0.5
        center_distance = abs(base_obs['target_center_distance'][0])
        
        # Track visual servoing phase (only in evaluation/hybrid mode)
        if not self.training_mode:
            if target_found:
                if self.visual_servoing_start_step is None:
                    self.visual_servoing_start_step = self.current_step
                    print(f"🎯 Target found at step {self.current_step}! Starting visual servoing phase...")
                    print(f"   Switching to gentle action scaling (1.3 rad/s max)")
                
                self.visual_servoing_steps = self.current_step - self.visual_servoing_start_step
                
                # Check if target is centered
                self.target_centered = center_distance < self.centering_threshold
                
                # Progress logging every 50 steps during visual servoing
                if self.visual_servoing_steps > 0 and self.visual_servoing_steps % 50 == 0:
                    print(f"   📏 Visual servoing step {self.visual_servoing_steps}: center_distance={center_distance:.3f}, centered={self.target_centered}")
                
            else:
                # Target lost, reset visual servoing tracking
                if self.visual_servoing_start_step is not None:
                    print("❌ Target lost during visual servoing! Returning to search mode...")
                    print(f"   Switching back to aggressive action scaling (8.0+ rad/s max)")
                self.visual_servoing_start_step = None
                self.visual_servoing_steps = 0
                self.target_centered = False
        else:
            # Training mode: no visual servoing tracking needed
            self.target_centered = target_found  # Simple: found = success
        
        # Update episode tracking
        current_joint_pos = base_obs['joint_positions']
        self._update_episode_tracking(current_joint_pos, action, target_found)
        
        # Compute RL-specific reward
        reward = self._compute_reward(base_obs, action, target_found)
        
        # Check termination conditions based on mode
        if self.training_mode:
            # Training mode: never terminate early, learn both search + visual servoing
            terminated = False
        else:
            # Evaluation/hybrid mode: terminate when target is well centered
            terminated = (
                target_found and 
                self.target_centered and 
                self.visual_servoing_steps >= self.min_visual_servoing_steps
            )
        
        truncated = (
            self.current_step >= self.max_search_steps
        )
        
        # Get RL observation and info
        observation = self._get_rl_observation()
        info = self._get_episode_info()
        
        # Add search-specific info
        info.update({
            'target_found': target_found,
            'target_centered': self.target_centered,
            'visual_servoing_steps': self.visual_servoing_steps,
            'center_distance': center_distance if target_found else -1.0,
            'search_time': time.time() - self.search_start_time,
            'steps_taken': self.current_step,
            'outcome': 'success' if terminated else ('timeout' if truncated else 'ongoing'),
            'phase': 'visual_servoing' if target_found else 'search'
        })
        
        return observation, reward, terminated, truncated, info
    
    def get_hybrid_observation(self) -> Tuple[np.ndarray, Dict]:
        """
        Get both RL state vector and base environment observation
        
        This is needed for hybrid policies that need both the RL state
        and access to camera image for target visibility detection.
        
        Returns:
            rl_state: 11D state vector for RL policy
            base_obs: Full base environment observation dict
        """
        rl_state = self._get_rl_observation()
        base_obs = self.base_env._get_observation()
        return rl_state, base_obs
    
    def _setup_search_scenario(self):
        """
        Setup the environment for a search scenario
        Uses static or randomized target placement based on configuration
        """
        if self.static_target:
            # Apply position randomization if enabled
            if self.randomize_position:
                # Generate small random deviations from base position
                x_deviation = np.random.uniform(-self.position_deviation, self.position_deviation)
                y_deviation = np.random.uniform(-self.position_deviation, self.position_deviation)
                z_deviation = np.random.uniform(-self.position_deviation * 0.6, self.position_deviation * 0.6)  # Less Z variation
                
                # Apply deviations to base position
                self.custom_target_position = self.base_target_position + np.array([
                    x_deviation, y_deviation, z_deviation
                ])
                
                # Ensure position stays within reasonable bounds
                self.custom_target_position[0] = np.clip(self.custom_target_position[0], 0.3, 2.0)  # X bounds
                self.custom_target_position[1] = np.clip(self.custom_target_position[1], -0.8, 0.8)  # Y bounds  
                self.custom_target_position[2] = np.clip(self.custom_target_position[2], 0.1, 0.6)  # Z bounds
                
                print(f"🎲 Randomized target: deviation=({x_deviation:.3f}, {y_deviation:.3f}, {z_deviation:.3f})")
            else:
                # Use fixed position
                self.custom_target_position = self.base_target_position.copy()
                
            # Set target position in environment
            self.base_env.target_position = self.custom_target_position.copy()
            self.last_target_position = self.base_env.target_position.copy()
            
            # Debug output for target placement validation
            x, y, z = self.base_env.target_position
            target_angle_deg = np.degrees(np.arctan2(y, x))
            target_distance = np.linalg.norm([x, y])
            
            if self.randomize_position:
                print(f"🎯 Random target: angle={target_angle_deg:.1f}°, distance={target_distance:.2f}m, height={z:.2f}m")
            else:
                print(f"🎯 Static target: angle={target_angle_deg:.1f}°, distance={target_distance:.2f}m, height={z:.2f}m")
            print(f"   Position: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            
            # Setup distractor objects for occlusion curriculum
            if self.distractors:
                self._setup_distractor_objects()
            
        else:
            # Original dynamic placement logic (kept for future use)
            print("⚠️  Dynamic target placement not implemented in simplified version")
            # Fallback to static position
            self.base_env.target_position = self.custom_target_position.copy()
            self.last_target_position = self.base_env.target_position.copy()
        
        # Update target object position in simulation
        if self.base_env.target_object_id is not None:
            import pybullet as p
            p.resetBasePositionAndOrientation(
                self.base_env.target_object_id,
                self.base_env.target_position,
                [0, 0, 0, 1]
            )
    
    def _get_rl_observation(self) -> np.ndarray:
        """
        Convert base environment observation to RL state vector
        
        Returns:
            11D state vector for RL policy
        """
        # Get base environment observation
        base_obs = self.base_env._get_observation()
        
        # Extract joint positions and normalize
        joint_positions = base_obs['joint_positions']
        normalized_joints = normalize_joint_positions(joint_positions, self.joint_limits)
        
        # Time since search started (normalized to [0, 1])
        time_lost = min(self.current_step / self.max_search_steps, 1.0)
        
        # Last known target position (normalized to workspace bounds)
        target_pos_norm = np.array([
            (self.last_target_position[0] - 0.2) / 0.8,  # X: [0.2, 1.0] -> [0, 1]
            (self.last_target_position[1] + 0.5) / 1.0,  # Y: [-0.5, 0.5] -> [0, 1]  
            (self.last_target_position[2] - 0.1) / 0.5   # Z: [0.1, 0.6] -> [0, 1]
        ])
        target_pos_norm = np.clip(target_pos_norm, 0.0, 1.0)
        
        # Current workspace distance (normalized)
        current_ee_pos = self._estimate_end_effector_position(joint_positions)
        workspace_dist = compute_workspace_distance(current_ee_pos)
        
        # Combine into state vector
        state = np.concatenate([
            normalized_joints,     # 6D
            [time_lost],          # 1D
            target_pos_norm,      # 3D  
            [workspace_dist]      # 1D
        ]).astype(np.float32)
        
        return state
    
    def _estimate_end_effector_position(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Rough estimate of end-effector position for workspace calculations
        
        In a full implementation, this would use proper forward kinematics.
        For now, using a simplified approximation.
        """
        # Simplified FK approximation
        # This is just for workspace distance calculation
        base_angle = joint_positions[0]
        shoulder_angle = joint_positions[1] 
        elbow_angle = joint_positions[2]
        
        # Rough arm lengths (approximate SO101 dimensions)
        l1 = 0.15  # Shoulder to elbow
        l2 = 0.15  # Elbow to wrist
        
        # 2D projection in base frame
        elbow_x = l1 * np.cos(shoulder_angle)
        elbow_z = l1 * np.sin(shoulder_angle)
        
        ee_x = elbow_x + l2 * np.cos(shoulder_angle + elbow_angle)
        ee_z = elbow_z + l2 * np.sin(shoulder_angle + elbow_angle) + 0.1  # Base height
        
        # Rotate by base angle
        ee_world_x = ee_x * np.cos(base_angle)
        ee_world_y = ee_x * np.sin(base_angle)
        ee_world_z = ee_z
        
        return np.array([ee_world_x, ee_world_y, ee_world_z])
    
    def _compute_reward(self, base_obs: Dict, action: np.ndarray, target_found: bool) -> float:
        """
        Combined reward function for search + visual servoing training
        
        When target not found: encourage search behavior
        When target found: apply visual servoing rewards for centering
        
        Args:
            base_obs: Base environment observation
            action: Action taken this step
            target_found: Whether target was found
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        if target_found:
            # Visual servoing phase: reward for centering the target
            center_distance = abs(base_obs['target_center_distance'][0])
            
            # Large reward for keeping target in view
            reward += 10.0
            
            # Distance-based reward for centering (closer to center = higher reward)
            centering_reward = 20.0 * (1.0 - center_distance)  # 0-20 points based on centering
            reward += centering_reward
            
            # Bonus for being very well centered
            if center_distance < 0.1:  # Well centered
                reward += 30.0
            elif center_distance < 0.2:  # Reasonably centered  
                reward += 10.0
            
            # Small movement penalty to encourage gentle corrections
            action_magnitude = np.linalg.norm(action)
            reward -= 0.05 * action_magnitude  # Gentle movement penalty
            
        else:
            # Search phase: encourage exploration to find target
            action_magnitude = np.linalg.norm(action)
            if action_magnitude > 0.3:
                reward -= 0.1  # Small time penalty for active search
            else:
                reward -= 1.0  # Larger penalty for not moving (encourages exploration)
        
        # Safety constraint: joint limits (always applied)
        joint_positions = base_obs['joint_positions']
        if not check_joint_limits(joint_positions, self.joint_limits):
            reward -= 5.0  # Moderate penalty for violating joint limits
            self.episode_data['joint_violations'] += 1
        
        # Update tracking for stuck detection
        current_ee_pos = self._estimate_end_effector_position(joint_positions)
        self.visited_positions.append(current_ee_pos.copy())
        
        # Simple stuck penalty (encourage movement) - only during search
        if not target_found and len(self.visited_positions) >= 2:
            recent_movement = np.linalg.norm(
                self.visited_positions[-1] - self.visited_positions[-2]
            )
            if recent_movement < 0.02:  # Less than 2cm movement
                self.episode_data['stuck_counter'] += 1
                if self.episode_data['stuck_counter'] > 3:
                    reward -= 2.0  # Penalty for being stuck
            else:
                self.episode_data['stuck_counter'] = 0
        
        # Update previous joint positions for next step
        self.previous_joint_positions = joint_positions.copy()
        
        return reward
    
    def _update_episode_tracking(self, joint_positions: np.ndarray, 
                               action: np.ndarray, target_found: bool):
        """Update episode tracking data for metrics"""
        
        # Track exploration
        current_ee_pos = self._estimate_end_effector_position(joint_positions)
        grid_pos = tuple(np.round(current_ee_pos / 0.1).astype(int))
        self.episode_data['exploration_cells_visited'].add(grid_pos)
    
    def _get_episode_info(self) -> Dict[str, Any]:
        """Get episode information for logging and analysis"""
        
        info = {
            'episode_step': self.current_step,
            'max_steps': self.max_search_steps,
            'joint_violations': self.episode_data['joint_violations'],
            'exploration_coverage': len(self.episode_data['exploration_cells_visited']),
            'stuck_counter': self.episode_data['stuck_counter']
        }
        
        return info
    
    def render(self, mode: str = 'rgb_array'):
        """Render the environment (delegates to base environment)"""
        return self.base_env.render()
    
    def _setup_distractor_objects(self):
        """
        Setup distractor objects for occlusion curriculum
        Places blue cubes between robot and target to create occlusion challenges
        """
        import pybullet as p
        
        # Robot base position (assumed at origin)
        robot_pos = np.array([0.0, 0.0, 0.0])
        target_pos = self.base_env.target_position.copy()
        
        # Calculate direction vector from robot to target
        direction = target_pos - robot_pos
        direction_norm = np.linalg.norm(direction[:2])  # Only X,Y for horizontal direction
        direction_unit = direction[:2] / direction_norm if direction_norm > 0 else np.array([1.0, 0.0])
        
        # Generate distractor positions between robot and target
        self.distractor_positions = []
        distractor_ids = []
        
        for i in range(self.num_distractors):
            # Place distractors at different distances along the robot-target line
            # Closer to target but with minimum gap
            distance_ratio = 0.6 + (i * 0.1)  # Start at 60% of distance, increase by 10%
            base_distance = direction_norm * distance_ratio
            
            # Add some lateral offset to avoid perfect alignment
            lateral_offset = np.random.uniform(-0.05, 0.05)  # ±5cm lateral variation
            perpendicular = np.array([-direction_unit[1], direction_unit[0]])  # Perpendicular vector
            
            # Calculate distractor position
            distractor_pos_2d = robot_pos[:2] + direction_unit * base_distance + perpendicular * lateral_offset
            distractor_height = target_pos[2] + np.random.uniform(-0.02, 0.02)  # Similar height to target
            distractor_pos = np.array([distractor_pos_2d[0], distractor_pos_2d[1], distractor_height])
            
            # Ensure minimum gap from target
            dist_to_target = np.linalg.norm(distractor_pos - target_pos)
            if dist_to_target < self.distractor_gap:
                # Move distractor away from target while maintaining robot-target line
                move_direction = (distractor_pos - target_pos) / dist_to_target
                distractor_pos = target_pos + move_direction * self.distractor_gap
            
            self.distractor_positions.append(distractor_pos)
            
            # Create blue cube as distractor object
            try:
                if hasattr(self.base_env, 'physics_client'):
                    # Create blue cube (0.09m = 9cm cube, slightly smaller than 10cm target)
                    distractor_half_size = 0.045  # 9cm total size vs 10cm target
                    cube_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[distractor_half_size, distractor_half_size, distractor_half_size])
                    cube_visual = p.createVisualShape(
                        p.GEOM_BOX, 
                        halfExtents=[distractor_half_size, distractor_half_size, distractor_half_size],
                        rgbaColor=[0.2, 0.4, 0.8, 1.0]  # Blue color
                    )
                    
                    distractor_id = p.createMultiBody(
                        baseMass=0.1,  # Light weight
                        baseCollisionShapeIndex=cube_shape,
                        baseVisualShapeIndex=cube_visual,
                        basePosition=distractor_pos,
                        physicsClientId=self.base_env.physics_client
                    )
                    distractor_ids.append(distractor_id)
                    
                    print(f"🔵 Distractor {i+1}: x={distractor_pos[0]:.3f}, y={distractor_pos[1]:.3f}, z={distractor_pos[2]:.3f}")
                    print(f"   Distance to target: {np.linalg.norm(distractor_pos - target_pos):.3f}m")
                
            except Exception as e:
                print(f"⚠️  Could not create distractor {i+1}: {e}")
        
        # Store distractor object IDs for cleanup if needed
        if not hasattr(self, 'distractor_ids'):
            self.distractor_ids = []
        self.distractor_ids.extend(distractor_ids)
        
        if self.distractor_positions:
            print(f"✅ Created {len(self.distractor_positions)} blue distractor cubes for occlusion")
            print(f"   Placement: {self.occlusion_placement}")
            print(f"   Minimum gap to target: {self.distractor_gap:.3f}m")
    
    def close(self):
        """Close the environment"""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()