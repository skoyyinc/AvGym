import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import time
from typing import Tuple, Dict, Optional
import math
from pathlib import Path

class SO101CameraTrackingEnv(gym.Env):
    """
    SO-ARM101 with camera end-effector for object tracking
    Goal: Move camera to center target object in view
    """
    
    def __init__(self, render_mode="human", camera_width=640, camera_height=480):
        super().__init__()
        
        # Environment parameters
        self.render_mode = render_mode
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Physics simulation
        self.physics_client = None
        self.robot_id = None
        self.target_object_id = None
        self.distractor_ids = []  # List of blue distractor cubes
        self.plane_id = None
        
        # Robot parameters - SO-ARM101 specific
        self.num_joints = 6  # SO-ARM101 has 6 DOF
        self.joint_indices = []
        self.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        
        # Joint limits (from SO-ARM101 specs)
        self.joint_limits = [
            (-np.pi, np.pi),      # Base rotation
            (-np.pi/2, np.pi/2),  # Shoulder
            (-np.pi/2, np.pi/2),  # Elbow  
            (-np.pi/2, np.pi/2),  # Wrist 1
            (-np.pi/2, np.pi/2),  # Wrist 2
            (-np.pi, np.pi)       # Wrist rotation
        ]
        
        # Action space: joint velocities for 6 DOF arm
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Dict({
            'camera_image': spaces.Box(
                low=0, high=255, 
                shape=(camera_height, camera_width, 3), 
                dtype=np.uint8
            ),
            'joint_positions': spaces.Box(
                low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32
            ),
            'target_in_view': spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            ),
            'target_center_distance': spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )
        })
        
        # Tracking state
        self.target_position = np.array([0.5, 0.0, 0.1])
        self.camera_center = np.array([camera_width//2, camera_height//2])
        self.max_episode_steps = float('inf')  # No episode limit for demo
        self.current_step = 0
        
        # Performance tracking
        self.target_centered_steps = 0
        self.total_tracking_error = 0.0
        
        # URDF path
        self.urdf_path = self._get_urdf_path()
        
    def _get_urdf_path(self):
        """Get path to SO-ARM101 URDF with fallback options"""
        urdf_dir = Path("urdf")
        
        # Try different URDF options in order of preference
        urdf_options = [
            urdf_dir / "so101_new_calib.urdf",   # Proper calibration with camera
            urdf_dir / "so_arm101_camera.urdf",  # With meshes (backup)
            urdf_dir / "so_arm101_simple.urdf",  # Simple geometry
        ]
        
        for urdf_path in urdf_options:
            if urdf_path.exists():
                print(f"📁 Found URDF: {urdf_path}")
                return str(urdf_path)
        
        print("⚠️  No SO-ARM101 URDF found. Please run setup first.")
        return None

    def _create_so101_robot(self):
        """Create SO-ARM101 robot model with better error handling"""
        if self.urdf_path and Path(self.urdf_path).exists():
            print(f"📁 Loading SO-ARM101 URDF: {self.urdf_path}")
            
            base_position = [0, 0, 0]
            base_orientation = p.getQuaternionFromEuler([0, 0, 0])
            
            try:
                self.robot_id = p.loadURDF(
                    self.urdf_path, 
                    base_position, 
                    base_orientation,
                    useFixedBase=True,
                    flags=p.URDF_USE_INERTIA_FROM_FILE
                )
                print("✅ SO-ARM101 URDF loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load URDF: {e}")
                print("🔄 Trying fallback robot...")
                self.robot_id = self._create_fallback_robot()
        else:
            print("🔧 Using fallback robot model")
            self.robot_id = self._create_fallback_robot()
            
        # Setup joint control
        self._setup_joint_control()
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.target_centered_steps = 0
        self.total_tracking_error = 0.0
        
        # Initialize PyBullet
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            # Set initial camera position closer to robot
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,     # Much closer than default
                cameraYaw=45,           # Angle around robot
                cameraPitch=-30,        # Looking down slightly  
                cameraTargetPosition=[0, 0, 0.2]  # Focus on robot base
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)  # 240 Hz for stable simulation
        
        # Create environment
        self._create_environment()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _create_environment(self):
        """Create the simulation environment"""
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create SO-ARM101 robot
        self._create_so101_robot()
        
        # Create target object
        self._create_target_object()
        
        # Create distractor objects
        self._create_distractor_objects()
        
        # Set initial robot configuration
        self._reset_robot_pose()
        
    
        
    def _create_fallback_robot(self):
        """Create fallback robot if URDF fails"""
        # Use a simple 6-DOF arm
        try:
            robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
            print("✅ Using KUKA IIWA as fallback")
            return robot_id
        except:
            print("❌ No suitable robot URDF found")
            return None
            
    def _setup_joint_control(self):
        """Setup joint control for the robot"""
        if self.robot_id is None:
            return
            
        # Get all joints
        num_joints = p.getNumJoints(self.robot_id)
        print(f"🔧 Robot has {num_joints} total joints")
        
        self.joint_indices = []
        joint_info = []
        
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            # Only use revolute and prismatic joints
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
                joint_info.append((i, joint_name, joint_type))
                print(f"  Joint {i}: {joint_name} (type: {joint_type})")
        
        # Limit to 6 controllable joints for SO-ARM101
        self.joint_indices = self.joint_indices[:6]
        self.num_joints = len(self.joint_indices)
        
        print(f"✅ Using {self.num_joints} controllable joints: {self.joint_indices}")
        
        # Enable force/torque sensors for all joints
        for joint_idx in self.joint_indices:
            p.enableJointForceTorqueSensor(self.robot_id, joint_idx, 1)
        
    def _create_target_object(self):
        """Create target object to track"""
        # Create a bright red cube (wee bit smaller)
        cube_size = 0.05
        object_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size, cube_size, cube_size])
        object_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[cube_size, cube_size, cube_size],
            rgbaColor=[1, 0, 0, 1]  # Bright red
        )
        
        # Randomize target position within reach - AVOID FRONT OF ROBOT
        # Robot is at origin facing +X, so avoid the front cone (small Y, positive X)
        position_found = False
        attempts = 0
        
        while not position_found and attempts < 50:
            x = np.random.uniform(0.4, 0.9)  # Further from robot
            y = np.random.uniform(-0.6, 0.6)  # Wider range
            z = np.random.uniform(0.15, 0.7)  # Higher range
            
            # Avoid front cone of robot (where camera initially looks)
            # Front cone: positive X, small |Y| 
            distance_from_origin = np.sqrt(x**2 + y**2)
            angle_from_front = abs(np.arctan2(y, x))  # Angle from +X axis
            
            # MUCH MORE RESTRICTIVE: Force target to sides/back
            # Avoid front 120-degree cone (±60° from front)
            if distance_from_origin > 0.5 and angle_from_front > np.pi/3:  # > 60 degrees from front
                position_found = True
                self.target_position = np.array([x, y, z])
            
            attempts += 1
        
        # Fallback if no position found
        if not position_found:
            # Place target to the side/back - DEFINITELY NOT IN FRONT
            angle = np.random.uniform(2*np.pi/3, 4*np.pi/3)  # 120° to 240° (left side to back to right side)
            distance = np.random.uniform(0.6, 0.8)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)  
            z = np.random.uniform(0.15, 0.6)
            self.target_position = np.array([x, y, z])
        
        self.target_object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=object_shape,
            baseVisualShapeIndex=object_visual,
            basePosition=self.target_position
        )
        
        print(f"🎯 Target object created at {self.target_position}")
    
    def _create_distractor_objects(self):
        """Create blue distractor cubes to test object detection"""
        cube_size = 0.05  # Same size as target
        num_distractors = 3  # Number of blue cubes
        
        # Create collision and visual shapes for blue cubes
        distractor_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size, cube_size, cube_size])
        distractor_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[cube_size, cube_size, cube_size],
            rgbaColor=[0, 0, 1, 1]  # Bright blue
        )
        
        # Ensure minimum distance between objects and robot
        min_distance_between_objects = 0.35  # Minimum 35cm between objects
        min_distance_from_robot = 0.4  # Minimum 40cm from robot base
        positions = [self.target_position]  # Start with target position
        
        for i in range(num_distractors):
            attempts = 0
            while attempts < 50:  # Try up to 50 times to find good position
                # Generate random position - further from robot
                x = np.random.uniform(0.3, 1.0)  # Extended range
                y = np.random.uniform(-0.7, 0.7)  # Wider range
                z = np.random.uniform(0.1, 0.7)   # Higher range
                new_pos = np.array([x, y, z])
                
                # Check minimum distance from robot origin
                distance_from_robot = np.linalg.norm(new_pos)
                if distance_from_robot < min_distance_from_robot:
                    continue
                
                # Check distance from all existing objects
                too_close = False
                for existing_pos in positions:
                    distance = np.linalg.norm(new_pos - existing_pos)
                    if distance < min_distance_between_objects:
                        too_close = True
                        break
                
                if not too_close:
                    # Good position found
                    positions.append(new_pos)
                    
                    # Create distractor cube
                    distractor_id = p.createMultiBody(
                        baseMass=0.1,
                        baseCollisionShapeIndex=distractor_shape,
                        baseVisualShapeIndex=distractor_visual,
                        basePosition=new_pos
                    )
                    self.distractor_ids.append(distractor_id)
                    print(f"🔷 Blue distractor {i+1} created at {new_pos}")
                    break
                    
                attempts += 1
            
            if attempts >= 50:
                print(f"⚠️  Could not place distractor {i+1} after 50 attempts")
        
    def _reset_robot_pose(self):
        """Reset robot to initial pose"""
        if self.robot_id is None or not self.joint_indices:
            return
            
        # Initial joint positions for SO-ARM101
        initial_positions = [0, -0.3, 0.5, 0, 0.2, 0]
        
        # Apply initial positions
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(initial_positions):
                p.resetJointState(self.robot_id, joint_idx, initial_positions[i])
                
        # Let physics settle
        for _ in range(100):
            p.stepSimulation()
            
        print("✅ Robot reset to initial pose")
        
    def step(self, action):
        """Execute one step in the environment"""
        if self.robot_id is None:
            # Return dummy observation if robot failed to load
            return self._get_dummy_observation(), 0, True, False, {}
            
        self.current_step += 1
        
        # Apply action
        self._apply_action(action)
        
        # Step simulation multiple times for stability
        for _ in range(5):
            p.stepSimulation()
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(observation)
        
        # Check termination
        terminated = self._is_terminated(observation)
        truncated = self.current_step >= self.max_episode_steps
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
        
    def _apply_action(self, action):
        """Apply action to robot joints with proper control"""
        if self.robot_id is None or not self.joint_indices:
            return
            
        # Scale action to reasonable velocities
        max_velocity = 1.5  # rad/s
        target_velocities = np.clip(action, -1.0, 1.0) * max_velocity
        
        # Apply velocity control to each joint
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(target_velocities):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=target_velocities[i],
                    maxVelocity=max_velocity,
                    force=100  # Maximum force/torque
                )
                
    def _get_observation(self):
        """Get current observation"""
        if self.robot_id is None:
            return self._get_dummy_observation()
            
        # Get camera image
        camera_image = self._get_camera_image()
        
        # Get joint positions
        joint_positions = self._get_joint_positions()
        
        # Get target info
        target_info = self._get_target_info(camera_image)
        
        observation = {
            'camera_image': camera_image,
            'joint_positions': joint_positions,
            'target_in_view': np.array([target_info['in_view']], dtype=np.float32),
            'target_center_distance': np.array([target_info['center_distance']], dtype=np.float32)
        }
        
        return observation
        
    def _get_dummy_observation(self):
        """Get dummy observation if robot failed to load"""
        return {
            'camera_image': np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8),
            'joint_positions': np.zeros(6, dtype=np.float32),
            'target_in_view': np.array([0.0], dtype=np.float32),
            'target_center_distance': np.array([1.0], dtype=np.float32)
        }
        
    def _get_camera_image(self):
        """Get camera image from end-effector"""
        if self.robot_id is None:
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            
        # Get camera link state (last link or specific camera link)
        camera_link_idx = -1  # Use last link as camera
        
        # Try to find camera_link specifically
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            link_info = p.getJointInfo(self.robot_id, i)
            link_name = link_info[12].decode('utf-8')  # Link name
            if 'camera' in link_name.lower():
                camera_link_idx = i
                break
        
        # Get link state
        if camera_link_idx >= 0:
            link_state = p.getLinkState(self.robot_id, camera_link_idx)
        else:
            # Use end-effector (last joint)
            if self.joint_indices:
                link_state = p.getLinkState(self.robot_id, self.joint_indices[-1])
            else:
                # Fallback to base
                link_state = p.getBasePositionAndOrientation(self.robot_id)
                link_state = (link_state[0], link_state[1])
        
        camera_pos = link_state[0]
        camera_orn = link_state[1]
        
        # Convert quaternion to camera direction
        rotation_matrix = p.getMatrixFromQuaternion(camera_orn)
        
        # Camera looks along +X axis in link frame
        camera_direction = [
            rotation_matrix[0],  # X component
            rotation_matrix[3],  # Y component  
            rotation_matrix[6]   # Z component
        ]
        
        up_vector = [
            rotation_matrix[2],  # X component of Z axis
            rotation_matrix[5],  # Y component of Z axis
            rotation_matrix[8]   # Z component of Z axis
        ]
        
        # Target point for camera
        look_distance = 1.0
        target_pos = [
            camera_pos[0] + camera_direction[0] * look_distance,
            camera_pos[1] + camera_direction[1] * look_distance,
            camera_pos[2] + camera_direction[2] * look_distance
        ]
        
        # Create view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vector
        )

        # Create projection matrix
        proj_matrix = p.computeProjectionMatrixFOV(
           fov=60,  # Field of view
           aspect=self.camera_width / self.camera_height,
           nearVal=0.1,
           farVal=3.0
       )
       
       # Render camera image
        try:
           (_, _, px, _, _) = p.getCameraImage(
               width=self.camera_width,
               height=self.camera_height,
               viewMatrix=view_matrix,
               projectionMatrix=proj_matrix,
               renderer=p.ER_BULLET_HARDWARE_OPENGL
           )
           
           # Convert to RGB
           rgb_array = np.array(px, dtype=np.uint8)
           rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
           
        except Exception as e:
           print(f"⚠️  Camera rendering failed: {e}")
           # Return black image as fallback
           rgb_array = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
           
        return rgb_array
       
    def _get_joint_positions(self):
        """Get current joint positions"""
        if self.robot_id is None or not self.joint_indices:
            return np.zeros(self.num_joints, dtype=np.float32)
            
        joint_positions = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_positions.append(joint_state[0])  # Position
            
        # Pad with zeros if needed
        while len(joint_positions) < self.num_joints:
            joint_positions.append(0.0)
            
        return np.array(joint_positions[:self.num_joints], dtype=np.float32)
        
    def _get_target_info(self, camera_image):
        """Detect target in camera image"""
        target_info = {
            'in_view': 0.0,
            'center_distance': 1.0,
            'pixel_position': None
        }
        
        if camera_image is None or camera_image.size == 0:
            return target_info
            
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(camera_image, cv2.COLOR_RGB2HSV)
            
            # Red color detection (wider range for better detection)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Morphological operations to clean up mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 100:  # Minimum area threshold
                    # Get center of mass
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        target_info['in_view'] = 1.0
                        target_info['pixel_position'] = (cx, cy)
                        
                        # Calculate distance from center
                        center_x, center_y = self.camera_center
                        distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                        max_distance = np.sqrt(center_x**2 + center_y**2)
                        target_info['center_distance'] = min(distance / max_distance, 1.0)
                        
        except Exception as e:
            print(f"⚠️  Target detection failed: {e}")
            
        return target_info
        
    def _calculate_reward(self, observation):
        """Calculate reward for current state"""
        reward = 0.0
        
        target_in_view = observation['target_in_view'][0]
        center_distance = observation['target_center_distance'][0]
        
        # Base reward for having target in view
        if target_in_view > 0.5:
            reward += 2.0
            
            # Strong reward for centering target
            centering_reward = (1.0 - center_distance) * 5.0
            reward += centering_reward
            
            # Bonus for very good centering
            if center_distance < 0.1:
                self.target_centered_steps += 1
                reward += 10.0
            elif center_distance < 0.2:
                reward += 5.0
                
        else:
            # Penalty for losing target
            reward -= 1.0
            
        # Small penalty for excessive joint movements
        joint_positions = observation['joint_positions']
        if len(joint_positions) > 0:
            movement_penalty = np.sum(np.abs(joint_positions)) * 0.01
            reward -= movement_penalty
            
        # Track total error
        self.total_tracking_error += center_distance
        
        return reward
        
    def _is_terminated(self, observation):
        """Check if episode should terminate - disabled for demo"""
        # No auto-termination for demo mode
        return False
        
    def _get_info(self):
        """Get additional info"""
        return {
            'target_centered_steps': self.target_centered_steps,
            'average_tracking_error': self.total_tracking_error / max(self.current_step, 1),
            'current_step': self.current_step,
            'robot_loaded': self.robot_id is not None
        }
        
    def render(self):
        """Render environment"""
        if self.render_mode == "human":
            pass  # PyBullet GUI handles rendering
        return None
        
    def close(self):
        """Close environment"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)