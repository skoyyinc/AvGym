"""
Utility functions for RL search training
"""

import numpy as np
from typing import List, Dict, Tuple, Any


def normalize_joint_positions(joint_positions: np.ndarray, 
                            joint_limits: List[Tuple[float, float]]) -> np.ndarray:
    """
    Normalize joint positions to [-1, 1] range based on joint limits
    
    Args:
        joint_positions: Current joint angles [rad]
        joint_limits: List of (min, max) tuples for each joint
        
    Returns:
        Normalized joint positions in [-1, 1]
    """
    normalized = np.zeros_like(joint_positions)
    
    for i, (pos, (min_pos, max_pos)) in enumerate(zip(joint_positions, joint_limits)):
        # Normalize to [-1, 1]
        range_size = max_pos - min_pos
        normalized[i] = 2.0 * (pos - min_pos) / range_size - 1.0
        
    return normalized


def denormalize_joint_velocities(normalized_velocities: np.ndarray,
                               max_velocities: List[float]) -> np.ndarray:
    """
    Convert normalized velocities [-1, 1] to actual joint velocities
    
    Args:
        normalized_velocities: Normalized velocity commands [-1, 1]
        max_velocities: Maximum velocity for each joint [rad/s]
        
    Returns:
        Actual joint velocities [rad/s]
    """
    return normalized_velocities * np.array(max_velocities)


def compute_workspace_distance(current_pos: np.ndarray, 
                             workspace_center: np.ndarray = np.array([0.5, 0.0, 0.3])) -> float:
    """
    Compute normalized distance from workspace center
    
    Args:
        current_pos: Current end-effector position [m]
        workspace_center: Center of robot workspace [m]
        
    Returns:
        Normalized distance [0, 1]
    """
    distance = np.linalg.norm(current_pos - workspace_center)
    # Normalize by approximate workspace radius
    workspace_radius = 0.8  # meters
    return min(distance / workspace_radius, 1.0)


def check_joint_limits(joint_positions: np.ndarray,
                      joint_limits: List[Tuple[float, float]],
                      tolerance: float = 0.1) -> bool:
    """
    Check if joint positions are within safe limits
    
    Args:
        joint_positions: Current joint angles [rad]
        joint_limits: List of (min, max) tuples for each joint
        tolerance: Safety margin [rad]
        
    Returns:
        True if within limits, False otherwise
    """
    for pos, (min_pos, max_pos) in zip(joint_positions, joint_limits):
        if pos < (min_pos + tolerance) or pos > (max_pos - tolerance):
            return False
    return True


def compute_exploration_bonus(visited_positions: List[np.ndarray],
                            current_position: np.ndarray,
                            grid_size: float = 0.1) -> float:
    """
    Compute exploration bonus for visiting new areas
    
    Args:
        visited_positions: List of previously visited positions
        current_position: Current end-effector position
        grid_size: Size of exploration grid [m]
        
    Returns:
        Exploration bonus [0, 1]
    """
    if not visited_positions:
        return 1.0
    
    # Discretize current position to grid
    grid_pos = np.round(current_position / grid_size) * grid_size
    
    # Check if this grid cell has been visited
    for prev_pos in visited_positions:
        prev_grid_pos = np.round(prev_pos / grid_size) * grid_size
        if np.allclose(grid_pos, prev_grid_pos, atol=grid_size/2):
            return 0.0  # Already visited
    
    return 1.0  # New area


def moving_average(data: List[float], window: int = 10) -> List[float]:
    """
    Compute moving average of a data series
    
    Args:
        data: Input data series
        window: Window size for averaging
        
    Returns:
        Moving average series
    """
    if len(data) < window:
        return data
    
    averaged = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        window_data = data[start_idx:i+1]
        averaged.append(sum(window_data) / len(window_data))
    
    return averaged


def detect_plateau(metric_history: List[float], 
                  window: int = 50, 
                  threshold: float = 0.01) -> bool:
    """
    Detect if a metric has plateaued (stopped improving)
    
    Args:
        metric_history: History of metric values
        window: Window to check for plateau
        threshold: Minimum improvement threshold
        
    Returns:
        True if plateaued, False otherwise
    """
    if len(metric_history) < window:
        return False
    
    recent_values = metric_history[-window:]
    improvement = max(recent_values) - min(recent_values)
    
    return improvement < threshold