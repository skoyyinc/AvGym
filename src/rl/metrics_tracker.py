"""
Comprehensive metrics tracking for RL search training

This module provides detailed monitoring and logging capabilities for 
training the search optimization policy.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional
from collections import deque
import csv
import os


class ComprehensiveMetricsTracker:
    """
    Tracks comprehensive metrics during RL training for search optimization
    """
    
    def __init__(self, log_dir: str = "search_rl_logs", window_size: int = 100):
        """
        Initialize metrics tracker
        
        Args:
            log_dir: Directory for saving logs
            window_size: Rolling window size for statistics
        """
        self.log_dir = log_dir
        self.window_size = window_size
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Episode-level metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_outcomes = []  # 'success', 'timeout', 'collision', 'stuck'
        
        # Performance metrics
        self.search_times = []  # Time to find target (successful episodes only)
        self.success_rate_history = []
        self.mean_reward_history = []
        
        # Spatial/behavioral metrics
        self.workspace_coverage = []  # How much workspace explored per episode
        self.joint_usage_patterns = []  # Joint velocity patterns
        self.search_trajectories = []  # Full trajectories for analysis
        
        # Safety metrics
        self.joint_limit_violations = []  # Per episode
        self.collision_count = []
        self.stuck_episodes = []  # Episodes where robot got stuck
        
        # Curriculum metrics
        self.difficulty_level = []
        self.target_distances = []  # Distance from start to target
        self.distractor_counts = []
        
        # Training metrics (from PPO)
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.kl_divergences = []
        self.explained_variances = []
        
        # Rolling windows for efficiency
        self._reward_window = deque(maxlen=window_size)
        self._success_window = deque(maxlen=window_size)
        
        # CSV logger
        self.csv_file = os.path.join(log_dir, "detailed_metrics.csv")
        self._init_csv_logger()
        
        print(f"📊 MetricsTracker initialized: {log_dir}/")
    
    def _init_csv_logger(self):
        """Initialize CSV logger with headers"""
        headers = [
            'episode', 'timestamp', 'reward', 'length', 'outcome',
            'search_time', 'success_rate', 'mean_reward',
            'workspace_coverage', 'joint_violations', 'collisions',
            'target_distance', 'difficulty_level'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def update_episode(self, episode_data: Dict[str, Any]):
        """
        Update metrics after each episode
        
        Args:
            episode_data: Dictionary with episode information
        """
        episode_num = len(self.episode_rewards)
        
        # Basic episode data
        total_reward = episode_data.get('total_reward', 0.0)
        steps = episode_data.get('steps', 0)
        outcome = episode_data.get('outcome', 'unknown')
        
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.episode_outcomes.append(outcome)
        
        # Update rolling windows
        self._reward_window.append(total_reward)
        self._success_window.append(1.0 if outcome == 'success' else 0.0)
        
        # Success-specific metrics
        search_time = 0.0
        if outcome == 'success':
            search_time = episode_data.get('search_time', 0.0)
            self.search_times.append(search_time)
        
        # Rolling statistics
        current_success_rate = np.mean(self._success_window)
        current_mean_reward = np.mean(self._reward_window)
        
        self.success_rate_history.append(current_success_rate)
        self.mean_reward_history.append(current_mean_reward)
        
        # Spatial metrics
        coverage = episode_data.get('workspace_coverage', 0.0)
        self.workspace_coverage.append(coverage)
        
        if 'joint_usage_pattern' in episode_data:
            self.joint_usage_patterns.append(episode_data['joint_usage_pattern'])
        
        if 'trajectory' in episode_data:
            self.search_trajectories.append(episode_data['trajectory'])
        
        # Safety metrics
        violations = episode_data.get('joint_violations', 0)
        collisions = episode_data.get('collisions', 0)
        
        self.joint_limit_violations.append(violations)
        self.collision_count.append(collisions)
        self.stuck_episodes.append(outcome == 'stuck')
        
        # Environment metrics
        difficulty = episode_data.get('difficulty_level', 1)
        target_dist = episode_data.get('target_distance', 0.0)
        distractors = episode_data.get('distractor_count', 0)
        
        self.difficulty_level.append(difficulty)
        self.target_distances.append(target_dist)
        self.distractor_counts.append(distractors)
        
        # Log to CSV
        self._log_to_csv(episode_num, episode_data, current_success_rate, 
                        current_mean_reward, search_time, coverage)
        
        # Print progress every 50 episodes
        if (episode_num + 1) % 50 == 0:
            self._print_progress_update(episode_num + 1)
    
    def update_training_metrics(self, training_data: Dict[str, float]):
        """
        Update training metrics from PPO
        
        Args:
            training_data: Dictionary with PPO training metrics
        """
        if 'policy_loss' in training_data:
            self.policy_losses.append(training_data['policy_loss'])
        if 'value_loss' in training_data:
            self.value_losses.append(training_data['value_loss'])
        if 'entropy_loss' in training_data:
            self.entropy_losses.append(training_data['entropy_loss'])
        if 'approx_kl' in training_data:
            self.kl_divergences.append(training_data['approx_kl'])
        if 'explained_variance' in training_data:
            self.explained_variances.append(training_data['explained_variance'])
    
    def _log_to_csv(self, episode_num: int, episode_data: Dict, 
                    success_rate: float, mean_reward: float,
                    search_time: float, coverage: float):
        """Log episode data to CSV file"""
        
        row = [
            episode_num,
            time.time(),
            episode_data.get('total_reward', 0.0),
            episode_data.get('steps', 0),
            episode_data.get('outcome', 'unknown'),
            search_time,
            success_rate,
            mean_reward,
            coverage,
            episode_data.get('joint_violations', 0),
            episode_data.get('collisions', 0),
            episode_data.get('target_distance', 0.0),
            episode_data.get('difficulty_level', 1)
        ]
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def _print_progress_update(self, episode_num: int):
        """Print training progress update"""
        
        if not self.success_rate_history:
            return
            
        current_success = self.success_rate_history[-1]
        current_reward = self.mean_reward_history[-1]
        
        # Recent performance
        recent_search_times = self.search_times[-10:] if self.search_times else [0]
        avg_search_time = np.mean(recent_search_times) if recent_search_times else 0
        
        # Safety metrics
        recent_violations = np.mean(self.joint_limit_violations[-50:]) if self.joint_limit_violations else 0
        
        print(f"\n📊 Episode {episode_num} Progress:")
        print(f"   Success Rate: {current_success:.3f}")
        print(f"   Mean Reward: {current_reward:.1f}")
        print(f"   Avg Search Time: {avg_search_time:.1f}s")
        print(f"   Violations/Episode: {recent_violations:.2f}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.episode_rewards:
            return {'status': 'no_data'}
        
        # Basic statistics
        total_episodes = len(self.episode_rewards)
        successful_episodes = self.episode_outcomes.count('success')
        
        summary = {
            # Overall performance
            'total_episodes': total_episodes,
            'overall_success_rate': successful_episodes / total_episodes,
            'mean_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            
            # Search performance
            'mean_search_time': np.mean(self.search_times) if self.search_times else 0.0,
            'median_search_time': np.median(self.search_times) if self.search_times else 0.0,
            'search_time_std': np.std(self.search_times) if self.search_times else 0.0,
            
            # Recent performance (last 100 episodes)
            'recent_success_rate': self.success_rate_history[-1] if self.success_rate_history else 0.0,
            'recent_mean_reward': self.mean_reward_history[-1] if self.mean_reward_history else 0.0,
            
            # Safety metrics
            'avg_joint_violations': np.mean(self.joint_limit_violations) if self.joint_limit_violations else 0.0,
            'avg_collisions': np.mean(self.collision_count) if self.collision_count else 0.0,
            'stuck_episode_rate': np.mean(self.stuck_episodes) if self.stuck_episodes else 0.0,
            
            # Behavioral metrics
            'avg_workspace_coverage': np.mean(self.workspace_coverage) if self.workspace_coverage else 0.0,
            'exploration_diversity': self._compute_exploration_diversity(),
            
            # Training stability
            'reward_trend': self._compute_trend(self.mean_reward_history[-50:]),
            'success_trend': self._compute_trend(self.success_rate_history[-50:]),
        }
        
        # Training metrics (if available)
        if self.policy_losses:
            summary.update({
                'final_policy_loss': self.policy_losses[-1],
                'final_value_loss': self.value_losses[-1] if self.value_losses else 0.0,
                'final_kl_divergence': self.kl_divergences[-1] if self.kl_divergences else 0.0,
                'final_explained_variance': self.explained_variances[-1] if self.explained_variances else 0.0
            })
        
        return summary
    
    def _compute_exploration_diversity(self) -> float:
        """Compute diversity of exploration patterns"""
        if len(self.search_trajectories) < 2:
            return 0.0
        
        # Simplified diversity metric: variance in trajectory lengths
        traj_lengths = [len(traj) for traj in self.search_trajectories[-20:]]
        return np.std(traj_lengths) / (np.mean(traj_lengths) + 1e-8)
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction for a metric"""
        if len(values) < 10:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining" 
        else:
            return "stable"
    
    def save_summary_report(self, filepath: Optional[str] = None):
        """Save comprehensive summary report to file"""
        
        if filepath is None:
            filepath = os.path.join(self.log_dir, "training_summary.txt")
        
        summary = self.get_performance_summary()
        
        with open(filepath, 'w') as f:
            f.write("RL Search Training Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Training Episodes: {summary.get('total_episodes', 0)}\n")
            f.write(f"Overall Success Rate: {summary.get('overall_success_rate', 0):.3f}\n")
            f.write(f"Mean Episode Reward: {summary.get('mean_episode_reward', 0):.2f}\n")
            f.write(f"Mean Search Time: {summary.get('mean_search_time', 0):.2f}s\n\n")
            
            f.write("Safety Metrics:\n")
            f.write(f"  Joint Violations/Episode: {summary.get('avg_joint_violations', 0):.3f}\n")
            f.write(f"  Collision Rate: {summary.get('avg_collisions', 0):.3f}\n")
            f.write(f"  Stuck Episode Rate: {summary.get('stuck_episode_rate', 0):.3f}\n\n")
            
            f.write("Performance Trends:\n")
            f.write(f"  Reward Trend: {summary.get('reward_trend', 'unknown')}\n")
            f.write(f"  Success Trend: {summary.get('success_trend', 'unknown')}\n\n")
            
            if 'final_policy_loss' in summary:
                f.write("Final Training Metrics:\n")
                f.write(f"  Policy Loss: {summary['final_policy_loss']:.6f}\n")
                f.write(f"  Value Loss: {summary['final_value_loss']:.6f}\n")
                f.write(f"  KL Divergence: {summary['final_kl_divergence']:.6f}\n")
                f.write(f"  Explained Variance: {summary['final_explained_variance']:.3f}\n")
        
        print(f"📄 Summary report saved: {filepath}")
    
    def detect_training_issues(self) -> List[str]:
        """
        Detect potential training issues
        
        Returns:
            List of detected issues
        """
        issues = []
        
        if len(self.success_rate_history) < 50:
            return issues
        
        # Check for success rate plateau
        recent_success = self.success_rate_history[-50:]
        if np.std(recent_success) < 0.05 and np.mean(recent_success) < 0.5:
            issues.append("LOW_SUCCESS_RATE_PLATEAU")
        
        # Check for reward collapse
        if len(self.mean_reward_history) >= 20:
            recent_rewards = self.mean_reward_history[-20:]
            if recent_rewards[-1] < -20 and all(r < -15 for r in recent_rewards[-10:]):
                issues.append("REWARD_COLLAPSE")
        
        # Check for excessive safety violations
        if len(self.joint_limit_violations) >= 10:
            recent_violations = self.joint_limit_violations[-10:]
            if np.mean(recent_violations) > 2:
                issues.append("HIGH_SAFETY_VIOLATIONS")
        
        # Check for training instability (if training metrics available)
        if len(self.kl_divergences) >= 10:
            recent_kl = self.kl_divergences[-10:]
            if any(kl > 0.5 for kl in recent_kl):
                issues.append("POLICY_INSTABILITY")
        
        return issues