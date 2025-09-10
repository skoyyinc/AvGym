"""
PPO Agent for search behavior optimization

This module implements the PPO-based agent for learning intelligent
search strategies when the target object is lost from view.
"""

import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import json

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
    from stable_baselines3.common import results_plotter
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("⚠️  stable-baselines3 not available. Please install: pip install stable-baselines3[extra]")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available. Install for progress bars: pip install tqdm")

from .metrics_tracker import ComprehensiveMetricsTracker


# Using stable-baselines3 Monitor wrapper for automatic metrics collection


class ProgressCallback(BaseCallback):
    """
    Custom callback for training progress tracking with tqdm
    """
    
    def __init__(self, total_timesteps: int, algorithm: str = "RL", verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.algorithm = algorithm.upper()
        self.pbar = None
        
    def _on_training_start(self) -> None:
        if TQDM_AVAILABLE:
            self.pbar = tqdm(
                total=self.total_timesteps,
                desc=f"🤖 {self.algorithm} Training",
                unit="steps",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )
        
    def _on_step(self) -> bool:
        if self.pbar is not None:
            # Update progress bar to current timestep (don't increment by 1 each time!)
            current_progress = min(self.num_timesteps, self.total_timesteps)
            self.pbar.n = current_progress
            
            # Update metrics every 1000 steps
            if self.num_timesteps % 1000 == 0:
                # Get recent episode info if available
                if hasattr(self.training_env, 'get_attr'):
                    try:
                        # Try to get episode info from vectorized env
                        episode_rewards = self.training_env.get_attr('episode_rewards')[0] if hasattr(self.training_env.get_attr('episode_rewards')[0], '__len__') else []
                        if episode_rewards and len(episode_rewards) > 0:
                            recent_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                            self.pbar.set_postfix({
                                'reward': f'{recent_reward:.1f}',
                                'episodes': len(episode_rewards)
                            })
                    except:
                        # Fallback to basic timestep info
                        self.pbar.set_postfix({'step': f'{self.num_timesteps:,}'})
                else:
                    self.pbar.set_postfix({'step': f'{self.num_timesteps:,}'})
                
                # Refresh the display
                self.pbar.refresh()
            
            # Check if we've reached the end
            if self.num_timesteps >= self.total_timesteps:
                self.pbar.set_description(f"🤖 {self.algorithm} Training Complete")
                return False  # Signal to stop training
        
        return True
    
    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class SearchRLAgent:
    """
    RL agent for SO101 search behavior optimization
    
    Implements PPO or SAC training for the search environment with comprehensive
    logging, checkpointing, and evaluation capabilities.
    """
    
    def __init__(self, 
                 env,
                 algorithm: str = "ppo",
                 log_dir: str = "search_rl_logs",
                 model_save_path: str = "models/search_rl_model",
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 device: str = 'cpu',
                 verbose: int = 1,
                 curriculum_type: str = "static_cube"):
        """
        Initialize RL agent for search training
        
        Args:
            env: Search environment (SearchRLEnv or vectorized)
            algorithm: Algorithm to use ('ppo' or 'sac')
            log_dir: Directory for training logs
            model_save_path: Path to save trained models
            learning_rate: Learning rate
            n_steps: Steps per rollout (PPO only)
            batch_size: Mini-batch size for optimization
            n_epochs: Optimization epochs per rollout (PPO only)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter (PPO only)
            clip_range: PPO clipping range (PPO only)
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient (PPO only)
            device: Training device (cpu/cuda)
            verbose: Verbosity level
            curriculum_type: Type of curriculum for naming (static_cube, etc.)
        """
        
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable-baselines3 required for RL training")
        
        self.algorithm = algorithm.lower()
        if self.algorithm not in ['ppo', 'sac']:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose 'ppo' or 'sac'")
            
        self.verbose = verbose
        self.curriculum_type = curriculum_type
        
        # Generate unique run identifier with curriculum type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"search_{self.algorithm}_{curriculum_type}_{timestamp}"
        
        # Create unique paths with curriculum type
        self.log_dir = os.path.join(log_dir, self.run_id)
        self.monitor_dir = os.path.join(self.log_dir, "monitor_logs")
        self.model_save_path = f"{model_save_path}_{self.algorithm}_{curriculum_type}_{timestamp}"

        
        print(f"🆔 Training run ID: {self.run_id}")
        print(f"📁 Log directory: {self.log_dir}")
        print(f"💾 Model save path: {self.model_save_path}")
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.monitor_dir, exist_ok=True)
        self.env = Monitor(env, self.monitor_dir)


        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # Algorithm-specific hyperparameters
        if self.algorithm == 'ppo':
            self.model_config = {
                'policy': 'MlpPolicy',
                'env': self.env,
                'learning_rate': learning_rate,
                'n_steps': n_steps,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'gamma': gamma,
                'gae_lambda': gae_lambda,
                'clip_range': clip_range,
                'ent_coef': ent_coef,  # Encourage exploration
                'vf_coef': vf_coef,
                'verbose': verbose,
                'tensorboard_log': self.log_dir,
                'device': device
            }
        else:  # SAC
            self.model_config = {
                'policy': 'MlpPolicy',
                'env': self.env,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'gamma': gamma,
                'ent_coef': ent_coef,  # Encourage exploration
                'verbose': verbose,
                'tensorboard_log': self.log_dir,
                'device': device,
                'buffer_size': 1000000,  # SAC-specific
                'learning_starts': 100,  # SAC-specific
                'train_freq': 1,  # SAC-specific
                'gradient_steps': 1  # SAC-specific
            }
        
        # Initialize model
        self.model = None
        self.metrics_tracker = None
        self.training_start_time = None
        
        print(f"🤖 SearchRLAgent initialized ({algorithm.upper()}):")
        print(f"   Log directory: {log_dir}")
        print(f"   Model save path: {model_save_path}")
        print(f"   Learning rate: {learning_rate}")
        if algorithm == 'ppo':
            print(f"   Steps per rollout: {n_steps}")
        print(f"   Batch size: {batch_size}")
    
    def create_model(self, seed: Optional[int] = None):
        """Create RL model with configured hyperparameters"""
        
        if seed is not None:
            set_random_seed(seed)
        
        print(f"🏗️  Creating {self.algorithm.upper()} model...")
        
        if self.algorithm == 'ppo':
            self.model = PPO(**self.model_config)
        elif self.algorithm == 'sac':
            self.model = SAC(**self.model_config)
        
        # Initialize metrics tracker
        # self.metrics_tracker = ComprehensiveMetricsTracker(
        #     log_dir=self.log_dir,
        #     window_size=100
        # )
        
        print(f"✅ {self.algorithm.upper()} model created successfully")
        return self.model
    
    def train(self, 
              total_timesteps: int = 500000,
              eval_freq: int = 10000,
              n_eval_episodes: int = 10,
              save_freq: int = 50000,
              checkpoint_freq: int = 100000,
              seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the RL agent
        
        Args:
            total_timesteps: Total training timesteps
            eval_freq: Frequency of evaluation episodes
            n_eval_episodes: Number of evaluation episodes
            save_freq: Frequency of model saves
            checkpoint_freq: Frequency of checkpoint saves
            seed: Random seed for reproducibility
            
        Returns:
            Training results and statistics
        """
        
        if self.model is None:
            self.create_model(seed=seed)
        
        print(f"🚀 Starting {self.algorithm.upper()} training...")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Evaluation every: {eval_freq:,} steps")
        print(f"   Checkpoints every: {checkpoint_freq:,} steps")
        
        self.training_start_time = time.time()
        
        # Setup callbacks
        callbacks = self._setup_callbacks(
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            save_freq=save_freq,
            checkpoint_freq=checkpoint_freq,
            total_timesteps=total_timesteps
        )
        
        # Train the model
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name=self.run_id,
                reset_num_timesteps=False,  # Continue from previous training if model was loaded
                progress_bar=True  # Disable SB3 progress bar since we have our own
            )
            
            # Explicitly close tensorboard writer to flush events
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'close'):
                self.model.logger.close()
            
            training_time = time.time() - self.training_start_time
            print(f"✅ Training completed in {training_time:.1f} seconds!")

            plot_results(["."], total_timesteps, results_plotter.X_TIMESTEPS, f"{self.algorithm.upper()} {self.curriculum_type.title()}")

            
            # Generate training metrics visualization
            self._generate_training_plots()
            
            return {
                'training_time': training_time,
                'total_timesteps': total_timesteps,
                'run_id': self.run_id,
                'log_dir': self.log_dir,
                'model_path': self.model_save_path
            }
            
        except KeyboardInterrupt:
            print("\n⏸️  Training interrupted by user")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
        
        finally:
            # Ensure progress bars are closed
            for callback in callbacks:
                if hasattr(callback, 'pbar') and callback.pbar is not None:
                    callback.pbar.close()
            
            # Save final model
            self.save_model()
            
            # Generate training report
            # training_results = self._generate_training_report()
            
        return training_results
    
    def _setup_callbacks(self, eval_freq, n_eval_episodes, save_freq, checkpoint_freq, total_timesteps):
        """Setup training callbacks for evaluation and saving"""
        
        callbacks = []
        
        # Progress callback
        if TQDM_AVAILABLE:
            progress_callback = ProgressCallback(total_timesteps=total_timesteps, algorithm=self.algorithm, verbose=self.verbose)
            callbacks.append(progress_callback)
        
        # Evaluation callback (disabled - run evaluation separately)
        # if eval_freq > 0:
        #     eval_callback = EvalCallback(
        #         eval_env=self.env,
        #         best_model_save_path=f"{self.log_dir}/best_model/",
        #         log_path=f"{self.log_dir}/evaluations/",
        #         eval_freq=eval_freq,
        #         n_eval_episodes=n_eval_episodes,
        #         deterministic=True,
        #         render=False,
        #         verbose=self.verbose
        #     )
        #     callbacks.append(eval_callback)
        
        # Checkpoint callback
        if checkpoint_freq > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=f"{self.log_dir}/checkpoints/",
                name_prefix=f"search_{self.algorithm}_checkpoint",
                verbose=self.verbose
            )
            callbacks.append(checkpoint_callback)
        
        return callbacks
    
    def evaluate(self, 
                 n_episodes: int = 100,
                 deterministic: bool = True,
                 render: bool = False) -> Dict[str, float]:
        """
        Evaluate trained model performance
        
        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy
            render: Render episodes during evaluation
            
        Returns:
            Evaluation metrics
        """
        
        if self.model is None:
            raise ValueError("No model available for evaluation. Train or load a model first.")
        
        print(f"📊 Evaluating model over {n_episodes} episodes...")
        
        eval_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_count': 0,
            'timeout_count': 0,
            'stuck_count': 0,
            'search_times': [],
            'joint_violations': [],
        }
        
        # Create progress bar for evaluation
        episode_iterator = range(n_episodes)
        if TQDM_AVAILABLE:
            episode_iterator = tqdm(
                episode_iterator, 
                desc="📊 Evaluating", 
                unit="episodes",
                ncols=80
            )
        
        for episode in episode_iterator:
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_start_time = time.time()
            
            while True:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    self.env.render()
                
                if terminated or truncated:
                    search_time = time.time() - episode_start_time
                    outcome = info.get('outcome', 'unknown')
                    
                    # Record metrics
                    eval_metrics['episode_rewards'].append(episode_reward)
                    eval_metrics['episode_lengths'].append(episode_length)
                    eval_metrics['search_times'].append(search_time)
                    eval_metrics['joint_violations'].append(info.get('joint_violations', 0))
                    
                    if outcome == 'success':
                        eval_metrics['success_count'] += 1
                    elif outcome == 'timeout':
                        eval_metrics['timeout_count'] += 1
                    elif outcome == 'stuck':
                        eval_metrics['stuck_count'] += 1
                    
                    # Update progress bar with metrics
                    if TQDM_AVAILABLE and hasattr(episode_iterator, 'set_postfix'):
                        success_rate = eval_metrics['success_count'] / (episode + 1)
                        avg_reward = np.mean(eval_metrics['episode_rewards'])
                        episode_iterator.set_postfix({
                            'success': f'{success_rate:.3f}',
                            'reward': f'{avg_reward:.1f}'
                        })
                    
                    # Print progress every 20 episodes (fallback)
                    elif (episode + 1) % 20 == 0:
                        success_rate = eval_metrics['success_count'] / (episode + 1)
                        avg_reward = np.mean(eval_metrics['episode_rewards'])
                        print(f"   Episodes {episode+1:3d}: success_rate={success_rate:.3f}, avg_reward={avg_reward:.1f}")
                    
                    break
        
        # Compute final statistics
        results = {
            'success_rate': eval_metrics['success_count'] / n_episodes,
            'mean_reward': np.mean(eval_metrics['episode_rewards']),
            'std_reward': np.std(eval_metrics['episode_rewards']),
            'mean_episode_length': np.mean(eval_metrics['episode_lengths']),
            'mean_search_time': np.mean(eval_metrics['search_times']),
            'timeout_rate': eval_metrics['timeout_count'] / n_episodes,
            'stuck_rate': eval_metrics['stuck_count'] / n_episodes,
            'avg_joint_violations': np.mean(eval_metrics['joint_violations'])
        }
        
        print(f"\n📈 Evaluation Results:")
        print(f"   Success rate: {results['success_rate']:.3f}")
        print(f"   Mean reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
        print(f"   Mean search time: {results['mean_search_time']:.1f}s")
        print(f"   Joint violations: {results['avg_joint_violations']:.2f}/episode")
        
        return results
    
    def save_model(self, path: Optional[str] = None):
        """Save trained model"""
        if self.model is None:
            print("⚠️  No model to save")
            return
            
        save_path = path or self.model_save_path
        self.model.save(save_path)
        print(f"💾 Model saved: {save_path}")
    
    def load_model(self, path: Optional[str] = None):
        """Load pre-trained model"""
        load_path = path or self.model_save_path
        
        if not os.path.exists(f"{load_path}.zip"):
            raise FileNotFoundError(f"Model not found: {load_path}.zip")
        
        print(f"📁 Loading model: {load_path}")
        self.model = PPO.load(load_path, env=self.env)
        print("✅ Model loaded successfully")
    
    def _generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        
        if self.metrics_tracker is None:
            return {"status": "no_metrics"}
        
        training_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        report = {
            'training_time_seconds': training_time,
            'training_time_formatted': f"{training_time/3600:.1f}h" if training_time > 3600 else f"{training_time/60:.1f}m",
            'final_model_path': self.model_save_path,
            'log_directory': self.log_dir,
        }
        
        # Add metrics if available
        if hasattr(self.metrics_tracker, 'get_performance_summary'):
            performance_summary = self.metrics_tracker.get_performance_summary()
            report.update(performance_summary)
        
        # Save report
        if self.metrics_tracker:
            self.metrics_tracker.save_summary_report()
        
        print(f"\n📄 Training report generated")
        print(f"   Training time: {report['training_time_formatted']}")
        
        return report
    
    def _calculate_ema(self, values, kappa):
        """
        Calculate Exponential Moving Average using the formula:
        EMA_t = κ·SCORE_t + (1-κ)·EMA_{t-1}
        
        Args:
            values: Array of values (e.g., episode rewards)
            kappa: Smoothing factor (0 < kappa <= 1)
                  - Higher kappa = more responsive to recent changes
                  - Lower kappa = smoother, less responsive
        
        Returns:
            Array of EMA values
        """
        if len(values) == 0:
            return []
        
        ema = np.zeros_like(values, dtype=float)
        ema[0] = values[0]  # Initialize with first value
        
        for i in range(1, len(values)):
            ema[i] = kappa * values[i] + (1 - kappa) * ema[i-1]
        
        return ema
    
    def _generate_training_plots(self):
        """Generate training metrics plots using official stable-baselines3 plot_results"""
        print("📈 Generating training metrics plots...")
        
        plots_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            # Check if we have monitor data
            df = load_results(self.monitor_dir)
            df.to_csv(os.path.join(plots_dir, "monitor_logs.csv"))
            
            if len(df) > 0:
                print(f"   📊 Found {len(df)} completed episodes in Monitor logs")
                print(f"   📏 Mean reward: {df['r'].mean():.1f}, Mean length: {df['l'].mean():.1f}")
                
                # Store in metrics tracker for compatibility
                if self.metrics_tracker:
                    self.metrics_tracker.episode_rewards = df['r'].tolist()
                    self.metrics_tracker.episode_lengths = df['l'].tolist()
                
                # Use official plot_results function
                timesteps = int(df['t'].iloc[-1]) if len(df) > 0 else 10000
                
                # Plot 1: Rewards vs Timesteps
                plt.figure(figsize=(12, 6))
                plt.plot(df['t'].tolist(), df['r'].tolist(), 'b-', linewidth=2)
                plt.title(f'Training Progress: Rewards vs Timesteps - {self.run_id}', fontsize=14, fontweight='bold')
                plt.savefig(os.path.join(plots_dir, 'rewards_vs_timesteps.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # # Plot 2: Rewards vs Episodes
                # plt.figure(figsize=(12, 6))
                # plot_results(["./monitor_logs"], len(df), results_plotter.X_EPISODES, f"PPO Search Training - {self.run_id}")
                # plt.title(f'Training Progress: Rewards vs Episodes - {self.run_id}', fontsize=14, fontweight='bold')
                # plt.savefig(os.path.join(plots_dir, 'rewards_vs_episodes.png'), dpi=300, bbox_inches='tight')
                # plt.close()
                
                # Additional reward analysis plot with EMA
                plt.figure(figsize=(12, 8))
                
                # Plot raw rewards
                episodes = range(1, len(df) + 1)
                rewards = df['r'].values
                plt.plot(episodes, rewards, alpha=0.4, color='lightblue', linewidth=0.8, label='Raw Rewards')
                
                if len(df) > 10:
                    # Simple Moving Average
                    window_size = min(50, len(df) // 10)
                    moving_avg = df['r'].rolling(window=window_size, center=True).mean()
                    plt.plot(episodes, moving_avg, color='orange', linewidth=2, label=f'SMA ({window_size} episodes)')
                    
                    # Exponential Moving Average (EMA)
                    kappa_values = [0.1, 0.05, 0.01]  # Different smoothing factors
                    colors = ['red', 'green', 'purple']
                    
                    for kappa, color in zip(kappa_values, colors):
                        ema = self._calculate_ema(rewards, kappa)
                        alpha_val = 1/(1/kappa) if kappa > 0 else len(rewards)  # Equivalent window size
                        plt.plot(episodes, ema, color=color, linewidth=2, 
                               label=f'EMA (κ={kappa}, ~{alpha_val:.0f} eps)')
                
                plt.title(f'Reward Analysis: SMA vs EMA - {self.run_id}', fontsize=14, fontweight='bold')
                plt.ylabel('Reward', fontsize=12)
                plt.xlabel('Episode', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plots_dir, 'rewards_with_ema.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Dedicated EMA comparison plot
                if len(df) > 50:  # Only create if we have enough data
                    plt.figure(figsize=(12, 6))
                    plt.plot(episodes, rewards, alpha=0.3, color='gray', linewidth=0.5, label='Raw Rewards')
                    
                    # Multiple EMA with different smoothing factors
                    ema_configs = [
                        (0.01, 'blue', 'Very Smooth'),
                        (0.05, 'green', 'Smooth'), 
                        (0.1, 'red', 'Responsive'),
                        (0.2, 'orange', 'Very Responsive')
                    ]
                    
                    for kappa, color, desc in ema_configs:
                        ema = self._calculate_ema(rewards, kappa)
                        plt.plot(episodes, ema, color=color, linewidth=2, 
                               label=f'{desc} EMA (κ={kappa})')
                    
                    plt.title(f'EMA Comparison - Different Smoothing Factors - {self.run_id}', fontsize=14, fontweight='bold')
                    plt.ylabel('Reward', fontsize=12)
                    plt.xlabel('Episode', fontsize=12)
                    plt.legend(fontsize=10, loc='best')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(plots_dir, 'ema_comparison.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                
                print(f"✅ Training plots saved to: {plots_dir}")
                print(f"   - rewards_vs_timesteps.png")
                print(f"   - rewards_with_ema.png (SMA vs EMA comparison)")
                if len(df) > 50:
                    print(f"   - ema_comparison.png (Multiple EMA smoothing factors)")
                
            else:
                print("⚠️  No completed episodes found in Monitor logs")
                print("   This might happen with very short training runs")
                self._create_basic_summary_plot(plots_dir)
                return
            
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")
            print(f"   Make sure training ran long enough to complete at least one episode")
            import traceback
            traceback.print_exc()
            # Fallback to basic summary
            self._create_basic_summary_plot(plots_dir)
    
    def _create_basic_summary_plot(self, plots_dir: str):
        """Create a basic summary plot when no episode data is available"""
        
        plt.figure(figsize=(10, 6))
        
        # Create a simple text summary
        plt.text(0.1, 0.8, f'Training Summary - {self.run_id}', 
                fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
        
        plt.text(0.1, 0.6, 'No episode metrics were collected during training.', 
                fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, 'This can happen with very short training runs.', 
                fontsize=12, transform=plt.gca().transAxes)
        
        plt.text(0.1, 0.3, f'Curriculum: {self.curriculum_type}', 
                fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.2, f'Training completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                fontsize=10, style='italic', transform=plt.gca().transAxes)
        
        plt.axis('off')
        plt.savefig(os.path.join(plots_dir, 'training_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Basic training summary saved to: {plots_dir}")
        print(f"   - training_summary.png")


def create_vectorized_env(env_class, env_kwargs: Dict, n_envs: int = 4, seed: int = 0):
    """
    Create vectorized environment for parallel training
    
    Args:
        env_class: Environment class to instantiate
        env_kwargs: Environment initialization arguments
        n_envs: Number of parallel environments
        seed: Random seed
        
    Returns:
        Vectorized environment
    """
    
    def make_env(rank: int):
        def _init():
            env = env_class(**env_kwargs)
            env = Monitor(env)  # Wrap with Monitor for logging
            env.reset(seed=seed + rank)
            return env
        set_random_seed(seed)
        return _init
    
    # Create parallel environments
    env_fns = [make_env(i) for i in range(n_envs)]
    
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    else:
        return SubprocVecEnv(env_fns)
