#!/usr/bin/env python3
"""
Main training script for RL search optimization

This script trains a PPO agent to learn intelligent search behavior
when the target object is lost from view.
"""

import sys
import os
import argparse
import time
from pathlib import Path


# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl import SearchRLEnv, ComprehensiveMetricsTracker
from src.rl.search_agent import SearchRLAgent, create_vectorized_env
from src.so_arm_gym_env import SO101CameraTrackingEnv


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PPO agent for SO101 search behavior')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps (default: 500000)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='PPO learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Mini-batch size (default: 64)')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Steps per rollout (default: 2048)')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Optimization epochs per rollout (default: 10)')
    
    # Environment parameters
    parser.add_argument('--max-search-steps', type=int, default=300,
                       help='Maximum steps per search episode (default: 300)')
    parser.add_argument('--n-envs', type=int, default=1,
                       help='Number of parallel environments (default: 1)')
    parser.add_argument('--static-target', action='store_true', default=True,
                       help='Use static target position (default: True)')
    parser.add_argument('--target-position', type=float, nargs=3, default=None,
                       help='Custom target position [x, y, z] (default: auto-calculated)')
    
    # Evaluation and saving
    parser.add_argument('--eval-freq', type=int, default=10000,
                       help='Evaluation frequency (default: 10000)')
    parser.add_argument('--n-eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--save-freq', type=int, default=50000,
                       help='Model save frequency (default: 50000)')
    parser.add_argument('--checkpoint-freq', type=int, default=100000,
                       help='Checkpoint save frequency (default: 100000)')
    
    # Paths and logging
    parser.add_argument('--log-dir', type=str, default='search_rl_logs',
                       help='Directory for training logs (default: search_rl_logs)')
    parser.add_argument('--model-path', type=str, default='models/search_ppo_model',
                       help='Path to save trained model (default: models/search_ppo_model)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to pre-trained model to continue training')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during training (slower)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    parser.add_argument('--curriculum', action='store_true',
                       help='Use curriculum learning (progressive difficulty)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'auto'],
                       help='Device for training: cpu (recommended for MLP), cuda, or auto (default: cpu)')
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, default='ppo', 
                       choices=['ppo', 'sac'],
                       help='RL algorithm to use: ppo (default) or sac')
    
    # Curriculum selection
    parser.add_argument('--curriculum-type', type=str, default='static_cube', 
                       choices=['static_cube', 'random_cube', 'moving_target', 'multi_object', 'occlusion'],
                       help='Type of curriculum environment (default: static_cube)')
    
    return parser.parse_args()


def get_curriculum_config(curriculum_type: str) -> dict:
    """Get configuration for different curriculum types"""
    
    curriculums = {
        'static_cube': {
            'name': 'Static Cube Curriculum',
            'description': 'Single red cube at fixed position - simplest training scenario',
            'static_target': True,
            'target_position': None,  # Will use default static position
            'max_search_steps': 300,
            'distractors': False,
            'occlusion': False,
            'moving_target': False,
            'randomize_position': False
        },
        
        'random_cube': {
            'name': 'Random Cube Curriculum', 
            'description': 'Single red cube with randomized position each episode - small deviations from base position',
            'static_target': True,
            'target_position': None,  # Base position, will be randomized per episode
            'max_search_steps': 350,  # Slightly longer for varied difficulty
            'distractors': False,
            'occlusion': False,
            'moving_target': False,
            'randomize_position': True,
            'position_deviation': 0.15  # ±15cm deviation in X,Y and ±10cm in Z
        },
        
        'moving_target': {
            'name': 'Moving Target Curriculum',
            'description': 'Target cube moves slowly during episodes',
            'static_target': False,
            'target_position': None,
            'max_search_steps': 400,
            'distractors': False,
            'occlusion': False,
            'moving_target': True,
            'randomize_position': False
        },
        
        'multi_object': {
            'name': 'Multi-Object Curriculum',
            'description': 'Multiple objects with distractors (blue cubes)',
            'static_target': True,
            'target_position': None,
            'max_search_steps': 350,
            'distractors': True,
            'occlusion': False,
            'moving_target': False,
            'randomize_position': False
        },
        
        'occlusion': {
            'name': 'Occlusion Curriculum',
            'description': 'Red target cube with blue distractor cubes positioned between robot and target for occlusion',
            'static_target': True,
            'target_position': None,
            'max_search_steps': 500,
            'distractors': True,
            'occlusion': True,
            'moving_target': False,
            'randomize_position': True,
            'position_deviation': 0.15,
            'num_distractors': 2,  # Number of blue distractor cubes
            'distractor_gap': 0.08,  # Minimum gap between target and distractors
            'occlusion_placement': 'between_robot_target'  # Place distractors between robot and target
        }
    }
    
    if curriculum_type not in curriculums:
        raise ValueError(f"Unknown curriculum type: {curriculum_type}")
    
    return curriculums[curriculum_type]


def create_search_environment(max_search_steps=300, render_mode="rgb_array", static_target=True, target_position=None, curriculum_config=None):
    """Create search RL environment with curriculum configuration"""
    if curriculum_config:
        print(f"🏗️  Creating {curriculum_config['name']} environment...")
    else:
        print("🏗️  Creating search environment...")
    
    # Create base SO101 environment
    base_env = SO101CameraTrackingEnv(render_mode=render_mode)
    
    # Apply curriculum-specific configurations
    if curriculum_config:
        # Note: Future curriculum features can be added here
        # For now, we only support static_cube curriculum
        if curriculum_config.get('distractors', False):
            print("   ⚠️  Distractor objects not yet implemented")
        if curriculum_config.get('occlusion', False):
            print("   ✅ Occlusion curriculum now implemented!")
            print(f"   📋 Using: {curriculum_config.get('num_distractors', 2)} blue distractors between robot and target")
            print(f"   📏 Gap: {curriculum_config.get('distractor_gap', 0.08):.2f}m minimum separation")
        if curriculum_config.get('moving_target', False):
            print("   ⚠️  Moving target not yet implemented")
    
    # Extract randomization parameters from curriculum
    randomize_position = curriculum_config.get('randomize_position', False) if curriculum_config else False
    position_deviation = curriculum_config.get('position_deviation', 0.15) if curriculum_config else 0.15
    
    # Wrap in search RL environment
    search_env = SearchRLEnv(
        base_env=base_env,
        max_search_steps=max_search_steps,
        static_target=static_target,
        target_position=target_position,
        training_mode=True,  # Pure RL search training - terminate immediately when target found
        randomize_position=randomize_position,
        position_deviation=position_deviation,
        # Occlusion curriculum parameters
        distractors=curriculum_config.get('distractors', False),
        num_distractors=curriculum_config.get('num_distractors', 2),
        distractor_gap=curriculum_config.get('distractor_gap', 0.08),
        occlusion_placement=curriculum_config.get('occlusion_placement', 'between_robot_target')
    )
    
    print("✅ Search environment created")
    print("   🏋️  Training mode: terminate immediately when target found (pure search)")
    if curriculum_config:
        print(f"   📚 Using {curriculum_config['name']}")
        if randomize_position:
            print(f"   🎲 Position randomization: ±{position_deviation:.2f}m deviation")
    elif static_target:
        print("   📍 Using static target position")
    else:
        print("   🎲 Using dynamic target placement")
    
    return search_env


def setup_curriculum_learning(agent, total_timesteps):
    """
    Setup curriculum learning stages
    
    Progressively increases difficulty during training
    """
    print("📚 Setting up curriculum learning...")
    
    curriculum_stages = [
        {
            'name': 'Stage 1: Basic Search',
            'timesteps': int(total_timesteps * 0.3),
            'description': 'Simple scenarios, targets in easy positions',
            'config': {
                'max_search_steps': 200,
                'target_distance_range': (0.4, 0.6),
                'difficulty_multiplier': 0.5
            }
        },
        {
            'name': 'Stage 2: Intermediate Search', 
            'timesteps': int(total_timesteps * 0.4),
            'description': 'Medium difficulty, varied target positions',
            'config': {
                'max_search_steps': 250,
                'target_distance_range': (0.3, 0.7),
                'difficulty_multiplier': 0.75
            }
        },
        {
            'name': 'Stage 3: Advanced Search',
            'timesteps': int(total_timesteps * 0.3),
            'description': 'Full difficulty, challenging scenarios',
            'config': {
                'max_search_steps': 300,
                'target_distance_range': (0.3, 0.8),
                'difficulty_multiplier': 1.0
            }
        }
    ]
    
    for i, stage in enumerate(curriculum_stages):
        print(f"   {stage['name']}: {stage['timesteps']:,} timesteps")
        print(f"     {stage['description']}")
    
    return curriculum_stages


def train_with_curriculum(agent, curriculum_stages, args):
    """Train agent using curriculum learning"""
    print("🎓 Starting curriculum-based training...")
    
    total_trained = 0
    
    for stage_idx, stage in enumerate(curriculum_stages):
        print(f"\n📖 {stage['name']} ({stage['timesteps']:,} timesteps)")
        print(f"   {stage['description']}")
        
        # Update environment configuration if needed
        # (This would require modifying the environment to support dynamic config)
        
        # Train for this stage
        stage_results = agent.train(
            total_timesteps=stage['timesteps'],
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            save_freq=args.save_freq,
            checkpoint_freq=args.checkpoint_freq
        )
        
        total_trained += stage['timesteps']
        progress = total_trained / sum(s['timesteps'] for s in curriculum_stages)
        
        print(f"✅ {stage['name']} completed!")
        print(f"   Overall progress: {progress:.1%}")
        
        # Save stage checkpoint
        stage_model_path = f"{args.model_path}_stage_{stage_idx+1}"
        agent.save_model(stage_model_path)
        print(f"   Stage model saved: {stage_model_path}")
    
    print("\n🎓 Curriculum training completed!")


def main():
    """Main training function"""
    args = parse_arguments()
    
    print("🚀 SO101 Search RL Training")
    print("=" * 50)
    print(f"Training PPO agent for intelligent search behavior")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Log directory: {args.log_dir}")
    print(f"Model save path: {args.model_path}")
    
    # Get curriculum configuration
    curriculum_config = get_curriculum_config(args.curriculum_type)
    print(f"📚 Curriculum: {curriculum_config['name']}")
    print(f"   {curriculum_config['description']}")
    print(f"   Max search steps: {curriculum_config['max_search_steps']}")
    
    if args.curriculum:
        print(f"Using progressive curriculum learning: ✅")
    
    print("=" * 50)
    
    try:
        # Create environment(s)
        render_mode = "human" if args.render else "rgb_array"
        
        if args.n_envs > 1:
            print(f"🌐 Creating {args.n_envs} parallel environments...")
            
            env_kwargs = {
                'max_search_steps': curriculum_config['max_search_steps'],
                'static_target': curriculum_config['static_target'],
                'target_position': curriculum_config['target_position'] or args.target_position,
                'curriculum_config': curriculum_config
            }
            
            env = create_vectorized_env(
                env_class=lambda **kwargs: create_search_environment(
                    max_search_steps=kwargs['max_search_steps'],
                    render_mode="rgb_array",  # No rendering for parallel envs
                    static_target=kwargs['static_target'],
                    target_position=kwargs['target_position'],
                    curriculum_config=kwargs['curriculum_config']
                ),
                env_kwargs=env_kwargs,
                n_envs=args.n_envs,
                seed=args.seed
            )
            
        else:
            print("🌐 Creating single environment...")
            env = create_search_environment(
                max_search_steps=curriculum_config['max_search_steps'],
                render_mode=render_mode,
                static_target=curriculum_config['static_target'],
                target_position=curriculum_config['target_position'] or args.target_position,
                curriculum_config=curriculum_config
            )
        
        # Create RL agent
        print(f"🤖 Creating {args.algorithm.upper()} agent...")
        if args.device == 'cpu':
            print("   💻 Using CPU (recommended for MLP policy)")
        elif args.device == 'cuda':
            print("   🚀 Using GPU (may be slower for MLP policy)")
        
        agent = SearchRLAgent(
            env=env,
            algorithm=args.algorithm,
            log_dir=args.log_dir,
            model_save_path=args.model_path,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device=args.device,
            verbose=args.verbose,
            curriculum_type=args.curriculum_type
        )
        
        # Load pre-trained model if specified
        if args.load_model:
            print(f"📁 Loading pre-trained model: {args.load_model}")
            agent.load_model(args.load_model)
        else:
            # Create new model
            agent.create_model(seed=args.seed)
        
        # Training
        start_time = time.time()
        
        if args.curriculum:
            # Curriculum learning
            curriculum_stages = setup_curriculum_learning(agent, args.timesteps)
            train_with_curriculum(agent, curriculum_stages, args)
        else:
            # Standard training
            print("🏋️  Starting standard PPO training...")
            training_results = agent.train(
                total_timesteps=args.timesteps,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                save_freq=args.save_freq,
                checkpoint_freq=args.checkpoint_freq,
                seed=args.seed
            )
        
        training_time = time.time() - start_time
        
        # Final evaluation (disabled - run separately)
        # print("\n📊 Running final evaluation...")
        # eval_results = agent.evaluate(
        #     n_episodes=50,
        #     deterministic=True,
        #     render=False
        # )
        
        # Print final results
        print("\n" + "=" * 50)
        print("🏁 TRAINING COMPLETED!")
        print(f"⏱️  Total training time: {training_time/3600:.1f} hours")
        print(f"💾 Model saved: {agent.model_save_path}")
        print(f"📊 Run evaluation separately with:")
        print(f"   python scripts/evaluate_hybrid_policy.py --rl-model {agent.model_save_path}.zip --curriculum-type {args.curriculum_type}")
        print("=" * 50)
        
        # Save training completion info
        training_log_path = os.path.join(agent.log_dir, "training_completed.txt")
        with open(training_log_path, 'w') as f:
            f.write("Training Completion Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Curriculum type: {args.curriculum_type}\n")
            f.write(f"Run ID: {agent.run_id}\n")
            f.write(f"Total timesteps: {args.timesteps:,}\n")
            f.write(f"Training time: {training_time/3600:.2f} hours\n")
            f.write(f"Model saved: {agent.model_save_path}.zip\n")
            f.write(f"Log directory: {agent.log_dir}\n")
            f.write(f"Evaluation command: python scripts/evaluate_hybrid_policy.py --rl-model {agent.model_save_path}.zip --curriculum-type {args.curriculum_type}\n")
        
        print(f"📄 Training summary saved: {training_log_path}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if 'env' in locals():
                env.close()
        except:
            pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
