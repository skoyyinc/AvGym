#!/usr/bin/env python3
"""
Evaluation script for HybridTrackingPolicy

This script evaluates the hybrid policy that automatically switches
between RL search and visual servoing based on target visibility.
"""

import sys
import os
import argparse
import time
import numpy as np
import threading
import queue
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def keyboard_listener(key_queue):
    """Background thread to listen for keyboard input"""
    try:
        while True:
            key = input()
            if key.lower() == 's':
                key_queue.put('success')
            elif key.lower() == 'q':
                key_queue.put('quit')
    except:
        pass  # Handle any input errors gracefully

from src.hybrid_tracking_policy import HybridTrackingPolicy
from src.so_arm_gym_env import SO101CameraTrackingEnv


def save_evaluation_csv(args, curriculum_config, episode_rewards, episode_lengths, 
                       total_episode_times, search_times, centering_times,
                       mode_switches, visual_ratios, search_ratios):
    """Save evaluation metrics to CSV files"""
    
    # Create output directory
    os.makedirs(args.csv_output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    curriculum_name = args.curriculum_type
    algorithm = args.algorithm.upper()
    
    print(f"\n💾 Saving CSV files to {args.csv_output_dir}...")
    
    # Create comprehensive DataFrame with all metrics per episode
    data = {
        'episode': list(range(1, len(episode_rewards) + 1)),
        'reward': episode_rewards,
        'episode_length': episode_lengths,
        'total_time': total_episode_times,
        'search_time': search_times,  # None if not found
        'centering_time': centering_times,  # None if not centered
        'mode_switches': mode_switches,
        'visual_ratio': visual_ratios,
        'search_ratio': search_ratios
    }
    
    df = pd.DataFrame(data)
    
    # Main metrics file
    main_csv = f"{args.csv_prefix}_{algorithm}_{curriculum_name}_{timestamp}.csv"
    main_path = Path(args.csv_output_dir) / main_csv
    df.to_csv(main_path, index=False)
    print(f"   ✅ Main metrics: {main_path}")
    
    # Create separate files for plotting compatibility
    
    # 1. Rewards over episodes (compatible with training plot generator)
    rewards_df = pd.DataFrame({
        'r': episode_rewards,
        'l': episode_lengths,
        't': total_episode_times
    })
    rewards_csv = f"{args.csv_prefix}_{algorithm}_{curriculum_name}_rewards_{timestamp}.csv"
    rewards_path = Path(args.csv_output_dir) / rewards_csv
    rewards_df.to_csv(rewards_path, index=False)
    print(f"   ✅ Rewards (training plot compatible): {rewards_path}")
    
    # 2. Search times over episodes (only successful searches)
    search_df_data = []
    for i, (ep, search_time) in enumerate(zip(range(1, len(episode_rewards) + 1), search_times)):
        if search_time is not None:  # Only include episodes where target was found
            search_df_data.append({
                'episode': ep,
                'search_time': search_time
            })
    
    if search_df_data:
        search_df = pd.DataFrame(search_df_data)
        search_csv = f"{args.csv_prefix}_{algorithm}_{curriculum_name}_search_times_{timestamp}.csv"
        search_path = Path(args.csv_output_dir) / search_csv
        search_df.to_csv(search_path, index=False)
        print(f"   ✅ Search times: {search_path}")
    
    # 3. Centering times over episodes (only successful centering)
    centering_df_data = []
    for i, (ep, centering_time) in enumerate(zip(range(1, len(episode_rewards) + 1), centering_times)):
        if centering_time is not None:  # Only include episodes where target was centered
            centering_df_data.append({
                'episode': ep,
                'centering_time': centering_time
            })
    
    if centering_df_data:
        centering_df = pd.DataFrame(centering_df_data)
        centering_csv = f"{args.csv_prefix}_{algorithm}_{curriculum_name}_centering_times_{timestamp}.csv"
        centering_path = Path(args.csv_output_dir) / centering_csv
        centering_df.to_csv(centering_path, index=False)
        print(f"   ✅ Centering times: {centering_path}")
    
    print(f"   📊 Saved evaluation data for {len(episode_rewards)} episodes")
    print(f"   🔍 Algorithm: {algorithm}, Curriculum: {curriculum_name}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate HybridTrackingPolicy')
    
    parser.add_argument('--rl-model', type=str, default='models/search_ppo_model.zip',
                       help='Path to trained RL search model')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac'],
                       help='RL algorithm used in the model (default: ppo)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes during evaluation')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic RL policy (default: True)')
    
    # Curriculum selection
    parser.add_argument('--curriculum-type', type=str, default='static_cube', 
                       choices=['static_cube', 'random_cube', 'moving_target', 'multi_object', 'occlusion'],
                       help='Type of curriculum environment (default: static_cube)')
    
    # CSV output options
    parser.add_argument('--save-csv', action='store_true',
                       help='Save evaluation metrics to CSV files')
    parser.add_argument('--csv-output-dir', type=str, default='evaluation_results',
                       help='Directory to save CSV files (default: evaluation_results)')
    parser.add_argument('--csv-prefix', type=str, default='hybrid_eval',
                       help='Prefix for CSV filenames (default: hybrid_eval)')
    
    return parser.parse_args()


def get_curriculum_config(curriculum_type: str) -> dict:
    """Get configuration for different curriculum types"""
    
    curriculums = {
        'static_cube': {
            'name': 'Static Cube Curriculum',
            'description': 'Single red cube at fixed position - simplest scenario',
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
            'description': 'Single red cube with randomized position each episode - small deviations',
            'static_target': True,
            'target_position': None,
            'max_search_steps': 350,
            'distractors': False,
            'occlusion': False,
            'moving_target': False,
            'randomize_position': True,
            'position_deviation': 0.15
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


def evaluate_hybrid_policy(args):
    """Evaluate the hybrid tracking policy"""
    
    print("🔀 Evaluating Hybrid Tracking Policy")
    print("=" * 50)
    
    # Get curriculum configuration
    curriculum_config = get_curriculum_config(args.curriculum_type)
    print(f"📚 Curriculum: {curriculum_config['name']}")
    print(f"   {curriculum_config['description']}")
    print("=" * 50)
    
    # Create environment with curriculum configuration
    render_mode = "human" if args.render else "rgb_array"
    from src.rl import SearchRLEnv
    base_env = SO101CameraTrackingEnv(render_mode=render_mode)
    
    # Apply curriculum-specific configurations
    if curriculum_config.get('distractors', False):
        print("   ⚠️  Distractor objects not yet implemented")
    if curriculum_config.get('occlusion', False):
        print("   ✅ Occlusion curriculum now implemented!")
        print(f"   📋 Using: {curriculum_config.get('num_distractors', 2)} blue distractors between robot and target")
        print(f"   📏 Gap: {curriculum_config.get('distractor_gap', 0.08):.2f}m minimum separation")
    if curriculum_config.get('moving_target', False):
        print("   ⚠️  Moving target not yet implemented")
    
    # Wrap in SearchRLEnv with curriculum configuration
    env = SearchRLEnv(
        base_env=base_env,
        static_target=curriculum_config['static_target'],
        max_search_steps=curriculum_config['max_search_steps'],
        target_position=curriculum_config['target_position'],
        min_visual_servoing_steps=100,  # 10 seconds at 10Hz
        centering_threshold=0.1,  # Target must be centered within 0.1 distance
        training_mode=False,  # Evaluation mode: full hybrid behavior (search + visual servoing)
        randomize_position=curriculum_config.get('randomize_position', False),
        position_deviation=curriculum_config.get('position_deviation', 0.15),
        # Occlusion curriculum parameters
        distractors=curriculum_config.get('distractors', False),
        num_distractors=curriculum_config.get('num_distractors', 2),
        distractor_gap=curriculum_config.get('distractor_gap', 0.08),
        occlusion_placement=curriculum_config.get('occlusion_placement', 'between_robot_target')
    )
    
    # Create hybrid policy
    visual_config = {
        'p_gain_fast': 1.2,
        'p_gain_slow': 0.5, 
        'max_velocity': 1.3
    }
    
    try:
        hybrid_policy = HybridTrackingPolicy(
            visual_config=visual_config,
            rl_model_path=args.rl_model,
            rl_algorithm=args.algorithm,  # Pass algorithm type for SAC support
            target_lost_threshold=5,
            target_found_threshold=2
        )
        print("✅ Hybrid policy created successfully")
    except Exception as e:
        print(f"❌ Failed to create hybrid policy: {e}")
        return
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    search_times = []  # Time until target found
    total_episode_times = []  # Total time per episode
    centering_times = []  # Time until properly centered (target_found + visual_servoing)
    mode_switches = []
    visual_ratios = []
    search_ratios = []
    
    # Start keyboard listener for manual control
    key_queue = queue.Queue()
    keyboard_thread = threading.Thread(target=keyboard_listener, args=(key_queue,), daemon=True)
    keyboard_thread.start()
    
    print("📋 Manual Controls:")
    print("   Press 's' + Enter: Mark episode as successful")
    print("   Press 'q' + Enter: Quit evaluation")
    print("   Episode will auto-terminate when target is well-centered\n")
    
    # Run evaluation episodes
    for episode in range(args.episodes):
        print(f"\n🎬 Episode {episode + 1}/{args.episodes}")
        
        # Reset environment and policy
        obs, info = env.reset()
        hybrid_policy.reset()
        
        episode_reward = 0
        episode_length = 0
        done = False
        manual_success = False
        target_found = False
        target_found_time = None
        target_centered = False
        target_centered_time = None
        
        episode_start_time = time.time()
        
        while not done and episode_length < args.max_steps:
            # Get hybrid observation (both RL state and base obs)
            rl_state, base_obs = env.get_hybrid_observation()
            
            # Pass both observations to hybrid policy
            hybrid_obs = {
                'base_obs': base_obs,
                'rl_state': rl_state
            }
            action = hybrid_policy.predict(hybrid_obs, deterministic=args.deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            # Track time until target found (same as baseline script)
            if not target_found and base_obs['target_in_view'][0]:
                target_found = True
                target_found_time = time.time() - episode_start_time
                print(f"   🎯 TARGET FOUND at step {episode_length}! Search time: {target_found_time:.2f}s")
            
            # Track time until target properly centered
            if not target_centered and info.get('target_centered', False):
                target_centered = True
                target_centered_time = time.time() - episode_start_time
                print(f"   🎯 TARGET CENTERED at step {episode_length}! Total centering time: {target_centered_time:.2f}s")
            
            # Check for manual input
            try:
                key_input = key_queue.get_nowait()
                if key_input == 'success':
                    print("✅ Manual success marked!")
                    manual_success = True
                    done = True
                elif key_input == 'quit':
                    print("🚫 Evaluation stopped by user")
                    return
            except queue.Empty:
                pass
            
            # Render if requested
            if args.render:
                env.render()
                time.sleep(0.1)  # 10 Hz
        
        episode_time = time.time() - episode_start_time
        
        # Get performance stats
        stats = hybrid_policy.get_performance_stats()
        
        # Determine success based on proper centering OR manual marking
        auto_success = info.get('target_centered', False) and info.get('visual_servoing_steps', 0) >= 100
        success = auto_success or manual_success
        if success:
            success_count += 1
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        total_episode_times.append(episode_time)
        
        # Store search time (None if target not found)
        search_times.append(target_found_time if target_found else None)
        
        # Store centering time (None if target not centered)
        centering_times.append(target_centered_time if target_centered else None)
        
        mode_switches.append(stats['mode_switches'])
        visual_ratios.append(stats['visual_servoing_ratio'])
        search_ratios.append(stats['search_ratio'])
        
        # Print episode results
        print(f"   Reward: {episode_reward:.1f}, Steps: {episode_length}")
        if target_found:
            print(f"   Search time: {target_found_time:.2f}s")
        else:
            print(f"   Search time: Target not found")
        print(f"   Mode switches: {stats['mode_switches']}")
        print(f"   Visual/Search ratio: {stats['visual_servoing_ratio']:.2f}/{stats['search_ratio']:.2f}")
        print(f"   Final mode: {stats['current_mode']}")
        print(f"   Target found: {info.get('target_found', False)}")
        print(f"   Visual servoing steps: {info.get('visual_servoing_steps', 0)}")
        print(f"   Target centered: {info.get('target_centered', False)} (distance: {info.get('center_distance', -1):.3f})")
        if manual_success:
            print(f"   Success: ✅ (Manual)")
        elif auto_success:
            print(f"   Success: ✅ (Auto)")
        else:
            print(f"   Success: ❌")
    
    # Clean up
    env.close()
    
    # Compute final statistics
    success_rate = success_count / args.episodes
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_switches = np.mean(mode_switches)
    avg_visual_ratio = np.mean(visual_ratios)
    avg_search_ratio = np.mean(search_ratios)
    
    # Search time statistics
    target_found_rate = len(search_times) / args.episodes
    if search_times:
        avg_search_time = np.mean(search_times)
        min_search_time = np.min(search_times)
        max_search_time = np.max(search_times)
        std_search_time = np.std(search_times)
    else:
        avg_search_time = min_search_time = max_search_time = std_search_time = float('inf')
    
    print("\n" + "=" * 60)
    print("📊 HYBRID POLICY EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Success rate (centering): {success_rate:.3f} ({success_count}/{args.episodes})")
    print(f"Target found rate: {target_found_rate:.3f} ({len(search_times)}/{args.episodes})")
    
    if search_times:
        print(f"\n⏱️  Search Time Statistics (episodes where target found):")
        print(f"   Mean search time: {avg_search_time:.2f}s ± {std_search_time:.2f}s")
        print(f"   Min search time: {min_search_time:.2f}s")
        print(f"   Max search time: {max_search_time:.2f}s")
        
        print(f"\n📋 Individual Search Times:")
        episode_num = 1
        for i, result in enumerate(search_times):
            print(f"   Episode {episode_num}: {result:.2f}s")
            episode_num += 1
    else:
        print(f"\n⏱️  No targets found in any episode")
    
    print(f"\n📈 General Performance:")
    print(f"   Average reward: {avg_reward:.2f} ± {np.std(episode_rewards):.2f}")
    print(f"   Average episode length: {avg_length:.1f} ± {np.std(episode_lengths):.1f}")
    print(f"   Average mode switches: {avg_switches:.1f} ± {np.std(mode_switches):.1f}")
    print(f"   Average visual ratio: {avg_visual_ratio:.3f}")
    print(f"   Average search ratio: {avg_search_ratio:.3f}")
    
    # Detailed statistics
    print(f"\n📊 Ranges:")
    print(f"   Reward range: {np.min(episode_rewards):.1f} to {np.max(episode_rewards):.1f}")
    print(f"   Length range: {np.min(episode_lengths)} to {np.max(episode_lengths)} steps")
    print(f"   Mode switch range: {np.min(mode_switches)} to {np.max(mode_switches)}")
    
    # Save CSV files if requested
    if args.save_csv:
        save_evaluation_csv(args, curriculum_config, episode_rewards, episode_lengths, 
                           total_episode_times, search_times, centering_times, 
                           mode_switches, visual_ratios, search_ratios)
    
    return {
        'success_rate': success_rate,
        'target_found_rate': target_found_rate,
        'avg_search_time': avg_search_time,
        'min_search_time': min_search_time,
        'max_search_time': max_search_time,
        'std_search_time': std_search_time,
        'search_times': search_times,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'avg_switches': avg_switches,
        'avg_visual_ratio': avg_visual_ratio,
        'avg_search_ratio': avg_search_ratio,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    # Check if RL model exists
    if not os.path.exists(args.rl_model):
        print(f"❌ RL model not found: {args.rl_model}")
        print("   Please train a model first using: python scripts/train_search_rl.py")
        return
    
    # Run evaluation
    results = evaluate_hybrid_policy(args)
    
    print(f"\n🎯 Hybrid policy evaluation completed!")
    print(f"   Success rate: {results['success_rate']:.3f}")
    print(f"   Average reward: {results['avg_reward']:.1f}")


if __name__ == "__main__":
    main()