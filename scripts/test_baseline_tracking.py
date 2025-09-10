#!/usr/bin/env python3
"""
Baseline Test Script for Classic Tracking Policy

This script evaluates the baseline performance of the classic visual servoing
tracking policy (ImprovedTrackingPolicy) with random sweep search behavior.
This provides a baseline to compare against RL-based search policies.
"""

import sys
import os
import argparse
import time
import numpy as np
from collections import deque
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tracking_policy import ImprovedTrackingPolicy
from src.so_arm_gym_env import SO101CameraTrackingEnv
from src.rl import SearchRLEnv


def save_baseline_csv(args, curriculum_config, episode_rewards, episode_lengths, 
                     total_episode_times, search_times, centering_times):
    """Save baseline evaluation metrics to CSV files"""
    
    # Create output directory
    os.makedirs(args.csv_output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    curriculum_name = args.curriculum_type
    algorithm = "BASELINE"  # Baseline visual servoing
    
    print(f"\n💾 Saving CSV files to {args.csv_output_dir}...")
    
    # Create comprehensive DataFrame with all metrics per episode
    data = {
        'episode': list(range(1, len(episode_rewards) + 1)),
        'reward': episode_rewards,
        'episode_length': episode_lengths,
        'total_time': total_episode_times,
        'search_time': search_times,  # None if not found
        'centering_time': centering_times,  # None if not centered
        'mode_switches': [0] * len(episode_rewards),  # Baseline has no mode switches
        'visual_ratio': [1.0] * len(episode_rewards),  # Always visual servoing
        'search_ratio': [0.0] * len(episode_rewards)   # No RL search
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
    
    print(f"   📊 Saved baseline evaluation data for {len(episode_rewards)} episodes")
    print(f"   🔍 Algorithm: {algorithm}, Curriculum: {curriculum_name}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Baseline Classic Tracking Policy')
    
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of test episodes (default: 50)')
    parser.add_argument('--max-time', type=float, default=30.0,
                       help='Maximum episode time in seconds (default: 30.0)')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes during testing')
    parser.add_argument('--success-threshold', type=float, default=0.2,
                       help='Distance threshold for success (default: 0.2)')
    parser.add_argument('--success-steps', type=int, default=5,
                       help='Number of steps target must be centered for success (default: 5)')
    
    # Curriculum selection
    parser.add_argument('--curriculum-type', type=str, default='static_cube', 
                       choices=['static_cube', 'random_cube', 'moving_target', 'multi_object', 'occlusion'],
                       help='Type of curriculum environment (default: static_cube)')
    
    # CSV output options
    parser.add_argument('--save-csv', action='store_true',
                       help='Save evaluation metrics to CSV files')
    parser.add_argument('--csv-output-dir', type=str, default='evaluation_results',
                       help='Directory to save CSV files (default: evaluation_results)')
    parser.add_argument('--csv-prefix', type=str, default='baseline_eval',
                       help='Prefix for CSV filenames (default: baseline_eval)')
    
    return parser.parse_args()


def get_curriculum_config(curriculum_type: str) -> dict:
    """Get configuration for different curriculum types"""
    
    curriculums = {
        'static_cube': {
            'name': 'Static Cube Curriculum',
            'description': 'Single red cube at fixed position - baseline scenario',
            'static_target': True,
            'target_position': None,  # Will use default static position
            'max_search_steps': 300,
            'randomize_position': False,
            'distractors': False,
            'occlusion': False,
            'moving_target': False
        },
        
        'random_cube': {
            'name': 'Random Cube Curriculum', 
            'description': 'Single red cube with randomized position each episode',
            'static_target': True,
            'target_position': None,  # Will be randomized
            'max_search_steps': 350,
            'randomize_position': True,
            'position_deviation': 0.15,
            'distractors': False,
            'occlusion': False,
            'moving_target': False
        },
        
        'moving_target': {
            'name': 'Moving Target Curriculum',
            'description': 'Target cube moves slowly during episodes - not implemented',
            'static_target': False,
            'target_position': None,
            'max_search_steps': 400,
            'randomize_position': False,
            'distractors': False,
            'occlusion': False,
            'moving_target': True
        },
        
        'multi_object': {
            'name': 'Multi-Object Curriculum',
            'description': 'Multiple objects with distractors - not implemented',
            'static_target': True,
            'target_position': None,
            'max_search_steps': 350,
            'randomize_position': False,
            'distractors': True,
            'occlusion': False,
            'moving_target': False
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


def run_episode(env, policy, max_time_steps, success_threshold, success_steps):
    """
    Run a single test episode
    
    Args:
        env: Environment instance
        policy: Tracking policy instance
        max_time_steps: Maximum steps per episode
        success_threshold: Distance threshold for success
        success_steps: Consecutive steps needed for success
        
    Returns:
        dict: Episode results
    """
    obs, info = env.reset()
    
    episode_length = 0
    episode_reward = 0.0
    target_found = False
    target_found_time = None
    target_centered = False
    target_centered_time = None
    episode_start_time = time.time()
    
    target_distances = []
    target_found_steps = 0
    target_lost_steps = 0
    
    for step in range(max_time_steps):
        # Get base observation for tracking policy (SearchRLEnv returns RL state, not base obs)
        rl_state, base_obs = env.get_hybrid_observation()
        
        # Get action from tracking policy using base observation
        action = policy.predict(base_obs)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_length += 1
        episode_reward += reward
        
        # Track target visibility and distance from base observation
        target_in_view = base_obs['target_in_view'][0]
        center_distance = base_obs['target_center_distance'][0]
        
        # Track when target is first found (for search time comparison)
        if target_in_view and not target_found:
            target_found = True
            target_found_time = time.time() - episode_start_time
            print(f"      🎯 TARGET FOUND at step {step}! Search time: {target_found_time:.2f}s")
        
        # Track when target is properly centered (like hybrid evaluation)
        if target_in_view and center_distance <= success_threshold and not target_centered:
            target_centered = True
            target_centered_time = time.time() - episode_start_time
            print(f"      ✅ TARGET CENTERED at step {step}! Total centering time: {target_centered_time:.2f}s")
            break  # Episode complete once target is centered
        
        # Track statistics
        if target_in_view:
            target_found_steps += 1
            target_distances.append(center_distance)
        else:
            target_lost_steps += 1
        
        # Check for early termination
        if terminated or truncated:
            print(f"      Episode terminated: terminated={terminated}, truncated={truncated}")
            break
    
    episode_time = time.time() - episode_start_time
    
    # Calculate statistics
    visibility_ratio = target_found_steps / episode_length if episode_length > 0 else 0.0
    avg_distance = np.mean(target_distances) if target_distances else float('inf')
    min_distance = np.min(target_distances) if target_distances else float('inf')
    
    return {
        'success': target_centered,  # Success means target was centered
        'target_found': target_found,
        'time_to_find': target_found_time if target_found else None,
        'time_to_center': target_centered_time if target_centered else None,
        'episode_length': episode_length,
        'episode_time': episode_time,
        'episode_reward': episode_reward,
        'visibility_ratio': visibility_ratio,
        'target_found_steps': target_found_steps,
        'target_lost_steps': target_lost_steps,
        'avg_distance': avg_distance,
        'min_distance': min_distance,
        'final_distance': center_distance,
        'target_distances': target_distances
    }


def test_baseline_tracking(args):
    """Test baseline tracking policy"""
    
    print("🎯 Testing Baseline Classic Tracking Policy")
    print("=" * 60)
    
    # Get curriculum configuration
    curriculum_config = get_curriculum_config(args.curriculum_type)
    print(f"📚 Curriculum: {curriculum_config['name']}")
    print(f"   {curriculum_config['description']}")
    print("=" * 60)
    
    # Check for unimplemented features
    if curriculum_config.get('moving_target', False):
        print("   ⚠️  Moving target not yet implemented")
    if curriculum_config.get('occlusion', False):
        print("   ✅ Occlusion curriculum now implemented!")
        print(f"   📋 Using: {curriculum_config.get('num_distractors', 2)} blue distractors between robot and target")
        print(f"   📏 Gap: {curriculum_config.get('distractor_gap', 0.08):.2f}m minimum separation")
    
    # Create environment with curriculum configuration (same as evaluation script)
    render_mode = "human" if args.render else "rgb_array"
    base_env = SO101CameraTrackingEnv(render_mode=render_mode)
    
    # Wrap in SearchRLEnv with curriculum configuration
    env = SearchRLEnv(
        base_env=base_env,
        static_target=curriculum_config['static_target'],
        max_search_steps=curriculum_config['max_search_steps'],
        target_position=curriculum_config['target_position'],
        min_visual_servoing_steps=0,  # No minimum for baseline test
        centering_threshold=args.success_threshold,  # Use success threshold
        training_mode=False,  # Evaluation mode: don't terminate early
        randomize_position=curriculum_config.get('randomize_position', False),
        position_deviation=curriculum_config.get('position_deviation', 0.15),
        # Occlusion curriculum parameters
        distractors=curriculum_config.get('distractors', False),
        num_distractors=curriculum_config.get('num_distractors', 2),
        distractor_gap=curriculum_config.get('distractor_gap', 0.08),
        occlusion_placement=curriculum_config.get('occlusion_placement', 'between_robot_target')
    )
    
    # Apply curriculum-specific configurations
    if curriculum_config.get('randomize_position', False):
        print(f"   🎲 Position randomization: ±{curriculum_config['position_deviation']:.2f}m deviation")
    
    # Create tracking policy with classic configuration
    tracking_config = {
        'p_gain_fast': 1.2,
        'p_gain_slow': 0.5,
        'max_velocity': 1.0
    }
    
    policy = ImprovedTrackingPolicy(tracking_config)
    
    print(f"\n📋 Test Configuration:")
    print(f"   Episodes: {args.episodes}")
    print(f"   Max time per episode: {args.max_time:.1f}s")
    print(f"   Success threshold: {args.success_threshold:.2f}")
    print(f"   Required success steps: {args.success_steps}")
    print(f"   Control frequency: 10 Hz")
    
    # Calculate max steps based on time limit
    control_freq = 10  # Hz
    max_time_steps = int(args.max_time * control_freq)
    
    # Run test episodes
    results = []
    success_count = 0
    episode_rewards = []
    episode_lengths = []
    total_episode_times = []
    search_times = []  # Time to find target (None if not found)
    centering_times = []  # Time to center target (None if not centered)
    visibility_ratios = []
    min_distances = []
    
    print(f"\n🚀 Starting {args.episodes} test episodes...")
    
    for episode in range(args.episodes):
        print(f"\n🎬 Episode {episode + 1}/{args.episodes}")
        
        # Run episode
        episode_result = run_episode(
            env, policy, max_time_steps, 
            args.success_threshold, args.success_steps
        )
        
        results.append(episode_result)
        
        # Update statistics
        if episode_result['success']:
            success_count += 1
            
        # Collect metrics for CSV (compatible with evaluation plot generator)
        episode_rewards.append(episode_result['episode_reward'])
        episode_lengths.append(episode_result['episode_length'])
        total_episode_times.append(episode_result['episode_time'])
        search_times.append(episode_result['time_to_find'])  # None if not found
        centering_times.append(episode_result['time_to_center'])  # None if not centered
        visibility_ratios.append(episode_result['visibility_ratio'])
        min_distances.append(episode_result['min_distance'])
        
        # Print episode results focused on search time
        status = "✅ FOUND" if episode_result['success'] else "❌ NOT FOUND"
        print(f"   {status}")
        if episode_result['success']:
            print(f"   Search time: {episode_result['time_to_find']:.2f}s")
        else:
            print(f"   Total time: {episode_result['episode_time']:.1f}s (timed out)")
        print(f"   Steps: {episode_result['episode_length']}")
        print(f"   Target visibility: {episode_result['visibility_ratio']:.2f}")
        
        # Render if requested
        if args.render:
            env.render()
            time.sleep(0.1)
    
    # Clean up
    env.close()
    
    # Calculate final statistics
    success_rate = success_count / args.episodes
    
    # Search time statistics (only successful episodes)
    valid_search_times = [t for t in search_times if t is not None]
    if valid_search_times:
        mean_search_time = np.mean(valid_search_times)
        min_search_time = np.min(valid_search_times)
        max_search_time = np.max(valid_search_times)
        std_search_time = np.std(valid_search_times)
    else:
        mean_search_time = min_search_time = max_search_time = std_search_time = float('inf')
    
    mean_visibility = np.mean(visibility_ratios)
    
    # Print final results
    print("\n" + "=" * 60)
    print("📊 BASELINE SEARCH PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"Curriculum: {curriculum_config['name']}")
    print(f"Episodes tested: {args.episodes}")
    print(f"Max time per episode: {args.max_time:.1f}s")
    
    print(f"\n🎯 Search Success Metrics:")
    print(f"   Success rate: {success_rate:.3f} ({success_count}/{args.episodes})")
    
    if valid_search_times:
        print(f"\n⏱️  Search Time Statistics (successful episodes only):")
        print(f"   Mean time to find: {mean_search_time:.2f}s ± {std_search_time:.2f}s")
        print(f"   Min time to find: {min_search_time:.2f}s")
        print(f"   Max time to find: {max_search_time:.2f}s")
        
        # Individual search times for each successful episode
        print(f"\n📋 Individual Search Times:")
        for i, result in enumerate(results):
            if result['success'] and result['time_to_find'] is not None:
                print(f"   Episode {i+1}: {result['time_to_find']:.2f}s")
    else:
        print(f"\n⏱️  No successful searches - all episodes timed out")
    
    print(f"\n📈 General Performance:")
    print(f"   Mean target visibility ratio: {mean_visibility:.3f}")
    
    # Failed episodes summary
    failed_count = args.episodes - success_count
    if failed_count > 0:
        print(f"\n❌ Failed Episodes: {failed_count}")
        print(f"   All failed episodes reached {args.max_time:.1f}s timeout")
    
    # Save CSV files if requested
    if args.save_csv:
        save_baseline_csv(args, curriculum_config, episode_rewards, episode_lengths, 
                         total_episode_times, search_times, centering_times)
    
    return {
        'success_rate': success_rate,
        'mean_search_time': mean_search_time,
        'min_search_time': min_search_time,
        'max_search_time': max_search_time,
        'std_search_time': std_search_time,
        'search_times': search_times,
        'mean_visibility': mean_visibility,
        'curriculum_type': args.curriculum_type,
        'episodes_tested': args.episodes,
        'all_results': results
    }


def main():
    """Main function"""
    args = parse_arguments()
    
    print("🔍 Classic Tracking Policy Baseline Test")
    print("=" * 60)
    print("This test evaluates the baseline performance of the classic visual")
    print("servoing policy with random sweep search when target is lost.")
    print("Results provide a baseline for comparison with RL policies.")
    
    try:
        # Run baseline test
        results = test_baseline_tracking(args)
        
        print(f"\n🏁 Baseline search test completed!")
        print(f"   Success rate: {results['success_rate']:.3f}")
        if results['search_times']:
            print(f"   Mean search time: {results['mean_search_time']:.2f}s")
        else:
            print(f"   No successful searches")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())