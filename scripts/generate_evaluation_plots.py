#!/usr/bin/env python3
"""
Evaluation Plot Generation Script

This script generates evaluation performance plots from saved evaluation CSV files.
Supports individual algorithm plots and comparison plots between different algorithms/curricula.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate evaluation plots from CSV files')
    
    parser.add_argument('--csv-files', type=str, nargs='+', required=True,
                       help='Path(s) to evaluation CSV files')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='Labels for each CSV file (default: extracted from filename)')
    parser.add_argument('--output-dir', type=str, default='evaluation_plots',
                       help='Output directory for plots (default: evaluation_plots)')
    parser.add_argument('--window-size', type=int, default=10,
                       help='Moving average window size (default: 10)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Plot resolution in DPI (default: 300)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 8],
                       help='Figure size width height in inches (default: 12 8)')
    
    return parser.parse_args()


def load_evaluation_data(csv_file: str) -> pd.DataFrame:
    """
    Load and validate evaluation CSV data
    
    Args:
        csv_file: Path to evaluation CSV file
        
    Returns:
        DataFrame with evaluation data
    """
    df = pd.read_csv(csv_file)
    
    # Check required columns for main evaluation file
    required_cols = ['episode', 'reward', 'episode_length', 'total_time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"📊 Loaded {len(df)} episodes from {csv_file}")
    print(f"   Mean reward: {df['reward'].mean():.2f}, Mean length: {df['episode_length'].mean():.1f}")
    
    return df


def calculate_moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Calculate moving average with padding"""
    if len(data) < window_size:
        return data
    
    # Use pandas rolling for better handling
    series = pd.Series(data)
    return series.rolling(window=window_size, center=True, min_periods=1).mean().values


def extract_algorithm_from_path(csv_file: str) -> str:
    """Extract algorithm name from file path"""
    path_lower = csv_file.lower()
    if 'ppo' in path_lower:
        return 'PPO'
    elif 'sac' in path_lower:
        return 'SAC'
    else:
        return Path(csv_file).stem


def generate_individual_plots(df: pd.DataFrame, label: str, output_dir: str, 
                            window_size: int, figsize: List[int], dpi: int):
    """Generate individual evaluation plots for one algorithm"""
    
    print(f"🎨 Generating individual evaluation plots for {label}...")
    
    episodes = df['episode'].values
    rewards = df['reward'].values
    episode_lengths = df['episode_length'].values
    total_times = df['total_time'].values
    
    print(f"   Data shape: {len(episodes)} episodes, rewards range [{rewards.min():.2f}, {rewards.max():.2f}]")
    
    # Determine colors based on algorithm
    if 'PPO' in label.upper():
        color = 'blue'
        dark_color = 'darkblue'
    elif 'SAC' in label.upper():
        color = 'green'
        dark_color = 'darkgreen'
    else:
        color = 'purple'
        dark_color = 'darkmagenta'

    # 1. Reward over episodes
    print(f"   Creating rewards plot...")
    plt.figure(figsize=figsize)
    
    # Raw rewards (transparent)
    plt.plot(episodes, rewards, alpha=0.3, color=color, linewidth=0.5, label='Raw Rewards')
    
    # Moving average with fill
    if len(rewards) > window_size:
        print(f"   Calculating moving average (window={window_size})...")
        ma_rewards = calculate_moving_average(rewards, window_size)
        
        # Plot moving average line
        plt.plot(episodes, ma_rewards, color=dark_color, linewidth=2, 
                label=f'Moving Average ({window_size} episodes)')
        
        # Fill between raw rewards and moving average
        plt.fill_between(episodes, rewards, ma_rewards, alpha=0.2, color=color)
    
    plt.title(f'{label} - Reward Over Episodes', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Episodes: {len(df)}\\nMean: {np.mean(rewards):.2f}\\nStd: {np.std(rewards):.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save rewards plot
    rewards_file = os.path.join(output_dir, f'{label}_rewards_over_episodes.png')
    print(f"   Saving rewards plot to {rewards_file}...")
    plt.savefig(rewards_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Search times over episodes (if available)
    if 'search_time' in df.columns:
        print(f"   Creating search times plot...")
        plt.figure(figsize=figsize)
        
        # Filter out None/NaN values for search times
        valid_search_mask = df['search_time'].notna()
        if valid_search_mask.sum() > 0:
            search_episodes = df[valid_search_mask]['episode'].values
            search_times = df[valid_search_mask]['search_time'].values
            
            plt.plot(search_episodes, search_times, 'o', alpha=0.6, color=color, 
                    markersize=4, label='Search Times')
            
            # Moving average for search times with fill
            if len(search_times) > window_size:
                ma_search_times = calculate_moving_average(search_times, min(window_size, len(search_times)//2))
                plt.plot(search_episodes, ma_search_times, color=dark_color, linewidth=2, 
                        label=f'Moving Average ({window_size} episodes)')
                
                # Fill between raw search times and moving average
                plt.fill_between(search_episodes, search_times, ma_search_times, alpha=0.2, color=color)
            
            plt.title(f'{label} - Time Until Found Over Episodes', fontsize=16, fontweight='bold')
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Search Time (seconds)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_search_time = np.mean(search_times)
            std_search_time = np.std(search_times)
            stats_text = f'Found: {len(search_times)}/{len(df)} episodes\\nMean: {mean_search_time:.2f}s\\nStd: {std_search_time:.2f}s'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save search times plot
            search_file = os.path.join(output_dir, f'{label}_search_times_over_episodes.png')
            print(f"   Saving search times plot to {search_file}...")
            plt.savefig(search_file, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            print(f"   ⚠️ No valid search times found, skipping search times plot")
    
    # 3. Centering times over episodes (if available)
    if 'centering_time' in df.columns:
        print(f"   Creating centering times plot...")
        plt.figure(figsize=figsize)
        
        # Filter out None/NaN values for centering times
        valid_centering_mask = df['centering_time'].notna()
        if valid_centering_mask.sum() > 0:
            centering_episodes = df[valid_centering_mask]['episode'].values
            centering_times = df[valid_centering_mask]['centering_time'].values
            
            plt.plot(centering_episodes, centering_times, 'o', alpha=0.6, color=color, 
                    markersize=4, label='Centering Times')
            
            # Moving average for centering times with fill
            if len(centering_times) > window_size:
                ma_centering_times = calculate_moving_average(centering_times, min(window_size, len(centering_times)//2))
                plt.plot(centering_episodes, ma_centering_times, color=dark_color, linewidth=2, 
                        label=f'Moving Average ({window_size} episodes)')
                
                # Fill between raw centering times and moving average
                plt.fill_between(centering_episodes, centering_times, ma_centering_times, alpha=0.2, color=color)
            
            plt.title(f'{label} - Total Time Until Centered Over Episodes', fontsize=16, fontweight='bold')
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Centering Time (seconds)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_centering_time = np.mean(centering_times)
            std_centering_time = np.std(centering_times)
            stats_text = f'Centered: {len(centering_times)}/{len(df)} episodes\\nMean: {mean_centering_time:.2f}s\\nStd: {std_centering_time:.2f}s'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save centering times plot
            centering_file = os.path.join(output_dir, f'{label}_centering_times_over_episodes.png')
            print(f"   Saving centering times plot to {centering_file}...")
            plt.savefig(centering_file, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            print(f"   ⚠️ No valid centering times found, skipping centering times plot")
    
    print(f"✅ Generated individual plots for {label}")


def generate_comparison_plots(data_dict: Dict[str, pd.DataFrame], output_dir: str,
                            window_size: int, figsize: List[int], dpi: int):
    """Generate comparison plots between algorithms"""
    
    if len(data_dict) < 2:
        print("⚠️  Need at least 2 datasets for comparison plots")
        return
    
    print(f"🎨 Generating comparison plots for {len(data_dict)} algorithms...")
    
    # Color mapping
    colors = {
        'PPO': 'blue',
        'SAC': 'green', 
        'DQN': 'red',
        'A2C': 'orange',
        'TD3': 'purple'
    }
    
    # 1. Reward comparison
    plt.figure(figsize=figsize)
    
    for label, df in data_dict.items():
        episodes = df['episode'].values
        rewards = df['reward'].values
        
        # Determine color
        color = colors.get(label.upper(), 'gray')
        for key in colors.keys():
            if key in label.upper():
                color = colors[key]
                break
        
        # Plot moving average
        if len(rewards) > window_size:
            ma_rewards = calculate_moving_average(rewards, window_size)
            plt.plot(episodes, ma_rewards, color=color, linewidth=2.5, label=f'{label}')
    
    plt.title('Evaluation Comparison - Rewards Over Episodes', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_rewards.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Search times comparison (if available)
    search_data_available = all('search_time' in df.columns for df in data_dict.values())
    if search_data_available:
        plt.figure(figsize=figsize)
        
        for label, df in data_dict.items():
            # Filter valid search times
            valid_mask = df['search_time'].notna()
            if valid_mask.sum() > 0:
                search_episodes = df[valid_mask]['episode'].values
                search_times = df[valid_mask]['search_time'].values
                
                # Determine color
                color = colors.get(label.upper(), 'gray')
                for key in colors.keys():
                    if key in label.upper():
                        color = colors[key]
                        break
                
                # Plot moving average
                if len(search_times) > window_size//2:
                    ma_search_times = calculate_moving_average(search_times, min(window_size//2, len(search_times)//2))
                    plt.plot(search_episodes, ma_search_times, color=color, linewidth=2.5, label=f'{label}')
        
        plt.title('Evaluation Comparison - Search Times Over Episodes', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Search Time (seconds)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_search_times.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
    
    # 3. Performance statistics comparison
    plt.figure(figsize=(10, 6))
    
    algorithms = list(data_dict.keys())
    means = [data_dict[alg]['reward'].mean() for alg in algorithms]
    stds = [data_dict[alg]['reward'].std() for alg in algorithms]
    
    x_pos = np.arange(len(algorithms))
    colors_list = [colors.get(alg.upper(), 'gray') for alg in algorithms]
    
    bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors_list)
    
    plt.title('Performance Comparison - Mean Rewards', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Mean Reward ± Std', fontsize=12)
    plt.xticks(x_pos, algorithms)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 4. Mean search time comparison (if search time data available)
    search_time_data_available = all('search_time' in df.columns for df in data_dict.values())
    if search_time_data_available:
        plt.figure(figsize=(10, 6))
        
        algorithms = list(data_dict.keys())
        search_means = []
        search_stds = []
        
        for alg in algorithms:
            # Filter valid search times (not None/NaN)
            valid_search_times = data_dict[alg]['search_time'].dropna()
            if len(valid_search_times) > 0:
                search_means.append(valid_search_times.mean())
                search_stds.append(valid_search_times.std())
            else:
                search_means.append(float('inf'))  # No successful searches
                search_stds.append(0)
        
        x_pos = np.arange(len(algorithms))
        colors_list = [colors.get(alg.upper(), 'gray') for alg in algorithms]
        
        # Only plot algorithms that have valid search times
        valid_indices = [i for i, mean in enumerate(search_means) if not np.isinf(mean)]
        if valid_indices:
            valid_algorithms = [algorithms[i] for i in valid_indices]
            valid_means = [search_means[i] for i in valid_indices]
            valid_stds = [search_stds[i] for i in valid_indices]
            valid_colors = [colors_list[i] for i in valid_indices]
            valid_x_pos = np.arange(len(valid_algorithms))
            
            bars = plt.bar(valid_x_pos, valid_means, yerr=valid_stds, capsize=5, alpha=0.7, color=valid_colors)
            
            plt.title('Performance Comparison - Mean Search Times', fontsize=16, fontweight='bold')
            plt.xlabel('Algorithm', fontsize=12)
            plt.ylabel('Mean Search Time ± Std (seconds)', fontsize=12)
            plt.xticks(valid_x_pos, valid_algorithms)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, valid_means, valid_stds):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                        f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'search_time_comparison.png'), dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            print("   ⚠️ No valid search times found for any algorithm, skipping search time comparison")
    
    print(f"✅ Generated comparison plots for {len(data_dict)} algorithms")


def main():
    """Main function"""
    args = parse_arguments()
    
    print("📊 Evaluation Plot Generation Script")
    print("=" * 50)
    
    # Test matplotlib setup
    print("🔧 Testing matplotlib setup...")
    plt.figure(figsize=(2, 2))
    plt.plot([1, 2], [1, 2])
    plt.close()
    print("✅ Matplotlib working correctly")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    # Load data
    data_dict = {}
    
    for i, csv_file in enumerate(args.csv_files):
        print(f"\\n📂 Processing file {i+1}/{len(args.csv_files)}: {csv_file}")
        
        if not os.path.exists(csv_file):
            print(f"❌ File not found: {csv_file}")
            continue
            
        df = load_evaluation_data(csv_file)
        if df is None:
            print(f"⚠️  Skipping {csv_file} due to loading error")
            continue
            
        # Determine label
        if args.labels and i < len(args.labels):
            label = args.labels[i]
        else:
            label = extract_algorithm_from_path(csv_file)
            
        print(f"🏷️  Using label: {label}")
        data_dict[label] = df
        
        # Generate individual plots
        generate_individual_plots(df, label, args.output_dir, 
                                args.window_size, args.figsize, args.dpi)
    
    # Generate comparison plots if multiple datasets
    if len(data_dict) > 1:
        generate_comparison_plots(data_dict, args.output_dir, 
                                args.window_size, args.figsize, args.dpi)
    
    # Summary
    print("\\n" + "=" * 50)
    print("📈 Evaluation Plot Generation Complete!")
    print(f"   Processed {len(data_dict)} datasets")
    print(f"   Output directory: {args.output_dir}")
    
    # List generated files
    plot_files = [f for f in os.listdir(args.output_dir) if f.endswith('.png')]
    print(f"   Generated {len(plot_files)} plot files:")
    for plot_file in sorted(plot_files):
        print(f"     - {plot_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())