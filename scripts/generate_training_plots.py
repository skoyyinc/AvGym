#!/usr/bin/env python3
"""
Training Plot Generation Script

This script generates training performance plots from saved monitor CSV files.
Supports individual algorithm plots and comparison plots between PPO and SAC.
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
    parser = argparse.ArgumentParser(description='Generate training plots from monitor CSV files')
    
    parser.add_argument('--csv-files', type=str, nargs='+', required=True,
                       help='Path(s) to monitor CSV files')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='Labels for each CSV file (default: extracted from filename)')
    parser.add_argument('--output-dir', type=str, default='training_plots',
                       help='Output directory for plots (default: training_plots)')
    parser.add_argument('--window-size', type=int, default=50,
                       help='Moving average window size (default: 50)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Plot resolution in DPI (default: 300)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 8],
                       help='Figure size width height in inches (default: 12 8)')
    
    return parser.parse_args()


def load_monitor_data(csv_file: str) -> pd.DataFrame:
    """
    Load and validate monitor CSV data
    
    Args:
        csv_file: Path to monitor CSV file
        
    Returns:
        DataFrame with monitor data
    """
    try:
        df = pd.read_csv(csv_file)
        
        # Check required columns
        required_cols = ['r', 'l', 't']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"📊 Loaded {len(df)} episodes from {csv_file}")
        print(f"   Mean reward: {df['r'].mean():.2f}, Mean length: {df['l'].mean():.1f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading {csv_file}: {e}")
        return None


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
    """Generate individual training plots for one algorithm"""
    
    print(f"🎨 Generating individual plots for {label}...")
    
    episodes = np.arange(1, len(df) + 1)
    rewards = df['r'].values
    lengths = df['l'].values
    
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

    # 1. Reward over time with moving average
    print(f"   Creating rewards plot...")
    plt.figure(figsize=figsize)
    
    # Raw rewards (transparent)
    plt.plot(episodes, rewards, alpha=0.3, color=color, linewidth=0.5, label='Raw Rewards')
    
    # Moving average
    if len(rewards) > window_size:
        print(f"   Calculating moving average (window={window_size})...")
        ma_rewards = calculate_moving_average(rewards, window_size)
        plt.plot(episodes, ma_rewards, color=dark_color, linewidth=2, 
                label=f'Moving Average ({window_size} episodes)')
    
    
    plt.title(f'{label} - Reward Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Episodes: {len(df)}\nMean: {np.mean(rewards):.2f}\nStd: {np.std(rewards):.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save rewards plot
    rewards_file = os.path.join(output_dir, f'{label}_rewards_over_time.png')
    print(f"   Saving rewards plot to {rewards_file}...")
    plt.savefig(rewards_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Episode length over time
    print(f"   Creating episode lengths plot...")
    plt.figure(figsize=figsize)
    
    plt.plot(episodes, lengths, alpha=0.6, color=color, linewidth=0.8, label='Episode Length')
    
    if len(lengths) > window_size:
        ma_lengths = calculate_moving_average(lengths, window_size)
        plt.plot(episodes, ma_lengths, color=dark_color, linewidth=2, 
                label=f'Moving Average ({window_size} episodes)')
    
    plt.title(f'{label} - Episode Length Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Length (steps)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save lengths plot
    lengths_file = os.path.join(output_dir, f'{label}_episode_lengths.png')
    print(f"   Saving lengths plot to {lengths_file}...")
    plt.savefig(lengths_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated individual plots for {label}")


def generate_comparison_plots(data_dict: Dict[str, pd.DataFrame], output_dir: str,
                            window_size: int, figsize: List[int], dpi: int):
    """Generate comparison plots between algorithms"""
    
    if len(data_dict) < 2:
        print("⚠️  Need at least 2 datasets for comparison plots")
        return
    
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
        episodes = np.arange(1, len(df) + 1)
        rewards = df['r'].values
        
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
    
    plt.title('Training Comparison - Rewards Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_rewards.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Learning curves comparison (smoothed)
    plt.figure(figsize=figsize)
    
    for label, df in data_dict.items():
        episodes = np.arange(1, len(df) + 1)
        rewards = df['r'].values
        
        # Determine color
        color = colors.get(label.upper(), 'gray')
        for key in colors.keys():
            if key in label.upper():
                color = colors[key]
                break
        
        # Heavy smoothing for learning curve
        heavy_smooth = calculate_moving_average(rewards, min(100, len(rewards)//5))
        plt.plot(episodes, heavy_smooth, color=color, linewidth=3, label=label)
        
        # Fill area for variance
        if len(rewards) > 20:
            std_window = min(50, len(rewards)//4)
            rolling_std = pd.Series(rewards).rolling(window=std_window, center=True, min_periods=1).std()
            plt.fill_between(episodes, heavy_smooth - rolling_std, heavy_smooth + rolling_std, 
                           alpha=0.2, color=color)
    
    plt.title('Learning Curves Comparison (Heavily Smoothed)', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves_comparison.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 3. Performance statistics comparison
    plt.figure(figsize=(10, 6))
    
    algorithms = list(data_dict.keys())
    means = [data_dict[alg]['r'].mean() for alg in algorithms]
    stds = [data_dict[alg]['r'].std() for alg in algorithms]
    
    x_pos = np.arange(len(algorithms))
    colors_list = [colors.get(alg.upper(), 'gray') for alg in algorithms]
    
    bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors_list)
    
    plt.title('Final Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Mean Reward ± Std', fontsize=12)
    plt.xticks(x_pos, algorithms)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated comparison plots for {len(data_dict)} algorithms")


def main():
    """Main function"""
    args = parse_arguments()
    
    print("📊 Training Plot Generation Script")
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
        print(f"\n📂 Processing file {i+1}/{len(args.csv_files)}: {csv_file}")
        
        if not os.path.exists(csv_file):
            print(f"❌ File not found: {csv_file}")
            continue
            
        df = load_monitor_data(csv_file)
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
    print("\n" + "=" * 50)
    print("📈 Plot Generation Complete!")
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