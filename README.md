# AvGym - Active Vision Gymnasium

A comprehensive robotics simulation environment for the SO-ARM101 robot focused on active vision tasks, object tracking, and reinforcement learning-based search strategies.

## 🎯 Overview

AvGym implements a hybrid approach combining **Reinforcement Learning (RL) search** with **Visual Servoing** for autonomous object tracking. The system intelligently switches between RL-based search behavior (when target is lost) and precise visual servoing control (when target is visible).



https://github.com/user-attachments/assets/cf3c8689-3261-44fe-91e4-149aecd7acc9

With occlusion

https://github.com/user-attachments/assets/2fbbe2ab-2f3d-49ea-b8ae-260b878ecd08




### Key Features

- **Hybrid Control System**: Automatic switching between RL search and visual servoing
- **Multiple RL Algorithms**: Support for PPO and SAC training
- **Comprehensive Curricula**: Static, random, moving target, and occlusion scenarios  
- **Physics Simulation**: High-fidelity PyBullet-based SO-ARM101 robot simulation
- **Performance Analysis**: Detailed metrics tracking and visualization tools
- **Baseline Comparisons**: Classical visual servoing baseline for performance evaluation

## 🏗️ Architecture

```
AvGym/
├── src/
│   ├── active_vision_system.py    # Main active vision orchestrator
│   ├── hybrid_tracking_policy.py  # Hybrid RL+visual servoing policy
│   ├── tracking_policy.py         # Classical visual servoing policy
│   ├── so_arm_gym_env.py          # SO-ARM101 Gymnasium environment
│   ├── rl/
│   │   ├── search_env.py          # RL search environment wrapper
│   │   └── search_agent.py        # PPO/SAC training agent
│   └── ...
├── scripts/
│   ├── train_search_rl.py         # RL training script
│   ├── evaluate_hybrid_policy.py  # Hybrid policy evaluation
│   ├── test_baseline_tracking.py  # Baseline performance testing
│   └── generate_*.py              # Plot generation utilities
├── models/                        # Pre-trained models
├── urdf/                          # Robot model files

```

## 🔧 Installation

### Prerequisites


### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/skoyyinc/AvGym.git
cd AvGym
```

2. **Create conda environment**:
```bash
conda create -n avgym python=3.8
conda activate avgym
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
conda install -c conda-forge pybullet
```
⚠️ **Critical**: Install PyBullet via conda-forge LAST as it requires NumPy 1.x for building. Installing other packages first may cause compatibility issues.




## 🚀 Quick Start

### 1. Basic Demo
```bash
# Object tracking demonstration
python ./scripts/evaluate_hybrid_policy.py --rl-model models/ppo_final.zip --render


### 2. Train RL Search Policy

**PPO Training** (recommended):
```bash
python scripts/train_search_rl.py \
    --algorithm ppo \
    --curriculum-type static_cube \
    --timesteps 500000 \
    --save-freq 50000
```

**SAC Training**:
```bash
python scripts/train_search_rl.py \
    --algorithm sac \
    --curriculum-type static_cube \
    --timesteps 500000 \
    --save-freq 50000
```

**Advanced Curricula**:
```bash
# Random cube positions
python scripts/train_search_rl.py --curriculum-type random_cube

# Occlusion challenges with distractor objects
python scripts/train_search_rl.py --curriculum-type occlusion
```

### 3. Evaluate Hybrid Policy

**Using Pre-trained Models**:
```bash
# Evaluate PPO hybrid policy
python scripts/evaluate_hybrid_policy.py \
    --rl-model models/search_ppo_model.zip \
    --algorithm ppo \
    --curriculum-type static_cube \
    --episodes 50 \
    --save-csv

# Evaluate SAC hybrid policy  
python scripts/evaluate_hybrid_policy.py \
    --rl-model models/search_sac_model.zip \
    --algorithm sac \
    --curriculum-type occlusion \
    --episodes 50 \
    --save-csv
```

### 4. Baseline Performance Testing

```bash
# Test classical visual servoing baseline
python scripts/test_baseline_tracking.py \
    --curriculum-type static_cube \
    --episodes 50 \
    --save-csv
```



## 📊 Training Parameters

### PPO Hyperparameters
- **Learning Rate**: 3e-4
- **Batch Size**: 64  
- **N Steps**: 2048 (rollout size)
- **N Epochs**: 10
- **Gamma**: 0.99 (discount factor)
- **GAE Lambda**: 0.95
- **Clip Range**: 0.2
- **Entropy Coefficient**: 0.01

### SAC Hyperparameters  
- **Learning Rate**: 3e-4
- **Batch Size**: 64
- **Gamma**: 0.99
- **Buffer Size**: 1,000,000
- **Learning Starts**: 100
- **Train Frequency**: 1
- **Gradient Steps**: 1
- **Entropy Coefficient**: 0.01



## 📈 Performance Metrics

The system tracks comprehensive metrics:

- **Search Time**: Time until target is first found
- **Centering Time**: Total time until target is properly centered  
- **Success Rate**: Fraction of successful episodes
- **Mode Switches**: Transitions between RL search and visual servoing
- **Visual Servoing Ratio**: Time spent in visual servoing mode
- **Episode Rewards**: Cumulative reward per episode





## 🛠️ Advanced Usage

### Custom Training Configuration

```bash
python scripts/train_search_rl.py \
    --algorithm ppo \
    --curriculum-type random_cube \
    --timesteps 1000000 \
    --learning-rate 1e-4 \
    --batch-size 128 \
    --n-steps 4096 \
    --device cuda \
    --eval-freq 25000
```



### Custom Evaluation

```bash
python scripts/evaluate_hybrid_policy.py \
    --rl-model models/search_ppo_model.zip \
    --curriculum-type occlusion \
    --episodes 100 \
    --max-steps 2000 \
    --render \
    --save-csv \
    --csv-prefix custom_eval
```



## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔧 Troubleshooting

### Common Issues

**PyBullet Installation**: 
- Always install via `conda install -c conda-forge pybullet`
- Install PyBullet LAST to avoid NumPy conflicts

**CUDA Issues**:
- Use `--device cpu` if CUDA is unavailable
- Install PyTorch with CUDA support if needed

**Environment Setup**:
- Ensure all URDF files are properly configured

**Training Issues**:
- Start with `static_cube` curriculum for initial testing
- Reduce `--timesteps` if training takes too long
- Monitor logs in `search_rl_logs/` directory

For more issues, please check the [Issues](https://github.com/your-username/AvGym/issues) page.

---

**Happy Robot Learning! 🤖✨**
