# Scripts Directory

This directory contains training and evaluation scripts for the RL search system.

## Available Scripts:

### Training and Evaluation
- ✅ `train_search_rl.py` - Main PPO training script
- ✅ `evaluate_search.py` - Model evaluation and comparison script  
- ✅ `test_search_env.py` - Environment validation tests
- ✅ `visual_test_search_env.py` - Visual testing with GUI
- ✅ `test_movement_dynamics.py` - Movement validation tests
- ✅ `test_target_placement.py` - Target placement validation

### Future Scripts (Phase 3):
- `compare_policies.py` - Compare RL vs baseline policies
- `monitor_training.py` - Real-time training monitoring

## Usage:

### Install RL Dependencies:
```bash
cd av_so101
pip install -r requirements_rl.txt
```

### Train PPO Model:
```bash
python scripts/train_search_rl.py --timesteps 500000 --n-envs 4
```

### Evaluate Trained Model:
```bash
python scripts/evaluate_search.py --model-path models/search_ppo_model --compare-baseline
```

### Visual Testing:
```bash
python scripts/visual_test_search_env.py
```

## Training Options:

### Basic Training:
```bash
python scripts/train_search_rl.py
```

### Advanced Training with Curriculum:
```bash
python scripts/train_search_rl.py \
    --timesteps 1000000 \
    --n-envs 8 \
    --curriculum \
    --learning-rate 3e-4 \
    --log-dir my_training_logs
```

### Continue Training from Checkpoint:
```bash
python scripts/train_search_rl.py \
    --load-model models/search_ppo_model \
    --timesteps 200000
```