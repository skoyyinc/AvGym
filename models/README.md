# Models Directory

This directory will contain saved RL models for the search optimization system.

## Structure:
- `search_ppo_model.zip` - Trained PPO model for search behavior
- `checkpoints/` - Training checkpoints and intermediate models
- `best_models/` - Best performing models during training

## Usage:
Models are saved/loaded using Stable-Baselines3 format:
```python
from stable_baselines3 import PPO
model = PPO.load("models/search_ppo_model.zip")
```