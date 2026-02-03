# Battleship RL Training

Train a reinforcement learning agent to play Battleship using DQN.

## Quick Start

```bash
# 1. Install dependencies
cd train
uv sync  # or: pip install -e .

# 2. Train a model
python main.py train

# 3. Export TorchScript model for submission
python main.py export --model models/battleship_model.zip --output model.pt
```

The `model.pt` TorchScript file is your competition submission.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Training

Train a new model with default settings:

```bash
python main.py train
```

Customize training parameters:

```bash
python main.py train \
    --total-timesteps 2000000 \
    --num-envs 8 \
    --output-dir my_models \
    --learning-rate 1e-4
```

Training outputs:
- `models/battleship_model.zip` - Stable Baselines3 checkpoint (intermediate)
- `models/battleship_model.pt` - TorchScript model (for submission)
- `models/checkpoints/` - Periodic checkpoints
- `models/best_model/` - Best model based on evaluation
- `models/tensorboard/` - Training logs

## Competition Submission

**Your submission must be a TorchScript `.pt` file.**

Export your trained model:

```bash
python main.py export --model models/battleship_model.zip --output model.pt
```

The training script also automatically exports a TorchScript model to `models/battleship_model.pt`.

### What is TorchScript?

TorchScript is a serialized PyTorch model format that:
- Is self-contained (no Python class definitions needed at load time)
- Can be loaded with just `torch.jit.load()`
- Works in the competition's inference container

### Testing Your Submission Locally

```bash
# Run the inference server with your model
python ../launch_battleship_rl.py --model model.pt --port 8001

# The server exposes:
#   GET  /health     - Health check
#   POST /board      - Get board placement
#   POST /next-move  - Get next shot coordinates
```

## Evaluation

Compare your trained model against a random baseline:

```bash
python main.py evaluate --model models/battleship_model.zip --episodes 50
```

Watch a single game:

```bash
python main.py play --model models/battleship_model.zip
```

## Architecture

### Environment (`env.py`)

A Gymnasium environment for single-player Battleship:
- **Observation**: 3-channel 10x10 tensor
  - Channel 0: Cells shot at (1.0) vs not (0.0)
  - Channel 1: Hits (1.0) vs misses (0.0)
  - Channel 2: Sunk ship cells (1.0) vs not (0.0)
- **Action**: Integer 0-99 (flattened board position)
- **Rewards**: Miss (-0.1), Hit (+1.0), Sink (+5.0), Win (+20.0)

### Model

DQN with custom CNN feature extractor:
- 3 convolutional layers (32 -> 64 -> 64 filters)
- 2 fully-connected layers (256 units each)
- Output: Q-values for all 100 board positions

## Files

| File               | Description                                  |
| ------------------ | -------------------------------------------- |
| `main.py`          | CLI entry point for train/evaluate/export    |
| `train.py`         | Training script with DQN and SubprocVecEnv   |
| `env.py`           | Gymnasium environment for Battleship         |
| `agent.py`         | RLShooter class for inference                |
| `state_encoder.py` | Converts game history to tensor observations |
| `test_model.py`    | Model evaluation and visualization           |

## Hyperparameters

| Parameter               | Default   | Description            |
| ----------------------- | --------- | ---------------------- |
| `total_timesteps`       | 1,000,000 | Total training steps   |
| `num_envs`              | 4         | Parallel environments  |
| `learning_rate`         | 3e-5      | Adam learning rate     |
| `buffer_size`           | 500,000   | Replay buffer capacity |
| `batch_size`            | 32        | Training batch size    |
| `gamma`                 | 0.99      | Discount factor        |
| `exploration_fraction`  | 0.3       | Epsilon decay period   |
| `exploration_final_eps` | 0.05      | Final exploration rate |
