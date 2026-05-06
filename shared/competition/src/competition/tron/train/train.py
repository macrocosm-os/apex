"""
Minimal Tron RL training script.

Trains a DQN agent against a random opponent and exports a TorchScript .pt
file ready for competition submission.

Run from this directory:
    python train.py --total-timesteps 500000

Outputs:
    models/tron_model.pt   <- TorchScript file to submit
"""

import argparse
import os

import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env import TronEnv


class TronCNN(BaseFeaturesExtractor):
    """Small CNN feature extractor for the 5-channel Tron observation."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]  # 5
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs))


def export_torchscript(model: DQN, output_path: str, height: int, width: int):
    """Trace the q_net into a self-contained TorchScript file."""
    model.q_net.eval()
    device = next(model.q_net.parameters()).device
    dummy = torch.zeros(1, 5, height, width, device=device)
    scripted = torch.jit.trace(model.q_net, dummy)
    scripted.save(output_path)
    print(f"TorchScript model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--exploration-fraction", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    env = TronEnv(width=args.width, height=args.height)

    policy_kwargs = {
        "features_extractor_class": TronCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [256, 256],
    }

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
    )

    print(f"Training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    sb3_path = os.path.join(args.output_dir, "tron_model")
    model.save(sb3_path)
    print(f"sb3 checkpoint saved to {sb3_path}.zip")

    pt_path = os.path.join(args.output_dir, "tron_model.pt")
    export_torchscript(model, pt_path, args.height, args.width)


if __name__ == "__main__":
    main()
