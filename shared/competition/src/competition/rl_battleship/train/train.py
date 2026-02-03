# train.py - Training script using stable-baselines3 with SubprocVecEnv
import argparse
import os

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

# Local import for direct script execution
from env import make_env


class BattleshipCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the Battleship environment.
    Processes the 3-channel 10x10 observation.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 3 channels

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened size
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def train(args):
    """Main training function."""

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    print(f"Number of parallel environments: {args.num_envs}")

    # Create vectorized environment with SubprocVecEnv for true parallelism
    # Each subprocess runs its own environment instance
    env_fns = [make_env(size=10, seed=i) for i in range(args.num_envs)]
    # Using spawn avoids fork-related deadlocks with PyTorch on macOS
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecMonitor(vec_env)  # Wrap with monitor for logging

    # Create a separate eval environment (optional)
    # Use an in-process env for eval to avoid nested subprocess hangs
    eval_env = None
    if not args.disable_eval:
        eval_env = DummyVecEnv([make_env(size=10, seed=1000)])
        eval_env = VecMonitor(eval_env)

    # Custom policy kwargs with our CNN extractor
    policy_kwargs = {
        "features_extractor_class": BattleshipCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [256, 256],  # Additional MLP layers after feature extraction
    }

    # Create the DQN model
    model = DQN(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=(4, "step"),
        gradient_steps=1,
        max_grad_norm=10.0,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=os.path.join(args.output_dir, "tensorboard"),
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.num_envs,  # Adjust for vectorized env
        save_path=os.path.join(args.output_dir, "checkpoints"),
        name_prefix="battleship_dqn",
    )

    callbacks = [checkpoint_callback]
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.output_dir, "best_model"),
            log_path=os.path.join(args.output_dir, "eval_logs"),
            eval_freq=args.eval_freq // args.num_envs,
            n_eval_episodes=10,
            deterministic=True,
        )
        callbacks.append(eval_callback)

    # Train
    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)

    # Save final model
    final_path = os.path.join(args.output_dir, "battleship_model")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}")

    # Export as TorchScript for the generic RL inference API
    # TorchScript is self-contained and doesn't require class definitions at load time
    model.q_net.eval()
    dummy_input = torch.zeros(1, 3, 10, 10, device=device)  # [batch, channels, height, width]
    scripted_model = torch.jit.trace(model.q_net, dummy_input)
    scripted_model.save(os.path.join(args.output_dir, "battleship_model.pt"))
    print(f"TorchScript model saved to {os.path.join(args.output_dir, 'battleship_model.pt')}")

    # Cleanup
    vec_env.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Battleship RL agent with stable-baselines3")

    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments (SubprocVecEnv)")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save models and logs")

    # DQN hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=500_000, help="Replay buffer size")
    parser.add_argument("--learning-starts", type=int, default=50_000, help="Steps before learning starts")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--tau", type=float, default=1.0, help="Soft update coefficient (1.0 = hard update)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--target-update-interval", type=int, default=1000, help="Target network update interval")
    parser.add_argument(
        "--exploration-fraction", type=float, default=0.3, help="Fraction of training for epsilon decay"
    )
    parser.add_argument("--exploration-final-eps", type=float, default=0.05, help="Final exploration epsilon")

    # Callback frequencies
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Checkpoint save frequency (timesteps)")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency (timesteps)")
    parser.add_argument("--disable-eval", action="store_true", help="Disable EvalCallback to avoid stalls")

    args = parser.parse_args()
    train(args)
