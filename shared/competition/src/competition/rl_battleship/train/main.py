#!/usr/bin/env python
"""
Battleship RL Training CLI

Train a reinforcement learning agent to play Battleship optimally.
The trained model can be exported as TorchScript for submission to the competition.

Usage:
    # Train a new model (outputs to ./models by default)
    python main.py train

    # Train with custom settings
    python main.py train --total-timesteps 500000 --num-envs 8 --output-dir my_models

    # Evaluate a trained model
    python main.py evaluate --model models/battleship_model.zip

    # Watch a single game with the trained model
    python main.py play --model models/battleship_model.zip

    # Export a trained model to TorchScript (for competition submission)
    python main.py export --model models/battleship_model.zip --output model.pt
"""
import argparse
import sys


def cmd_train(args):
    """Train a new Battleship RL model."""
    from train import train

    # Build args namespace for train.py
    train_args = argparse.Namespace(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        disable_eval=args.disable_eval,
    )
    train(train_args)


def cmd_evaluate(args):
    """Evaluate a trained model against random baseline."""
    from test_model import evaluate

    evaluate(args.model, n_episodes=args.episodes, show_game=False)


def cmd_play(args):
    """Watch a single game with the trained model."""
    from test_model import show_single_game

    show_single_game(args.model)


def cmd_export(args):
    """Export a trained model to TorchScript format for competition submission."""
    import torch
    from stable_baselines3 import DQN

    print(f"Loading model from {args.model}...")
    model = DQN.load(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.q_net.to(device)
    model.q_net.eval()

    # Create dummy input for tracing
    dummy_input = torch.zeros(1, 3, 10, 10, device=device)

    print("Exporting to TorchScript...")
    scripted_model = torch.jit.trace(model.q_net, dummy_input)
    scripted_model.save(args.output)

    print(f"TorchScript model saved to {args.output}")
    print("\nTo use this model in the competition:")
    print(f"  1. Include {args.output} in your submission")
    print("  2. Use launch_battleship_rl.py with --model flag")


def main():
    parser = argparse.ArgumentParser(
        description="Battleship RL Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--total-timesteps", type=int, default=1_000_000, help="Total training timesteps (default: 1M)"
    )
    train_parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments (default: 4)")
    train_parser.add_argument("--output-dir", type=str, default="models", help="Output directory (default: models)")
    train_parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    train_parser.add_argument("--buffer-size", type=int, default=500_000, help="Replay buffer size")
    train_parser.add_argument("--learning-starts", type=int, default=50_000, help="Steps before learning")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    train_parser.add_argument("--tau", type=float, default=1.0, help="Soft update coefficient")
    train_parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    train_parser.add_argument("--target-update-interval", type=int, default=1000, help="Target net update interval")
    train_parser.add_argument("--exploration-fraction", type=float, default=0.3, help="Exploration decay fraction")
    train_parser.add_argument("--exploration-final-eps", type=float, default=0.05, help="Final exploration epsilon")
    train_parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Checkpoint frequency")
    train_parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency")
    train_parser.add_argument("--disable-eval", action="store_true", help="Disable evaluation callback")
    train_parser.set_defaults(func=cmd_train)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    eval_parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    eval_parser.set_defaults(func=cmd_evaluate)

    # Play command
    play_parser = subparsers.add_parser("play", help="Watch a single game")
    play_parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    play_parser.set_defaults(func=cmd_play)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to TorchScript")
    export_parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    export_parser.add_argument("--output", type=str, default="battleship_model.pt", help="Output .pt file")
    export_parser.set_defaults(func=cmd_export)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
