# test_model.py - Test the trained Battleship RL model
import numpy as np
from stable_baselines3 import DQN
from env import BattleshipEnv


def run_episode(env, model=None, render=False):
    """Run a single episode, optionally using a trained model."""
    obs, info = env.reset()
    total_reward = 0
    steps = 0

    while True:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Random valid action
            valid_actions = [i for i in range(env.action_space.n) if (i % env.size, i // env.size) not in env.shots]
            action = np.random.choice(valid_actions) if valid_actions else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if render:
            env.render()
            print(f"Step {steps}: action={action}, reward={reward:.1f}, total={total_reward:.1f}")
            print(f"  Ships remaining: {info['ships_remaining']}, Hits: {info['hits']}")
            print()

        if terminated or truncated:
            break

    return total_reward, steps, info


def evaluate(model_path, n_episodes=20, show_game=False):
    """Evaluate the trained model."""
    env = BattleshipEnv(size=10, render_mode="ansi" if show_game else None)

    # Load trained model
    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path)
    print("Model loaded successfully!\n")

    # Evaluate trained model
    print("=" * 50)
    print("TRAINED MODEL EVALUATION")
    print("=" * 50)

    model_rewards = []
    model_steps = []
    model_wins = 0

    for i in range(n_episodes):
        reward, steps, info = run_episode(env, model, render=(show_game and i == 0))
        model_rewards.append(reward)
        model_steps.append(steps)
        if info["ships_remaining"] == 0:
            model_wins += 1
        if not show_game:
            print(f"Episode {i+1}: reward={reward:.1f}, steps={steps}, ships_remaining={info['ships_remaining']}")

    print(f"\nTrained Model Results ({n_episodes} episodes):")
    print(f"  Mean reward: {np.mean(model_rewards):.2f} ± {np.std(model_rewards):.2f}")
    print(f"  Mean steps:  {np.mean(model_steps):.1f} ± {np.std(model_steps):.1f}")
    print(f"  Win rate:    {model_wins}/{n_episodes} ({100*model_wins/n_episodes:.0f}%)")

    # Evaluate random baseline
    print("\n" + "=" * 50)
    print("RANDOM BASELINE EVALUATION")
    print("=" * 50)

    random_rewards = []
    random_steps = []
    random_wins = 0

    for i in range(n_episodes):
        reward, steps, info = run_episode(env, model=None, render=False)
        random_rewards.append(reward)
        random_steps.append(steps)
        if info["ships_remaining"] == 0:
            random_wins += 1
        print(f"Episode {i+1}: reward={reward:.1f}, steps={steps}, ships_remaining={info['ships_remaining']}")

    print(f"\nRandom Baseline Results ({n_episodes} episodes):")
    print(f"  Mean reward: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print(f"  Mean steps:  {np.mean(random_steps):.1f} ± {np.std(random_steps):.1f}")
    print(f"  Win rate:    {random_wins}/{n_episodes} ({100*random_wins/n_episodes:.0f}%)")

    # Comparison
    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    improvement = np.mean(model_rewards) - np.mean(random_rewards)
    print(f"Reward improvement: {improvement:+.2f}")
    print(f"Steps improvement:  {np.mean(random_steps) - np.mean(model_steps):+.1f} fewer steps")

    return model_rewards, random_rewards


def show_single_game(model_path):
    """Show a single game with the trained model."""
    env = BattleshipEnv(size=10, render_mode="human")
    model = DQN.load(model_path)

    print("Playing a single game with the trained model...\n")
    obs, info = env.reset()
    env.render()

    step = 0
    total_reward = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        x = action % env.size
        y = action // env.size

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        result = "HIT!" if reward > 0 else "miss"
        if reward >= 5:
            result = "SUNK!"
        if reward >= 20:
            result = "WIN!"

        print(f"\nStep {step}: Shot at ({x}, {y}) -> {result} (reward: {reward:+.1f})")
        print(f"Ships remaining: {info['ships_remaining']}, Total hits: {info['hits']}")
        env.render()

        if terminated or truncated:
            break

    print(f"\n{'='*50}")
    print(f"Game finished in {step} steps with total reward: {total_reward:.1f}")
    print(f"Ships sunk: {5 - info['ships_remaining']}/5")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test trained Battleship model")
    parser.add_argument("--model", type=str, default="models/best_model/best_model.zip", help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes for evaluation")
    parser.add_argument("--show-game", action="store_true", help="Show a single game visually")

    args = parser.parse_args()

    if args.show_game:
        show_single_game(args.model)
    else:
        evaluate(args.model, n_episodes=args.episodes)
