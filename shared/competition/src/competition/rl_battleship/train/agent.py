# agent.py - RL-based shooter for inference
import torch
import torch.nn as nn
import random
import os
from typing import List, Dict, Tuple, Optional

from state_encoder import encode_state


class BattleshipQNet(nn.Module):
    """
    Q-Network architecture matching what stable-baselines3 DQN uses.
    This is a standalone version for inference without needing sb3.
    """

    def __init__(self, board_size: int = 10, features_dim: int = 256):
        super().__init__()
        self.board_size = board_size

        # CNN feature extractor (matches BattleshipCNN in train.py)
        self.features_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * board_size * board_size, features_dim),
            nn.ReLU(),
        )

        # Q-network head (matches sb3 default)
        self.q_net = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, board_size * board_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(x)
        return self.q_net(features)


class RLShooter:
    """
    RL-based shooter that uses a trained model to select moves.
    Can be used as a drop-in replacement for RandomShooter.
    """

    def __init__(self, size: int = 10, model_path: Optional[str] = None, device: str = "cpu"):
        self.size = size
        self.device = torch.device(device)
        self.model = BattleshipQNet(board_size=size).to(self.device)
        self.model_loaded = False

        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model_loaded = True
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")
                print("Falling back to random actions")

        # Always set to eval mode for inference
        self.model.eval()

        self.tried: set = set()
        self.history: List[Dict] = []
        self.last_shot: Optional[Tuple[int, int]] = None

    def next_shot(self, result: Optional[Dict] = None) -> Tuple[int, int]:
        """
        Determines the next shot.
        If result is provided, it records the outcome of the LAST shot.
        """
        # Update history with result
        if result is not None and self.last_shot is not None:
            self.history.append(
                {
                    "type": "result",
                    "x": self.last_shot[0],
                    "y": self.last_shot[1],
                    "hit": bool(result.get("hit", False)),
                    "sunk": result.get("sunk"),
                }
            )

        # If model not loaded, fall back to random
        if not self.model_loaded:
            return self._random_shot()

        # Model inference
        state_tensor = encode_state(self.history, self.size).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)

        # Mask already tried positions
        for x, y in self.tried:
            idx = y * self.size + x
            q_values[idx] = -float("inf")

        # Select best action
        best_action_idx = torch.argmax(q_values).item()

        # If all actions are masked (shouldn't happen), fallback
        if q_values[best_action_idx] == -float("inf"):
            return self._random_shot()

        y = best_action_idx // self.size
        x = best_action_idx % self.size

        self.tried.add((x, y))
        self.last_shot = (x, y)
        return int(x), int(y)

    def _random_shot(self) -> Tuple[int, int]:
        """Simple random fallback."""
        candidates = []
        for y in range(self.size):
            for x in range(self.size):
                if (x, y) not in self.tried:
                    candidates.append((x, y))

        if not candidates:
            raise StopIteration("No more positions to try")

        choice = random.choice(candidates)
        self.tried.add(choice)
        self.last_shot = choice
        return choice
