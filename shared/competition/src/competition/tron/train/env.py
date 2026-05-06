"""
Gymnasium environment wrapping the Tron game engine for RL training.

The agent plays as Player 0; the opponent plays as Player 1. The default
opponent picks uniformly random valid actions. Swap in a smarter opponent
(e.g. a frozen snapshot of your own model) by passing `opponent_fn` to
`TronEnv` for self-play.
"""

import random
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from competition.tron.tron import GameConfig, TronGame, PLAYER_TRAIL_START, WALL


def encode_state(game: TronGame, player_id: int) -> np.ndarray:
    """5-channel observation matching launch_tron_rl.py exactly."""
    grid = game.grid
    h, w = grid.shape
    state = np.zeros((5, h, w), dtype=np.float32)

    state[0] = (grid == WALL).astype(np.float32)
    state[1] = (grid == PLAYER_TRAIL_START + player_id).astype(np.float32)

    opp = np.zeros_like(grid, dtype=np.float32)
    for oid in range(8):
        if oid != player_id:
            opp += (grid == PLAYER_TRAIL_START + oid).astype(np.float32)
    state[2] = np.clip(opp, 0.0, 1.0)

    me = game.get_player(player_id)
    if me is not None and me.alive:
        state[3, me.y, me.x] = 1.0

    for p in game.players:
        if p.id != player_id and p.alive:
            state[4, p.y, p.x] = 1.0

    return state


def random_opponent(game: TronGame, player_id: int) -> int:
    valid = game.get_valid_actions(player_id)
    if not valid:
        me = game.get_player(player_id)
        return int(me.direction) if me else 0
    return random.choice(valid)


class TronEnv(gym.Env):
    """
    Single-agent view of a Tron duel: agent is Player 0, opponent is Player 1.

    Observation: (5, H, W) float32 — see encode_state.
    Action: Discrete(4) — UP=0, RIGHT=1, DOWN=2, LEFT=3.

    Reward shaping:
        +0.01 per surviving step (encourages staying alive)
        +1.0  on win
        -1.0  on loss
         0.0  on draw

    Picking an invalid action (reversing or driving into a wall/trail) just
    kills the agent — the env doesn't pre-filter; learning to mask invalids
    is part of the task. To use action masking, expose `action_masks()`.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        max_steps: int = 500,
        opponent_fn: Optional[Callable[[TronGame, int], int]] = None,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.opponent_fn = opponent_fn or random_opponent

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5, height, width), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.game: Optional[TronGame] = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        config = GameConfig(width=self.width, height=self.height, max_steps=self.max_steps, num_players=2)
        self.game = TronGame(config)
        return encode_state(self.game, player_id=0), {}

    def step(self, action: int):
        opp_action = self.opponent_fn(self.game, 1)
        _, _, done, info = self.game.step({0: int(action), 1: int(opp_action)})

        me = self.game.get_player(0)
        opp = self.game.get_player(1)

        if done:
            if me.alive and not opp.alive:
                reward = 1.0
            elif opp.alive and not me.alive:
                reward = -1.0
            else:
                reward = 0.0  # draw (both dead, or max_steps reached)
        else:
            reward = 0.01

        truncated = bool(info.get("truncated", False))
        terminated = done and not truncated
        obs = encode_state(self.game, player_id=0)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Boolean mask of valid actions for the agent. Use with sb3-contrib MaskablePPO."""
        valid = set(self.game.get_valid_actions(0))
        return np.array([a in valid for a in range(4)], dtype=bool)

    def render(self):
        if self.game is None:
            return None
        chars = {0: ".", 1: "#"}
        rows = []
        for row in self.game.grid:
            rows.append("".join(chars.get(int(v), "x") for v in row))
        out = "\n".join(rows)
        print(out)
        return out
