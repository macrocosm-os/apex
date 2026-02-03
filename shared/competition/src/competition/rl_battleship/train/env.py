# env.py - Gym environment wrapper for Battleship
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

# Add battleship directory to path to import baseline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "battleship")))

# Import baseline board from battleship module
from baseline import Board


class BattleshipEnv(gym.Env):
    """
    Gym environment for Battleship shooting optimization.

    The agent's goal is to sink all ships in as few shots as possible.
    This is a single-player "solitaire" version - we're just optimizing
    the search strategy, not playing against an opponent.

    Observation: 3-channel 10x10 grid
        - Channel 0: Cells that have been shot (1.0) vs not shot (0.0)
        - Channel 1: Hits (1.0) vs not hits (0.0)
        - Channel 2: Cells belonging to sunk ships (1.0) vs not (0.0)

    Action: Integer 0-99 representing flattened (y * size + x) coordinate

    Reward:
        - Miss: -0.1
        - Hit: +1.0
        - Sunk ship: +5.0
        - Win (all ships sunk): +20.0
        - Invalid action (repeat shot): -1.0 (action is masked, shouldn't happen)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, size: int = 10, render_mode: str | None = None):
        super().__init__()
        self.size = size
        self.render_mode = render_mode

        # Observation space: 3 channels x size x size
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, size, size), dtype=np.float32)

        # Action space: flattened grid coordinates
        self.action_space = spaces.Discrete(size * size)

        # Game state
        self.board: Board | None = None
        self.shots: set = set()
        self.hits: set = set()
        self.sunk_cells: set = set()
        self.steps = 0
        self.max_steps = size * size

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Create new board with random ship placement
        self.board = Board(size=self.size, seed=seed)
        self.board.place_ships_randomly()

        # Reset tracking
        self.shots = set()
        self.hits = set()
        self.sunk_cells = set()
        self.steps = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: int):
        # Convert action to coordinates
        y = action // self.size
        x = action % self.size
        coord = (x, y)

        # Check for invalid action (repeat shot)
        if coord in self.shots:
            # Penalize repeats and count toward step budget to avoid infinite loops
            self.steps += 1
            truncated = self.steps >= self.max_steps
            obs = self._get_obs()
            return obs, -1.0, False, truncated, self._get_info()

        self.shots.add(coord)
        self.steps += 1

        # Process shot
        ship_name = self.board.occupied.get(coord)
        reward = -0.1  # Base penalty for each shot (encourages efficiency)

        if ship_name:
            # Hit!
            ship = self.board.ships[ship_name]
            ship.hits.add(coord)
            self.hits.add(coord)
            reward = 1.0

            # Check if sunk
            if ship.is_sunk():
                reward = 5.0
                # Mark all cells of this ship as sunk
                for cell in ship.cells:
                    self.sunk_cells.add(cell)
        else:
            # Miss
            self.board.misses.add(coord)

        # Check win condition
        terminated = all(s.is_sunk() for s in self.board.ships.values())
        if terminated:
            reward = 20.0

        # Check if we've exhausted all moves (shouldn't happen in normal play)
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Build the 3-channel observation tensor."""
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)

        # Channel 0: Shots taken
        for x, y in self.shots:
            obs[0, y, x] = 1.0

        # Channel 1: Hits
        for x, y in self.hits:
            obs[1, y, x] = 1.0

        # Channel 2: Sunk ships
        for x, y in self.sunk_cells:
            obs[2, y, x] = 1.0

        return obs

    def _get_info(self) -> dict:
        """Return auxiliary info."""
        ships_remaining = sum(1 for s in self.board.ships.values() if not s.is_sunk())
        return {
            "steps": self.steps,
            "ships_remaining": ships_remaining,
            "shots_taken": len(self.shots),
            "hits": len(self.hits),
        }

    def action_masks(self) -> np.ndarray:
        """
        Return a boolean mask of valid actions.
        Used by MaskablePPO from sb3-contrib.
        """
        mask = np.ones(self.size * self.size, dtype=bool)
        for x, y in self.shots:
            mask[y * self.size + x] = False
        return mask

    def render(self):
        if self.render_mode == "human" or self.render_mode == "ansi":
            grid = [["." for _ in range(self.size)] for _ in range(self.size)]

            # Mark ships (hidden)
            # for (x, y), name in self.board.occupied.items():
            #     grid[y][x] = name[0]

            # Mark hits
            for x, y in self.hits:
                grid[y][x] = "X"

            # Mark misses
            for x, y in self.shots - self.hits:
                grid[y][x] = "o"

            header = "   " + " ".join(f"{i:2d}" for i in range(self.size))
            rows = [header]
            for y in range(self.size):
                rows.append(f"{y:2d} " + " ".join(f"{grid[y][x]:>2}" for x in range(self.size)))

            output = "\n".join(rows)
            if self.render_mode == "human":
                print(output)
            return output
        return None


def make_env(size: int = 10, seed: int | None = None):
    """Factory function for creating environments (used by SubprocVecEnv)."""

    def _init():
        env = BattleshipEnv(size=size)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _init
