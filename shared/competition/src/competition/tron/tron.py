"""
Tron game engine and duel orchestrator.

Ported from apex-tron (tron/game/). Items system and rendering stripped.
The engine runs in the worker process; player sandboxes communicate via HTTP.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import requests
import time
import uuid
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Grid cell constants
# ---------------------------------------------------------------------------
EMPTY = 0
WALL = 1
PLAYER_TRAIL_START = 2  # Player 0 trail = 2, Player 1 trail = 3, etc.

# Death cause constants
DEATH_WALL = "wall"
DEATH_SELF = "self"
DEATH_OPPONENT = "opponent"
DEATH_HEADON = "headon"


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------
class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def delta(self) -> Tuple[int, int]:
        deltas = {
            Direction.UP: (-1, 0),
            Direction.RIGHT: (0, 1),
            Direction.DOWN: (1, 0),
            Direction.LEFT: (0, -1),
        }
        return deltas[self]

    @property
    def opposite(self) -> "Direction":
        return Direction((self.value + 2) % 4)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class Player:
    def __init__(self, player_id: int, y: int, x: int, direction: Direction):
        self.id = player_id
        self.y = y
        self.x = x
        self.direction = direction
        self.alive = True
        self.trail: List[Tuple[int, int, int]] = []  # (y, x, step_created)

    @property
    def position(self) -> Tuple[int, int]:
        return (self.y, self.x)

    def get_next_position(self) -> Tuple[int, int]:
        dy, dx = self.direction.delta
        return (self.y + dy, self.x + dx)

    def move(self, step: int) -> None:
        self.trail.append((self.y, self.x, step))
        dy, dx = self.direction.delta
        self.y += dy
        self.x += dx

    def set_direction(self, direction: Direction) -> None:
        if direction != self.direction.opposite:
            self.direction = direction

    def apply_absolute_action(self, action: int) -> None:
        self.set_direction(Direction(action))

    def kill(self) -> None:
        self.alive = False


# ---------------------------------------------------------------------------
# GameConfig
# ---------------------------------------------------------------------------
@dataclass
class GameConfig:
    width: int = 32
    height: int = 32
    num_players: int = 2
    max_steps: int = 500
    wall_wrap: bool = False
    spawn_mode: str = "corners"  # "corners", "random", "center_spread"
    trail_fade: int = 0  # 0 = permanent trails

    def __post_init__(self):
        if self.width < 4:
            raise ValueError("Width must be at least 4")
        if self.height < 4:
            raise ValueError("Height must be at least 4")
        if self.num_players < 1:
            raise ValueError("Must have at least 1 player")
        if self.num_players > min(self.width, self.height):
            raise ValueError("Too many players for grid size")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")


# ---------------------------------------------------------------------------
# TronGame
# ---------------------------------------------------------------------------
class TronGame:
    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self.grid: np.ndarray = None
        self.players: List[Player] = []
        self.step_count = 0
        self.game_over = False
        self.winner: Optional[int] = None
        self._initialize_game()

    def _initialize_game(self) -> None:
        self._create_grid()
        self._spawn_players()

    def _create_grid(self) -> None:
        self.grid = np.zeros((self.config.height, self.config.width), dtype=np.int32)
        if not self.config.wall_wrap:
            self.grid[0, :] = WALL
            self.grid[-1, :] = WALL
            self.grid[:, 0] = WALL
            self.grid[:, -1] = WALL

    def _get_spawn_positions(self) -> List[Tuple[int, int, Direction]]:
        num_players = self.config.num_players
        h, w = self.config.height, self.config.width
        margin = 1
        min_y, max_y = margin, h - margin - 1
        min_x, max_x = margin, w - margin - 1

        if self.config.spawn_mode == "corners":
            corners = [
                (min_y, min_x, Direction.DOWN),
                (max_y, max_x, Direction.UP),
                (min_y, max_x, Direction.DOWN),
                (max_y, min_x, Direction.UP),
            ]
            if num_players > 4:
                mid_y, mid_x = h // 2, w // 2
                extras = [
                    (min_y, mid_x, Direction.DOWN),
                    (max_y, mid_x, Direction.UP),
                    (mid_y, min_x, Direction.RIGHT),
                    (mid_y, max_x, Direction.LEFT),
                ]
                corners.extend(extras)
            return corners[:num_players]

        elif self.config.spawn_mode == "random":
            import random

            positions = []
            used: Set[Tuple[int, int]] = set()
            directions = list(Direction)
            for _ in range(num_players):
                while True:
                    y = random.randint(min_y, max_y)
                    x = random.randint(min_x, max_x)
                    if (y, x) not in used:
                        used.add((y, x))
                        positions.append((y, x, random.choice(directions)))
                        break
            return positions

        else:  # center_spread
            mid_y, mid_x = h // 2, w // 2
            spread = min(h, w) // 4
            positions = []
            for i in range(num_players):
                angle = (2 * np.pi * i) / num_players
                y = int(mid_y + spread * np.sin(angle))
                x = int(mid_x + spread * np.cos(angle))
                y = max(min_y, min(max_y, y))
                x = max(min_x, min(max_x, x))
                if y < mid_y:
                    direction = Direction.DOWN
                elif y > mid_y:
                    direction = Direction.UP
                elif x < mid_x:
                    direction = Direction.RIGHT
                else:
                    direction = Direction.LEFT
                positions.append((y, x, direction))
            return positions

    def _spawn_players(self) -> None:
        self.players = []
        spawn_positions = self._get_spawn_positions()
        for i, (y, x, direction) in enumerate(spawn_positions):
            player = Player(i, y, x, direction)
            self.players.append(player)
            self.grid[y, x] = PLAYER_TRAIL_START + i

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        import random

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.step_count = 0
        self.game_over = False
        self.winner = None
        self._create_grid()
        self._spawn_players()
        return self.grid.copy()

    def step(self, actions: Dict[int, int]) -> Tuple[np.ndarray, Dict[int, float], bool, Dict]:
        if self.game_over:
            return self.grid.copy(), {p.id: 0.0 for p in self.players}, True, {"winner": self.winner}

        self.step_count += 1

        alive_players = [p for p in self.players if p.alive]
        for player in alive_players:
            if player.id in actions:
                player.apply_absolute_action(actions[player.id])

        # Calculate next positions
        next_positions: Dict[int, Tuple[int, int]] = {}
        for player in alive_players:
            next_pos = player.get_next_position()
            if self.config.wall_wrap:
                y, x = next_pos
                next_pos = (y % self.config.height, x % self.config.width)
            next_positions[player.id] = next_pos

        # Detect collisions
        deaths_this_step: Set[int] = set()
        death_causes: Dict[int, Dict] = {}

        for player_id, (y, x) in next_positions.items():
            if not self._is_valid_position(y, x):
                deaths_this_step.add(player_id)
                death_causes[player_id] = {"cause": DEATH_WALL, "killed_by": None}
            elif self.grid[y, x] == WALL:
                deaths_this_step.add(player_id)
                death_causes[player_id] = {"cause": DEATH_WALL, "killed_by": None}
            elif self.grid[y, x] >= PLAYER_TRAIL_START:
                deaths_this_step.add(player_id)
                trail_owner = int(self.grid[y, x]) - PLAYER_TRAIL_START
                if trail_owner == player_id:
                    death_causes[player_id] = {"cause": DEATH_SELF, "killed_by": player_id}
                else:
                    death_causes[player_id] = {"cause": DEATH_OPPONENT, "killed_by": trail_owner}

        # Head-on collisions
        position_to_players: Dict[Tuple[int, int], List[int]] = {}
        for player_id, pos in next_positions.items():
            if player_id not in deaths_this_step:
                if pos not in position_to_players:
                    position_to_players[pos] = []
                position_to_players[pos].append(player_id)
        for pos, player_ids in position_to_players.items():
            if len(player_ids) > 1:
                deaths_this_step.update(player_ids)
                for pid in player_ids:
                    other_ids = [p for p in player_ids if p != pid]
                    death_causes[pid] = {"cause": DEATH_HEADON, "killed_by": other_ids}

        # Trail fading
        if self.config.trail_fade > 0:
            self._fade_trails()

        # Move surviving players
        for player in alive_players:
            if player.id in deaths_this_step:
                player.kill()
            elif player.id in next_positions:
                player.move(self.step_count)
                y, x = next_positions[player.id]
                player.y, player.x = y, x
                self.grid[y, x] = PLAYER_TRAIL_START + player.id

        # Check game end
        alive_count = sum(1 for p in self.players if p.alive)
        if alive_count <= 1:
            self.game_over = True
            if alive_count == 1:
                self.winner = next(p.id for p in self.players if p.alive)

        truncated = self.step_count >= self.config.max_steps
        if truncated:
            self.game_over = True

        info = {
            "step": self.step_count,
            "alive_players": [p.id for p in self.players if p.alive],
            "deaths_this_step": list(deaths_this_step),
            "death_causes": death_causes,
            "winner": self.winner,
            "truncated": truncated,
        }
        return self.grid.copy(), {}, self.game_over, info

    def get_valid_actions(self, player_id: int) -> List[int]:
        player = self.get_player(player_id)
        if player is None or not player.alive:
            return []
        valid = []
        for action in range(4):
            direction = Direction(action)
            if direction == player.direction.opposite:
                continue
            dy, dx = direction.delta
            ny, nx = player.y + dy, player.x + dx
            if self.config.wall_wrap:
                ny = ny % self.config.height
                nx = nx % self.config.width
            if self._is_valid_position(ny, nx) and self.grid[ny, nx] == EMPTY:
                valid.append(action)
        return valid

    def get_player(self, player_id: int) -> Optional[Player]:
        if 0 <= player_id < len(self.players):
            return self.players[player_id]
        return None

    def _is_valid_position(self, y: int, x: int) -> bool:
        return 0 <= y < self.config.height and 0 <= x < self.config.width

    def _fade_trails(self) -> None:
        fade_threshold = self.step_count - self.config.trail_fade
        for player in self.players:
            new_trail = []
            for y, x, step_created in player.trail:
                if step_created > fade_threshold:
                    new_trail.append((y, x, step_created))
                else:
                    if self.grid[y, x] == PLAYER_TRAIL_START + player.id:
                        self.grid[y, x] = EMPTY
            player.trail = new_trail


# ---------------------------------------------------------------------------
# Game result model
# ---------------------------------------------------------------------------
class StepRecord(BaseModel):
    """Record of a single game step for replay/visualization."""

    step: int
    actions: Dict[int, int]  # player_id -> action taken
    positions: Dict[int, List[int]]  # player_id -> [y, x] after move
    alive: Dict[int, bool]  # player_id -> alive after step
    deaths: List[int] = []  # player_ids that died this step
    death_causes: Dict[int, Dict] = {}


class TronGameResult(BaseModel):
    game_id: str
    winner: Optional[int] = None  # player_id of winner, None if draw
    steps: int = 0
    max_steps: int = 500
    game_result: str = ""
    death_causes: Dict[int, Dict] = {}
    p1_survived_steps: int = 0
    p2_survived_steps: int = 0
    # Replay data
    config: Optional[Dict] = None  # game config for replay
    spawn_positions: Optional[Dict[int, Dict]] = None  # player_id -> {y, x, direction}
    history: List[StepRecord] = []  # per-step records


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _health_check(session: requests.Session, url: str, timeout: int = 1) -> bool:
    try:
        resp = session.get(f"{url}/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("ok", False)
    except Exception:
        return False


def _post_game(
    session: requests.Session,
    url: str,
    game_id: str,
    player_id: int,
    config: GameConfig,
    grid: np.ndarray,
    players: List[Player],
    timeout: float = 2.0,
) -> bool:
    opponent_positions = [[p.y, p.x] for p in players if p.id != player_id]
    me = next(p for p in players if p.id == player_id)
    try:
        resp = session.post(
            f"{url}/game",
            json={
                "game_id": game_id,
                "player_id": player_id,
                "config": {
                    "width": config.width,
                    "height": config.height,
                    "max_steps": config.max_steps,
                    "num_players": config.num_players,
                    "wall_wrap": config.wall_wrap,
                    "trail_fade": config.trail_fade,
                },
                "grid": grid.tolist(),
                "your_position": [me.y, me.x],
                "your_direction": int(me.direction),
                "opponent_positions": opponent_positions,
            },
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _post_move(
    session: requests.Session,
    url: str,
    game_id: str,
    step: int,
    grid: np.ndarray,
    player_id: int,
    players: List[Player],
    valid_actions: List[int],
    timeout: float = 0.5,
) -> Optional[int]:
    """Request a move from the player. Returns action int or None on failure."""
    me = next(p for p in players if p.id == player_id)
    opponents = [p for p in players if p.id != player_id]
    try:
        resp = session.post(
            f"{url}/move",
            json={
                "game_id": game_id,
                "step": step,
                "grid": grid.tolist(),
                "your_position": [me.y, me.x],
                "your_direction": int(me.direction),
                "your_alive": me.alive,
                "opponent_positions": [[p.y, p.x] for p in opponents],
                "opponent_alive": [p.alive for p in opponents],
                "valid_actions": valid_actions,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("action")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Duel game orchestrator
# ---------------------------------------------------------------------------
def run_duel_game(
    p1_url: str,
    p2_url: str,
    config: GameConfig | None = None,
    seed: int = 42,
    move_timeout: float = 0.5,
    startup_health_check_timeout_in_seconds: int = 10,
) -> TronGameResult:
    """
    Run a single Tron duel game between two remote players.

    The game engine runs locally; players are HTTP API servers in sandboxes.
    """
    config = config or GameConfig()
    game = TronGame(config)
    grid = game.reset(seed=seed)
    game_id = str(uuid.uuid4())

    urls = {0: p1_url, 1: p2_url}
    sess = requests.Session()

    # Health check both players
    start_time = time.time()
    ready = {0: False, 1: False}
    while True:
        for pid in [0, 1]:
            if not ready[pid]:
                ready[pid] = _health_check(sess, urls[pid])
        if all(ready.values()):
            break
        if time.time() - start_time > startup_health_check_timeout_in_seconds:
            failed = [pid for pid, ok in ready.items() if not ok]
            return TronGameResult(
                game_id=game_id,
                steps=0,
                max_steps=config.max_steps,
                game_result=f"Health check failed for player(s) {failed}",
            )
        time.sleep(0.5)

    # Send game init to both players
    for pid in [0, 1]:
        ok = _post_game(sess, urls[pid], game_id, pid, config, grid, game.players)
        if not ok:
            return TronGameResult(
                game_id=game_id,
                steps=0,
                max_steps=config.max_steps,
                game_result=f"Player {pid} failed to initialize game",
            )

    # Record initial state for replay
    game_config_dict = {
        "width": config.width,
        "height": config.height,
        "max_steps": config.max_steps,
        "num_players": config.num_players,
        "wall_wrap": config.wall_wrap,
        "trail_fade": config.trail_fade,
        "seed": seed,
    }
    spawn_positions = {p.id: {"y": p.y, "x": p.x, "direction": int(p.direction)} for p in game.players}

    # Track when each player was last alive
    survived_steps = {0: 0, 1: 0}

    # Game loop
    all_death_causes: Dict[int, Dict] = {}
    step_history: List[StepRecord] = []
    while not game.game_over:
        actions: Dict[int, int] = {}
        for pid in [0, 1]:
            player = game.get_player(pid)
            if not player.alive:
                continue
            valid = game.get_valid_actions(pid)
            action = _post_move(
                sess,
                urls[pid],
                game_id,
                game.step_count,
                grid,
                pid,
                game.players,
                valid,
                timeout=move_timeout,
            )
            if action is not None and action in range(4):
                actions[pid] = action
            else:
                # Timeout or invalid response: keep current direction
                actions[pid] = int(player.direction)

        grid, _, done, info = game.step(actions)

        # Record this step
        step_history.append(
            StepRecord(
                step=info["step"],
                actions=actions,
                positions={p.id: [p.y, p.x] for p in game.players},
                alive={p.id: p.alive for p in game.players},
                deaths=info.get("deaths_this_step", []),
                death_causes=info.get("death_causes", {}),
            )
        )

        for pid in [0, 1]:
            if game.get_player(pid).alive:
                survived_steps[pid] = game.step_count

        for pid, cause in info.get("death_causes", {}).items():
            all_death_causes[pid] = cause

    # Determine result string
    if game.winner is not None:
        game_result = f"Player {game.winner} won"
    elif info.get("truncated"):
        game_result = "Draw (max steps reached)"
    else:
        game_result = "Draw (simultaneous death)"

    return TronGameResult(
        game_id=game_id,
        winner=game.winner,
        steps=game.step_count,
        max_steps=config.max_steps,
        game_result=game_result,
        death_causes=all_death_causes,
        p1_survived_steps=survived_steps[0],
        p2_survived_steps=survived_steps[1],
        config=game_config_dict,
        spawn_positions=spawn_positions,
        history=step_history,
    )
