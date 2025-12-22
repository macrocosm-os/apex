# To run:
# first run player 1 and player 2 servers in separate terminals
#     cd shared/competition/src/competition/battleship
#     python baseline.py --port 8001
#     python baseline.py --port 8002
# second run the validator client in a third terminal
#     python battleship.py --p1 http://localhost:8001 --p2 http://localhost:8002 --console

# To replay a game:
#     python battleship.py --replay ../../../../../worker-data/game_21db0542-9e3d-4dd7-b7eb-60e7ce9dc748.json --console


from typing import Dict, Set, Tuple, Optional, List, Any
from pydantic import BaseModel, Field
from enum import Enum
import time
import uuid
import argparse
import requests
import random
import json

Coord = Tuple[int, int]


DEFAULT_SHIPS: Dict[str, int] = {
    "Carrier": 5,
    "Battleship": 4,
    "Cruiser": 3,
    "Submarine": 3,
    "Destroyer": 2,
}


class Name(str, Enum):
    PLAYER_1 = "Player 1"
    PLAYER_2 = "Player 2"


class Ship(BaseModel):
    name: str
    length: int
    cells: Set[Coord]
    hits: Set[Coord] = Field(default_factory=set)

    def is_sunk(self) -> bool:
        return self.cells == self.hits


class RemotePlayer(BaseModel):
    id: str
    name: Name
    base_url: str
    last_result: Optional[Dict] = None  # last result for THIS player's previous shot
    ships: Dict[str, Ship] = Field(default_factory=dict)
    shot_history: Set[Coord] = Field(default_factory=set)


class GameResult(BaseModel):
    """Result for a solo battleship game."""

    name: str | None = None
    game_id: str
    turns: int = 0
    max_turns: int
    game_result: str = ""
    p1: RemotePlayer
    board: Dict[str, Any]
    board_size: int


def health_check(session: requests.Session, player_url: str, timeout: int = 1) -> bool:
    try:
        resp = session.get(f"{player_url}/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json()["ok"]
    except Exception as e:
        return False


def init_board(session: requests.Session, player_url: str, game_id: str, size: int, timeout: int = 1):
    resp = session.post(
        f"{player_url}/board",
        json={"game_id": game_id, "size": size, "ships_spec": DEFAULT_SHIPS},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["board"]


def ask_next_move(
    session: requests.Session, player_url: str, game_id: str, previous_result: Optional[Dict] = None, timeout: int = 1
) -> Coord:
    resp = session.post(
        f"{player_url}/next-move",
        json={"game_id": game_id, "result": previous_result},
        timeout=timeout,
    )
    resp.raise_for_status()
    d = resp.json()
    return int(d["x"]), int(d["y"])


class Board:
    """Own board model. Coordinates are 0‑based (x, y). x = col, y = row."""

    def __init__(self, size: int = 10, ships_spec: Optional[Dict[str, int]] = None, seed: Optional[int] = None):
        self.size = size
        self.ships_spec = ships_spec or DEFAULT_SHIPS
        self.ships: Dict[str, Ship] = {}
        self.occupied: Dict[Coord, str] = {}  # cell -> ship name
        self.misses: Set[Coord] = set()  # track missed shots
        self.rng = random.Random(seed)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def _can_place(self, coords: List[Coord]) -> bool:
        return all(self.in_bounds(x, y) and (x, y) not in self.occupied for x, y in coords)

    def _all_valid_positions(self, length: int) -> List[List[Coord]]:
        """Enumerate every valid, non-overlapping placement for a ship of 'length'."""
        positions: List[List[Coord]] = []
        # Horizontal placements
        for y in range(self.size):
            for x in range(self.size - length + 1):
                coords = [(x + i, y) for i in range(length)]
                if self._can_place(coords):
                    positions.append(coords)
        # Vertical placements
        for x in range(self.size):
            for y in range(self.size - length + 1):
                coords = [(x, y + i) for i in range(length)]
                if self._can_place(coords):
                    positions.append(coords)
        return positions

    def place_ships_randomly(self, max_tries: int = 1000) -> None:
        """Place all ships randomly with restart-backtracking to avoid dead-ends."""
        spec_items = sorted(self.ships_spec.items(), key=lambda kv: kv[1], reverse=True)
        for _ in range(max_tries):
            self.ships.clear()
            self.occupied.clear()
            success = True
            for name, length in spec_items:
                positions = self._all_valid_positions(length)
                if not positions:
                    success = False
                    break
                choice = self.rng.choice(positions)
                ship = Ship(name=name, length=length, cells=set(choice))
                self.ships[name] = ship
                for cell in choice:
                    self.occupied[cell] = name
            if success:
                return
        raise RuntimeError("Failed to place ships after many attempts; try a larger board or different seed.")

    # ---- Serialization helpers for remote play ----
    def to_payload(self) -> Dict[str, Any]:
        ships_payload = []
        for name, ship in self.ships.items():
            cells = sorted(list(ship.cells))
            ships_payload.append({"name": name, "cells": cells})
        return {"size": self.size, "ships": ships_payload}


class BoardManager:
    """Own board model. Coordinates are 0‑based (x, y). x = col, y = row."""

    def __init__(self, size: int = 10, ships_spec: Optional[Dict[str, int]] = None, seed: Optional[int] = None):
        self.size = size
        self.ships_spec = ships_spec or DEFAULT_SHIPS
        self.ships: Dict[str, Ship] = {}
        self.occupied: Dict[Coord, str] = {}  # cell -> ship name
        self.misses: Set[Coord] = set()  # track missed shots
        self.rng = random.Random(seed)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def receive_shot(self, x: int, y: int) -> Tuple[bool, Optional[str]]:
        """
        Process a shot at (x, y) on *this* board.
        Returns (hit, sunk_ship_name). sunk_ship_name is None unless the shot just sank a ship.
        """
        if not self.in_bounds(x, y):
            raise ValueError("Shot is out of bounds")
        name = self.occupied.get((x, y))
        if not name:
            self.misses.add((x, y))  # record the miss
            return False, None
        ship = self.ships[name]
        ship.hits.add((x, y))
        if ship.is_sunk():
            return True, ship.name
        return True, None

    def all_ships_sunk(self) -> bool:
        return all(s.is_sunk() for s in self.ships.values())

    def render(self, reveal: bool = True) -> str:
        """String visualization. If reveal=False, ships are hidden (opponent view)."""
        # ANSI color codes
        BLUE = "\033[94m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"

        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        # place ships
        for (x, y), name in self.occupied.items():
            n = name[0] if name != "Cruiser" else "R"  # rename cruiser to 'R' for readability
            grid[y][x] = n if reveal else "."
        # mark hits
        for ship in self.ships.values():
            for x, y in ship.hits:
                grid[y][x] = "X"
        # mark misses
        for x, y in self.misses:
            grid[y][x] = "o"
        # to string with headers
        header = "   " + " ".join(f"{x:2d}" for x in range(self.size))
        rows = [header]
        for y in range(self.size):
            colored_cells = []
            for c in grid[y]:
                if c == "X":
                    colored_cells.append(f"{RED}{c:>2}{RESET}")
                elif c in (".", "o"):
                    colored_cells.append(f"{BLUE}{c:>2}{RESET}")
                else:
                    colored_cells.append(f"{YELLOW}{c:>2}{RESET}")
            rows.append(f"{y:2d} " + " ".join(colored_cells))
        return "\n".join(rows)

    # ---- Serialization helpers for remote play ----
    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "BoardManager":
        size = int(payload["size"])
        b = cls(size=size)
        for s in payload["ships"]:
            name = s["name"]
            cells = set((int(x), int(y)) for x, y in s["cells"])
            ship = Ship(name=name, length=len(cells), cells=cells)
            b.ships[name] = ship
            for c in cells:
                b.occupied[c] = name
        return b


class Validator:
    """
    Independent board validator.
    Required checks:
      1) All ship cells are fully on the board.
      2) No ship overlaps another.
    Optional:
      - enforce_linear: ships must be straight contiguous lines of declared length.
      - enforce_no_touching: ships cannot touch (even diagonally).
    """

    def __init__(
        self,
        size: int = 10,
        ships_spec: Optional[Dict[str, int]] = None,
        enforce_no_touching: bool = False,
        enforce_linear: bool = True,
    ):
        self.size = size
        self.ships_spec = ships_spec or DEFAULT_SHIPS
        self.enforce_no_touching = enforce_no_touching
        self.enforce_linear = enforce_linear

    def validate_board(self, board: BoardManager, strict_names_and_lengths: bool = True):
        reasons: List[str] = []

        if strict_names_and_lengths:
            spec_names = set(self.ships_spec.keys())
            board_names = set(board.ships.keys())
            if board_names != spec_names:
                reasons.append(f"Ship set mismatch. Expected {sorted(spec_names)}, got {sorted(board_names)}.")
            for name, ship in board.ships.items():
                expected_len = self.ships_spec.get(name)
                if expected_len is not None and len(ship.cells) != expected_len:
                    reasons.append(f"{name} length mismatch. Expected {expected_len}, got {len(ship.cells)}.")

        all_cells: List[Coord] = []
        for name, ship in board.ships.items():
            for x, y in ship.cells:
                if not (0 <= x < self.size and 0 <= y < self.size):
                    reasons.append(f"{name} has out-of-bounds cell {(x, y)}.")
            all_cells.extend(list(ship.cells))

        if len(all_cells) != len(set(all_cells)):
            reasons.append("Overlapping ships detected (duplicate occupied cells).")

        if self.enforce_linear:
            for name, ship in board.ships.items():
                if not self._is_straight_line(ship.cells, ship.length):
                    reasons.append(f"{name} is not a straight contiguous line of length {ship.length}.")

        if self.enforce_no_touching:
            occupied_by: Dict[Coord, str] = {}
            for name, ship in board.ships.items():
                for c in ship.cells:
                    occupied_by[c] = name
            for name, ship in board.ships.items():
                for x, y in ship.cells:
                    for nx in (x - 1, x, x + 1):
                        for ny in (y - 1, y, y + 1):
                            if (nx, ny) == (x, y):
                                continue
                            if 0 <= nx < self.size and 0 <= ny < self.size:
                                other = occupied_by.get((nx, ny))
                                if other and other != name:
                                    reasons.append(f"{name} touches {other} at {(x, y)} near {(nx, ny)}.")
                                    break

        return (len(reasons) == 0), reasons

    def _is_straight_line(self, cells: Set[Coord], length: int) -> bool:
        if len(cells) != length:
            return False
        xs = sorted({x for x, _ in cells})
        ys = sorted({y for _, y in cells})
        if len(ys) == 1:
            row = next(iter(ys))
            cols = sorted(x for x, y in cells if y == row)
            return cols[-1] - cols[0] + 1 == length and len(cols) == length
        if len(xs) == 1:
            col = next(iter(xs))
            rows = sorted(y for x, y in cells if x == col)
            return rows[-1] - rows[0] + 1 == length and len(rows) == length
        return False


def run_game(
    name: str,
    p1_id: str,
    p1_url: str,
    size: int = 10,
    max_turns: int = 100,
    ships_spec: Optional[Dict[str, int]] = None,
    enforce_no_touching: bool = False,
    startup_health_check_timeout_in_seconds: int = 10,
    board_generation_timeout_in_seconds: int = 1,
    shot_timeout_in_seconds: int = 1,
    console_mode: bool = False,
    seed: int = 42,
) -> GameResult:
    """
    Solo validator:
      - Calls the player's server for health and board setup (for protocol compatibility).
      - Generates a hidden random board locally.
      - Asks the player for shots until all ships are sunk or the max turn limit is hit.
    """
    sess = requests.Session()
    game_id = str(uuid.uuid4())

    p1 = RemotePlayer(id=p1_id, name=Name.PLAYER_1, base_url=p1_url)
    max_turns = max_turns

    # Health check player
    start_time = time.time()
    while True:
        p1_ok = health_check(sess, p1.base_url, timeout=1)
        if p1_ok:
            break
        if time.time() - start_time > startup_health_check_timeout_in_seconds:
            return GameResult(
                name=name,
                game_id=game_id,
                turns=0,
                max_turns=max_turns,
                game_result=f"Player health check failed after {startup_health_check_timeout_in_seconds:.2f} seconds",
                p1=p1,
                board={"ships": {}, "misses": []},
                board_size=size,
            )
        time.sleep(1)

    # Ask the player to create/return its board to establish game session state.
    try:
        init_board(
            sess,
            p1.base_url,
            game_id,
            size,
            timeout=board_generation_timeout_in_seconds,
        )
    except requests.Timeout:
        return GameResult(
            name=name,
            game_id=game_id,
            turns=0,
            max_turns=max_turns,
            game_result="Player board generation timed out",
            p1=p1,
            board={"ships": {}, "misses": []},
            board_size=size,
        )
    except Exception as e:
        return GameResult(
            name=name,
            game_id=game_id,
            turns=0,
            max_turns=max_turns,
            game_result=f"Player board generation failed: {e}",
            p1=p1,
            board={"ships": {}, "misses": []},
            board_size=size,
        )

    # Hidden target board
    hidden_board = Board(size=size, ships_spec=ships_spec or DEFAULT_SHIPS, seed=seed)
    hidden_board.place_ships_randomly()
    target_board = BoardManager.from_payload(hidden_board.to_payload())

    # Validate board (sanity check)
    v = Validator(
        size=size,
        ships_spec=ships_spec or DEFAULT_SHIPS,
        enforce_no_touching=enforce_no_touching,
    )
    ok, reasons = v.validate_board(target_board, strict_names_and_lengths=True)
    if not ok:
        return GameResult(
            name=name,
            game_id=game_id,
            turns=0,
            max_turns=max_turns,
            game_result=f"Validator board invalid: {reasons}",
            p1=p1,
            board={"ships": {}, "misses": []},
            board_size=size,
        )

    turn = 0

    def board_to_log(board: BoardManager) -> Dict[str, Any]:
        ships_payload = {}
        for name, ship in board.ships.items():
            ships_payload[name] = {
                "cells": sorted(list(ship.cells)),
                "hits": sorted(list(ship.hits)),
            }
        return {"ships": ships_payload, "misses": sorted(list(board.misses))}

    while turn < max_turns:
        if console_mode:
            print("\033[2J\033[H", end="")
            print(f"Game ID: {game_id}\nTurn: {turn}\nNext: {p1.name}\n")
            print("--- Target Board (hidden) ---")
            print(target_board.render(reveal=True))
            time.sleep(0.3)

        turn += 1

        previous_result = None
        if p1.last_result is not None:
            previous_result = {"game_id": game_id, **p1.last_result}

        try:
            x, y = ask_next_move(sess, p1.base_url, game_id, previous_result, timeout=shot_timeout_in_seconds)
        except requests.Timeout:
            return GameResult(
                name=name,
                game_id=game_id,
                turns=turn - 1,
                max_turns=max_turns,
                game_result="Shot timed out",
                p1=p1,
                board=board_to_log(target_board),
                board_size=size,
            )
        except Exception as e:
            return GameResult(
                name=name,
                game_id=game_id,
                turns=turn - 1,
                max_turns=max_turns,
                game_result=f"Shot threw an exception: {e}",
                p1=p1,
                board=board_to_log(target_board),
                board_size=size,
            )

        # Bounds check
        if not (0 <= x < size and 0 <= y < size):
            return GameResult(
                name=name,
                game_id=game_id,
                turns=turn,
                max_turns=max_turns,
                game_result=f"Out-of-bounds shot {(x, y)}",
                p1=p1,
                board=board_to_log(target_board),
                board_size=size,
            )

        # No repeat shots
        if (x, y) in p1.shot_history:
            return GameResult(
                name=name,
                game_id=game_id,
                turns=turn,
                max_turns=max_turns,
                game_result=f"Duplicate shot at {(x, y)}",
                p1=p1,
                board=board_to_log(target_board),
                board_size=size,
            )
        p1.shot_history.add((x, y))

        # Apply shot
        hit, sunk_name = target_board.receive_shot(x, y)
        p1.last_result = {"x": x, "y": y, "hit": hit, "sunk": sunk_name}

        if target_board.all_ships_sunk():
            return GameResult(
                name=name,
                game_id=game_id,
                turns=turn,
                max_turns=max_turns,
                game_result=f"{Name.PLAYER_1.value} won",
                p1=p1,
                board=board_to_log(target_board),
                board_size=size,
            )

    # Max turns reached
    return GameResult(
        name=name,
        game_id=game_id,
        turns=max_turns,
        max_turns=max_turns,
        game_result=f"Max turns reached ({max_turns}) without sinking all ships",
        p1=p1,
        board=board_to_log(target_board),
        board_size=size,
    )


# -----------------------------
# Replay logic from a saved log
# -----------------------------
def _board_from_log_ships(ships_dict: Dict[str, Any], size: int) -> BoardManager:
    b = BoardManager(size=size)
    for name, s in ships_dict.items():
        cells = set((int(x), int(y)) for x, y in s.get("cells", []))
        ship = Ship(name=name, length=len(cells) or int(s.get("length", 0)), cells=cells)
        b.ships[name] = ship
        for c in cells:
            b.occupied[c] = name
    return b


def infer_board_size_from_log(log_obj: Dict[str, Any]) -> int:
    # Prefer explicit board_size if provided
    if "board_size" in log_obj:
        return int(log_obj["board_size"])

    maxx = maxy = -1

    # Solo log support: board -> ships
    board_log = log_obj.get("board")
    if board_log:
        for ship in board_log.get("ships", {}).values():
            for x, y in ship.get("cells", []):
                maxx = max(maxx, int(x))
                maxy = max(maxy, int(y))
        for x, y in board_log.get("misses", []):
            maxx = max(maxx, int(x))
            maxy = max(maxy, int(y))

    # Legacy duel log support: p1/p2
    for side_key in ("p1", "p2"):
        side = log_obj.get(side_key, {})
        # ship cells
        for ship in side.get("ships", {}).values():
            for x, y in ship.get("cells", []):
                maxx = max(maxx, int(x))
                maxy = max(maxy, int(y))
        # shots
        for x, y in side.get("shot_history", []):
            maxx = max(maxx, int(x))
            maxy = max(maxy, int(y))
    sz = max(maxx, maxy) + 1
    return sz if sz > 0 else 10


def replay_from_log(
    log_path: str,
    console_mode: bool = True,
    delay_seconds: float = 0.0,
) -> None:
    """Replay a completed match saved as JSON (.log) produced by this client.

    Args:
        log_path: path to the .log JSON file.
        console_mode: if True, renders boards every step with ANSI colors.
        delay_seconds: seconds to wait between frames. If 0, waits for Enter each turn.
    """
    with open(log_path, "r") as f:
        log_obj = json.load(f)

    solo_mode = bool(log_obj.get("board")) and not log_obj.get("p2")

    if solo_mode:
        size = int(log_obj.get("board_size") or infer_board_size_from_log(log_obj))
        board_log = log_obj.get("board", {})
        board = _board_from_log_ships(board_log.get("ships", {}), size=size)
        p1_meta = log_obj.get("p1", {})
        p1_moves = [(int(x), int(y)) for x, y in p1_meta.get("shot_history", [])]
        game_id = log_obj.get("game_id", "unknown")
        max_turns = log_obj.get("max_turns", "?")
        total_ships = len(board.ships)
        result_from_log = log_obj.get("game_result")

        def ships_remaining() -> int:
            return sum(1 for s in board.ships.values() if not s.is_sunk())

        def render_solo(turn: int, last_msg: str = "") -> str:
            sunk_count = total_ships - ships_remaining()
            lines = [
                f"Game ID: {game_id}",
                f"Turn: {turn} / {max_turns}",
                f"Ships Sunk: {sunk_count} / {total_ships}",
                "",
                "--- Target Board ---",
                board.render(reveal=True),
            ]
            if last_msg:
                lines.append("")
                lines.append(last_msg)
            return "\n".join(lines)

        def _wait():
            if delay_seconds and delay_seconds > 0:
                time.sleep(delay_seconds)
            else:
                try:
                    input("Press Enter for next turn (or Ctrl+C to stop)... ")
                except EOFError:
                    pass

        turn = 0
        last_msg = ""
        for x, y in p1_moves:
            if console_mode:
                print("\033[2J\033[H", end="")
                print(render_solo(turn=turn, last_msg=last_msg))
            turn += 1
            hit, sunk = board.receive_shot(x, y)
            last_msg = f"Shot {x},{y} -> {'HIT' if hit else 'MISS'}" + (f" - sank {sunk}!" if sunk else "")
            if board.all_ships_sunk():
                if console_mode:
                    print("\033[2J\033[H", end="")
                    print(render_solo(turn=turn, last_msg=last_msg))
                    print(f"\n🎉 All ships sunk in {turn} turns (limit was {max_turns})!")
                break
            _wait()

        if not console_mode:
            print(
                f"Replay complete. Result: {result_from_log or 'unknown'}. Total turns: {log_obj.get('turns', turn)}."
            )
    else:
        # Legacy duel replay
        size = int(log_obj.get("board_size") or infer_board_size_from_log(log_obj))
        p1_meta = log_obj.get("p1", {})
        p2_meta = log_obj.get("p2", {})
        board1 = _board_from_log_ships(p1_meta.get("ships", {}), size=size)
        board2 = _board_from_log_ships(p2_meta.get("ships", {}), size=size)

        p1_moves = [(int(x), int(y)) for x, y in p1_meta.get("shot_history", [])]
        p2_moves = [(int(x), int(y)) for x, y in p2_meta.get("shot_history", [])]

        game_id = log_obj.get("game_id", "unknown")
        result_from_log = log_obj.get("game_result")

        def render(next_name: str, turn: int, last_msg: str = "") -> str:
            lines = [
                f"Game ID: {game_id}",
                f"Turn: {turn}",
                f"Next: {next_name}",
                "",
                f"--- {Name.PLAYER_1.value} Board (self-view) ---",
                board1.render(reveal=True),
                "",
                f"--- {Name.PLAYER_2.value} Board (self-view) ---",
                board2.render(reveal=True),
            ]
            if last_msg:
                lines.append("")  # blank line
                lines.append(last_msg)
            return "\n".join(lines)

        i1 = i2 = 0
        current = Name.PLAYER_1
        turn = 0
        last_msg = ""

        def _wait():
            if delay_seconds and delay_seconds > 0:
                time.sleep(delay_seconds)
            else:
                try:
                    input("Press Enter for next turn (or Ctrl+C to stop)... ")
                except EOFError:
                    # non-interactive env: just continue
                    pass

        while True:
            if console_mode:
                print("\033[2J\033[H", end="")  # clear screen
                print(render(next_name=current.value, turn=turn, last_msg=last_msg))

            if current == Name.PLAYER_1:
                if i1 >= len(p1_moves):
                    break
                x, y = p1_moves[i1]
                i1 += 1
                hit, sunk = board2.receive_shot(x, y)
                last_msg = f"{Name.PLAYER_1.value} shoots {x},{y} -> {'HIT' if hit else 'MISS'}" + (
                    f", sank {sunk}" if sunk else ""
                )
                if board2.all_ships_sunk():
                    if console_mode:
                        print("\033[2J\033[H", end="")
                        print(render(next_name="-", turn=turn + 1, last_msg=last_msg))
                        print(f"\nWinner: {Name.PLAYER_1.value} in {turn + 1} turns.")
                    break
                current = Name.PLAYER_2
            else:
                if i2 >= len(p2_moves):
                    break
                x, y = p2_moves[i2]
                i2 += 1
                hit, sunk = board1.receive_shot(x, y)
                last_msg = f"{Name.PLAYER_2.value} shoots {x},{y} -> {'HIT' if hit else 'MISS'}" + (
                    f", sank {sunk}" if sunk else ""
                )
                if board1.all_ships_sunk():
                    if console_mode:
                        print("\033[2J\033[H", end="")
                        print(render(next_name="-", turn=turn + 1, last_msg=last_msg))
                        print(f"\nWinner: {Name.PLAYER_2.value} in {turn + 1} turns.")
                    break
                current = Name.PLAYER_1

            turn += 1
            _wait()

        if not console_mode:
            print(
                f"Replay complete. Result: {result_from_log or 'unknown'}. Total turns recorded: {log_obj.get('turns', turn)}."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validator client for remote Battleship")
    parser.add_argument("--game-name", default="default")
    parser.add_argument("--p1-id", default="p1")
    parser.add_argument("--p1", default="http://127.0.0.1:8001")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=100, help="Max turns before game ends")
    parser.add_argument("--console", action="store_true")
    parser.add_argument("--no-touching", action="store_true", help="Don't allow ships to touch")
    parser.add_argument("--replay", "--replay-log", dest="replay", help="Path to a saved .log (JSON) to replay")
    parser.add_argument("--speed", type=float, default=0.3, help="Seconds between frames in replay. 0 = step-by-step.")
    args = parser.parse_args()

    if args.replay:
        replay_from_log(args.replay, console_mode=args.console, delay_seconds=args.speed)
    else:
        run_game(
            name=args.game_name,
            p1_id=args.p1_id,
            p1_url=args.p1,
            size=args.size,
            max_turns=args.max_turns,
            console_mode=args.console,
            enforce_no_touching=args.no_touching,
        )
