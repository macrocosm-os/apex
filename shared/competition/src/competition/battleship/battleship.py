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
    name: str | None = None
    game_id: str
    winner: Name | None = None
    loser: Name | None = None
    turns: int = 0
    game_result: str = ""
    p1: RemotePlayer
    p2: RemotePlayer


def health_check(session: requests.Session, player_url: str, timeout: int = 1) -> bool:
    try:
        resp = session.get(f"{player_url}/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json()["ok"]
    except Exception as e:
        return False


def fetch_board(session: requests.Session, player_url: str, game_id: str, size: int, timeout: int = 1):
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
    p2_id: str,
    p1_url: str,
    p2_url: str,
    size: int = 10,
    ships_spec: Optional[Dict[str, int]] = None,
    repeat_on_hit: bool = False,
    enforce_no_touching: bool = False,
    startup_health_check_timeout_in_seconds: int = 10,
    board_generation_timeout_in_seconds: int = 1,
    shot_timeout_in_seconds: int = 1,
    starting_player: str = "p1",
    console_mode: bool = False,
) -> GameResult:
    """
    Validator as a *client*:
      - Calls each player's server to get their board.
      - Validates boards locally.
      - Alternates calling /result (for prior shot) and /next-move (for new shot).
      - Applies shots to the opponent's board and declares the winner.
    """
    sess = requests.Session()
    game_id = str(uuid.uuid4())

    p1 = RemotePlayer(id=p1_id, name=Name.PLAYER_1, base_url=p1_url)
    p2 = RemotePlayer(id=p2_id, name=Name.PLAYER_2, base_url=p2_url)
    game_result = GameResult(
        name=name,
        game_id=game_id,
        winner=None,
        loser=None,
        turns=0,
        game_result="",
        p1=p1,
        p2=p2,
    )

    # Health check players
    start_time = time.time()
    p1_ok = False
    p2_ok = False
    while True:
        if not p1_ok:
            p1_ok = health_check(sess, p1.base_url, timeout=1)
        if not p2_ok:
            p2_ok = health_check(sess, p2.base_url, timeout=1)

        if p1_ok and p2_ok:
            break
        if time.time() - start_time > startup_health_check_timeout_in_seconds:
            if not p1_ok:
                game_result.winner = Name.PLAYER_2 if p2_ok else None
                game_result.loser = Name.PLAYER_1
                game_result.game_result = (
                    f"Player 1 health check failed after {startup_health_check_timeout_in_seconds:.2f} seconds"
                )
                return game_result
            if not p2_ok:
                game_result.winner = Name.PLAYER_1 if p1_ok else None
                game_result.loser = Name.PLAYER_2
                game_result.game_result = (
                    f"Player 2 health check failed after {startup_health_check_timeout_in_seconds:.2f} seconds"
                )
        time.sleep(1)

    # Fetch boards (players are servers)
    try:
        b1_payload = fetch_board(sess, p1.base_url, game_id, size, timeout=board_generation_timeout_in_seconds)
        board1 = BoardManager.from_payload(b1_payload)
        p1.ships = board1.ships
    except requests.Timeout:
        game_result.winner = Name.PLAYER_2
        game_result.loser = Name.PLAYER_1
        game_result.game_result = "Player 1 board generation timed out"
        return game_result
    except Exception as e:
        game_result.winner = Name.PLAYER_2
        game_result.loser = Name.PLAYER_1
        game_result.game_result = "Player 1 board generation threw an exception"
        return game_result

    try:
        b2_payload = fetch_board(sess, p2.base_url, game_id, size, timeout=board_generation_timeout_in_seconds)
        board2 = BoardManager.from_payload(b2_payload)
        p2.ships = board2.ships
    except requests.Timeout:
        game_result.winner = Name.PLAYER_1
        game_result.loser = Name.PLAYER_2
        game_result.game_result = "Player 2 board generation timed out"
        return game_result
    except Exception as e:
        game_result.winner = Name.PLAYER_1
        game_result.loser = Name.PLAYER_2
        game_result.game_result = "Player 2 board generation threw an exception"
        return game_result

    # Validate
    v = Validator(
        size=size,
        ships_spec=ships_spec or DEFAULT_SHIPS,
        enforce_no_touching=enforce_no_touching,
    )
    ok1, reasons1 = v.validate_board(board1, strict_names_and_lengths=True)
    ok2, reasons2 = v.validate_board(board2, strict_names_and_lengths=True)
    if not ok1 and not ok2:
        game_result.winner = None
        game_result.loser = None
        game_result.game_result = f"Player 1: {reasons1}\nPlayer 2: {reasons2}"
        return game_result
    if not ok1 and ok2:
        game_result.winner = Name.PLAYER_2
        game_result.loser = Name.PLAYER_1
        game_result.game_result = f"Player 1: {reasons1}"
        return game_result
    if not ok2 and ok1:
        game_result.winner = Name.PLAYER_1
        game_result.loser = Name.PLAYER_2
        game_result.game_result = f"Player 2: {reasons2}"
        return game_result

    # Game loop
    if starting_player == "p1":
        current, opponent = p1, p2
        target_board = board2
    else:
        current, opponent = p2, p1
        target_board = board1

    turn = 0

    def render():
        lines = [
            f"Game ID: {game_id}",
            f"Turn: {turn}",
            f"Next: {current.name}",
            "",
            f"--- {p1.name} Board (self-view) ---",
            board1.render(reveal=True),
            "",
            f"--- {p2.name} Board (self-view) ---",
            board2.render(reveal=True),
        ]
        return "\n".join(lines)

    while True:
        if console_mode:
            print("\033[2J\033[H", end="")  # clear + home
            print(render())
            time.sleep(0.3)

        turn += 1

        # Send result of the *previous* shot to the current shooter (if any)
        previous_result = None
        if current.last_result is not None:
            body = {"game_id": game_id, **current.last_result}
            # We'll instead send the previous result to the next move request
            # post_result(sess, current.base_url, body)
            previous_result = body

        # Ask current shooter for next shot
        try:
            x, y = ask_next_move(sess, current.base_url, game_id, previous_result, timeout=shot_timeout_in_seconds)
        except requests.Timeout:
            if current == p1:
                game_result.winner = Name.PLAYER_2
                game_result.loser = Name.PLAYER_1
                game_result.game_result = "Player 1 shot timed out"
                return game_result
            else:
                game_result.winner = Name.PLAYER_1
                game_result.loser = Name.PLAYER_2
                game_result.game_result = "Player 2 shot timed out"
                return game_result
        except Exception as e:
            if current == p1:
                game_result.winner = Name.PLAYER_2
                game_result.loser = Name.PLAYER_1
                game_result.game_result = "Player 1 shot threw an exception"
                return game_result
            else:
                game_result.winner = Name.PLAYER_1
                game_result.loser = Name.PLAYER_2
                game_result.game_result = "Player 2 shot threw an exception"
                return game_result

        # Enforce bounds & no-repeat (validator-side guard)
        if not (0 <= x < size and 0 <= y < size):
            raise SystemExit(f"{current.name} proposed out-of-bounds shot {(x, y)}.")
        # if (x, y) in current.shot_history:
        #     raise SystemExit(f"{current.name} repeated shot {(x, y)}. A player must never repeat a shot.")
        current.shot_history.add((x, y))

        # Apply shot to opponent's board
        hit, sunk_name = target_board.receive_shot(x, y)

        # Prepare result for *current* shooter (to be delivered at their next turn)
        current.last_result = {"x": x, "y": y, "hit": hit, "sunk": sunk_name}

        # Victory check
        if target_board.all_ships_sunk():
            winner = Name.PLAYER_1 if current == p1 else Name.PLAYER_2
            loser = Name.PLAYER_1 if current == p2 else Name.PLAYER_2

            if console_mode:
                print("\033[2J\033[H", end="")
                print(render())
                print(f"\nWinner: {winner} in {turn} turns. (Loser: {loser})")
            game_result.winner = winner
            game_result.loser = loser
            game_result.turns = turn
            game_result.game_result = f"{winner.value} won"
            return game_result

        # Turn switch
        if repeat_on_hit and hit:
            continue  # same shooter goes again
        else:
            current, opponent = opponent, current
            target_board = board1 if target_board is board2 else board2


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
    maxx = maxy = -1
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

    # Build boards
    size = int(log_obj.get("board_size") or infer_board_size_from_log(log_obj))
    p1_meta = log_obj.get("p1", {})
    p2_meta = log_obj.get("p2", {})
    board1 = _board_from_log_ships(p1_meta.get("ships", {}), size=size)
    board2 = _board_from_log_ships(p2_meta.get("ships", {}), size=size)

    p1_moves = [(int(x), int(y)) for x, y in p1_meta.get("shot_history", [])]
    p2_moves = [(int(x), int(y)) for x, y in p2_meta.get("shot_history", [])]

    game_id = log_obj.get("game_id", "unknown")
    winner_from_log = log_obj.get("winner")

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

    # helper for pacing
    def _wait():
        if delay_seconds and delay_seconds > 0:
            time.sleep(delay_seconds)
        else:
            try:
                input("Press Enter for next turn (or Ctrl+C to stop)... ")
            except EOFError:
                # non-interactive env: just continue
                pass

    # main loop
    while True:
        # show current boards
        if console_mode:
            print("\033[2J\033[H", end="")  # clear screen
            print(render(next_name=current.value, turn=turn, last_msg=last_msg))

        # who shoots
        if current == Name.PLAYER_1:
            if i1 >= len(p1_moves):
                # no more recorded moves for P1; end
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

    # final line if not in console
    if not console_mode:
        print(
            f"Replay complete. Winner (from log): {winner_from_log}. Total turns recorded: {log_obj.get('turns', turn)}."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validator client for remote Battleship")
    parser.add_argument("--game-name", default="default")
    parser.add_argument("--p1-id", default="p1")
    parser.add_argument("--p2-id", default="p2")
    parser.add_argument("--p1", default="http://127.0.0.1:8001")
    parser.add_argument("--p2", default="http://127.0.0.1:8002")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--repeat-on-hit", action="store_true")
    parser.add_argument("--console", action="store_true")
    parser.add_argument("--no-touching", action="store_true", help="Don't allow ships to touch")
    parser.add_argument("--replay", "--replay-log", dest="replay", help="Path to a saved .log (JSON) to replay")
    parser.add_argument("--speed", type=float, default=0.3, help="Seconds between frames in replay. 0 = step-by-step.")
    args = parser.parse_args()

    if args.replay:
        replay_from_log(args.replay, console_mode=args.console, delay_seconds=args.speed)
    else:
        run_game(
            name=args.match_name,
            p1_id=args.p1_id,
            p2_id=args.p2_id,
            p1_url=args.p1,
            p2_url=args.p2,
            size=args.size,
            repeat_on_hit=args.repeat_on_hit,
            console_mode=args.console,
            enforce_no_touching=args.no_touching,
        )
