# player_server.py
from typing import Dict, Tuple, List, Optional, Set, Any
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


Coord = Tuple[int, int]  # (x, y) with 0-based indexing

DEFAULT_SHIPS: Dict[str, int] = {
    "Carrier": 5,
    "Battleship": 4,
    "Cruiser": 3,
    "Submarine": 3,
    "Destroyer": 2,
}


class Ship(BaseModel):
    name: str
    length: int
    cells: Set[Coord]
    hits: Set[Coord] = Field(default_factory=set)

    def is_sunk(self) -> bool:
        return self.cells == self.hits


# ----------- Pydantic models (wire contract) -----------
class ShipModel(BaseModel):
    name: str
    cells: List[Tuple[int, int]]


class BoardModel(BaseModel):
    size: int
    ships: List[ShipModel]


class BoardRequest(BaseModel):
    game_id: str
    size: int = 10
    ships_spec: Optional[Dict[str, int]] = None


class BoardResponse(BaseModel):
    game_id: str
    board: BoardModel


class ResultModel(BaseModel):
    game_id: str
    x: int
    y: int
    hit: bool
    sunk: Optional[str] = None


class NextMoveRequest(BaseModel):
    game_id: str
    result: Optional[ResultModel] = None


class NextMoveResponse(BaseModel):
    x: int
    y: int


class EndGameModel(BaseModel):
    game_id: str
    winner: str


# ------------------------------------------
# Board - used to generate a random board
# ------------------------------------------


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


# ------------------------------------------
# Random, non-repeating shooter
# ------------------------------------------


class RandomShooter:
    """
    Random, non-repeating shooter that selects a new (x, y) each turn.
    We accept previous result for logging only (keeps selection purely random).
    """

    def __init__(self, size: int = 10, seed: Optional[int] = None):
        self.size = size
        self.rng = random.Random(seed)
        self._bag: List[Coord] = [(x, y) for y in range(size) for x in range(size)]
        self.rng.shuffle(self._bag)
        self.tried: Set[Coord] = set()
        self.history: List[Dict] = []
        self.last_shot: Optional[Coord] = None

    def next_shot(self, result: Optional[Dict] = None) -> Coord:
        if result is not None and self.last_shot is not None:
            self.history.append(
                {
                    "x": self.last_shot[0],
                    "y": self.last_shot[1],
                    "hit": bool(result.get("hit", False)),
                    "sunk": result.get("sunk"),
                }
            )
        while self._bag:
            x, y = self._bag.pop()
            if (x, y) not in self.tried:
                self.tried.add((x, y))
                self.last_shot = (x, y)
                return x, y
        raise StopIteration("No more positions to try")


# ----------- In-memory session per game_id -----------


def generate_board(size: int = 10, ships_spec: Optional[Dict[str, int]] = None) -> Board:
    b = Board(size=size, ships_spec=ships_spec)
    b.place_ships_randomly()
    return b


class PlayerSession(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    game_id: str
    size: int
    board: Board
    shooter: RandomShooter
    pending_result_for_logging: Optional[Dict] = None
    history: List[Dict] = Field(default_factory=list)


def make_app() -> FastAPI:
    app = FastAPI(title="Battleship Player API")
    SESSIONS: Dict[str, PlayerSession] = {}

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/board", response_model=BoardResponse)
    def board(req: BoardRequest):
        """
        Create or return the board for this game_id (idempotent).
        The validator is the *client* and will store the board and run the match.
        """
        size = req.size
        ships_spec = req.ships_spec or DEFAULT_SHIPS
        sess = SESSIONS.get(req.game_id)
        if sess is None:
            board = generate_board(size=size, ships_spec=ships_spec)
            shooter = RandomShooter(size=size)
            sess = PlayerSession(game_id=req.game_id, size=size, board=board, shooter=shooter)
            SESSIONS[req.game_id] = sess
        return {"game_id": req.game_id, "board": sess.board.to_payload()}

    @app.post("/next-move", response_model=NextMoveResponse)
    def next_move(req: NextMoveRequest):
        """
        Return the next random, never-before-tried (x, y).
        Uses the pending result passed earlier via /result (if any) for logging.
        """
        sess = SESSIONS.get(req.game_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="Unknown game_id")

        # Store result of previous shot for logging
        if req.result is not None:
            sess.pending_result_for_logging = {
                "x": req.result.x,
                "y": req.result.y,
                "hit": req.result.hit,
                "sunk": req.result.sunk,
            }
            sess.history.append({"type": "result", **sess.pending_result_for_logging})

        # Get next shot
        x, y = sess.shooter.next_shot(sess.pending_result_for_logging)
        sess.pending_result_for_logging = None
        sess.history.append({"type": "shot", "x": x, "y": y})
        return {"x": x, "y": y}

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(make_app(), host="0.0.0.0", port=args.port)
