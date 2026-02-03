#!/usr/bin/env python
"""
Battleship RL Player API using TorchScript Models

A FastAPI server that implements the Battleship player protocol using any
TorchScript-exported RL model for shot selection. The model should output
Q-values or logits for all board positions.

This is a battleship-specific wrapper that:
  - Implements the standard Battleship API protocol (/health, /board, /next-move)
  - Uses state encoding to convert game history to tensor observations
  - Runs inference on a generic TorchScript model
  - Converts model output to shot coordinates

Models must be exported via torch.jit.trace() or torch.jit.script().
Expected model input: tensor of shape (3, board_size, board_size)
Expected model output: tensor of shape (board_size * board_size,) or (board_size, board_size)
"""

import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================================
# State Encoding Logic (from state_encoder.py)
# ============================================================================


def encode_state(history: List[Dict], size: int = 10) -> torch.Tensor:
    """
    Encodes the game state into a tensor for the model.

    Args:
        history: List of history events. Events are dicts with keys like
                 'type', 'x', 'y', 'hit', 'sunk'.
        size: Board size (default 10).

    Returns:
        torch.Tensor: Shape (3, size, size).
            Channel 0: 1.0 if cell was shot at (tried), 0.0 otherwise.
            Channel 1: 1.0 if cell was a HIT, 0.0 otherwise.
            Channel 2: 1.0 if cell belongs to a SUNK ship, 0.0 otherwise.
    """
    # Initialize 3 channels
    board_tensor = np.zeros((3, size, size), dtype=np.float32)

    # Track shots and hits for sunk inference
    shots = set()
    hits = set()
    sunk_events = []

    for event in history:
        if event.get("type") == "result":
            x, y = event["x"], event["y"]
            if 0 <= x < size and 0 <= y < size:
                hit = event.get("hit", False)
                sunk = event.get("sunk")

                # Channel 0: Tried
                board_tensor[0, y, x] = 1.0
                shots.add((x, y))

                # Channel 1: Hit
                if hit:
                    board_tensor[1, y, x] = 1.0
                    hits.add((x, y))

                # Track sunk events to process after
                if sunk:
                    sunk_events.append((x, y))

    # Channel 2: Sunk inference via flood fill
    for sx, sy in sunk_events:
        board_tensor[2, sy, sx] = 1.0

        # Flood fill to mark connected hits as sunk
        stack = [(sx, sy)]
        visited = set([(sx, sy)])

        while stack:
            cx, cy = stack.pop()
            board_tensor[2, cy, cx] = 1.0

            # Check neighbors (up, down, left, right)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if (nx, ny) in hits and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        stack.append((nx, ny))

    return torch.from_numpy(board_tensor)


# ============================================================================
# Board Generation (from battleship baseline)
# ============================================================================

DEFAULT_SHIPS: Dict[str, int] = {
    "Carrier": 5,
    "Battleship": 4,
    "Cruiser": 3,
    "Submarine": 3,
    "Destroyer": 2,
}


class Ship:
    """Represents a ship on the board."""

    def __init__(self, name: str, cells: Set[Tuple[int, int]]):
        self.name = name
        self.cells = cells
        self.hits: Set[Tuple[int, int]] = set()

    @property
    def is_sunk(self) -> bool:
        return self.cells == self.hits


class Board:
    """Represents a player's board."""

    def __init__(self, size: int, ships: List[Ship]):
        self.size = size
        self.ships = ships

    def to_payload(self) -> Dict:
        """Convert board to JSON-serializable format."""
        return {"ships": {ship.name: list(ship.cells) for ship in self.ships}}


def generate_board(
    size: int = 10,
    ships_spec: Optional[Dict[str, int]] = None,
    no_touching: bool = False,
) -> Board:
    """Generate a random valid board with ships placed."""
    if ships_spec is None:
        ships_spec = DEFAULT_SHIPS

    occupied: Set[Tuple[int, int]] = set()
    ships: List[Ship] = []

    for ship_name, length in ships_spec.items():
        placed = False
        attempts = 0
        max_attempts = 1000

        while not placed and attempts < max_attempts:
            attempts += 1
            horizontal = random.choice([True, False])

            if horizontal:
                x = random.randint(0, size - length)
                y = random.randint(0, size - 1)
                cells = {(x + i, y) for i in range(length)}
            else:
                x = random.randint(0, size - 1)
                y = random.randint(0, size - length)
                cells = {(x, y + i) for i in range(length)}

            # Check for overlap
            if cells & occupied:
                continue

            # Check for touching (if enforced)
            if no_touching:
                neighbors = set()
                for cx, cy in cells:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < size and 0 <= ny < size:
                                neighbors.add((nx, ny))
                if neighbors & occupied:
                    continue

            # Place the ship
            occupied |= cells
            ships.append(Ship(name=ship_name, cells=cells))
            placed = True

        if not placed:
            raise RuntimeError(f"Could not place ship {ship_name} after {max_attempts} attempts")

    return Board(size=size, ships=ships)


# ============================================================================
# RL Shooter using TorchScript Model
# ============================================================================


class TorchScriptShooter:
    """Shot selector using a TorchScript RL model."""

    def __init__(self, size: int, model: torch.jit.ScriptModule, device: torch.device):
        self.size = size
        self.model = model
        self.device = device
        self.tried: Set[Tuple[int, int]] = set()
        self.history: List[Dict] = []

    def next_shot(self, previous_result: Optional[Dict] = None) -> Tuple[int, int]:
        """Get the next shot coordinate using the RL model.

        Args:
            previous_result: Result of the previous shot (if any)

        Returns:
            Tuple of (x, y) coordinates for the next shot
        """
        # Update history with previous result
        if previous_result is not None:
            self.history.append({"type": "result", **previous_result})

        # Encode state
        state_tensor = encode_state(self.history, self.size)

        # Add batch dimension and move to device
        state_tensor = state_tensor.unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(state_tensor)

        # Convert output to shot probabilities
        # Handle different output shapes
        if output.dim() == 1:
            # Shape: (board_size * board_size,)
            q_values = output
        elif output.dim() == 2:
            if output.shape[0] == 1:
                # Shape: (1, board_size * board_size) - batched
                q_values = output.squeeze(0)
            else:
                # Shape: (board_size, board_size) - grid format
                q_values = output.flatten()
        elif output.dim() == 3:
            # Shape: (1, board_size, board_size) - batched grid
            q_values = output.squeeze(0).flatten()
        else:
            raise ValueError(f"Unexpected model output shape: {output.shape}")

        # Mask already-tried positions with very negative values
        q_values_np = q_values.cpu().numpy()
        for tx, ty in self.tried:
            idx = ty * self.size + tx
            if idx < len(q_values_np):
                q_values_np[idx] = float("-inf")

        # Find the best valid position
        best_idx = np.argmax(q_values_np)

        # Handle case where all positions are tried (shouldn't happen in normal play)
        if q_values_np[best_idx] == float("-inf"):
            # Fallback: find any untried position
            for y in range(self.size):
                for x in range(self.size):
                    if (x, y) not in self.tried:
                        self.tried.add((x, y))
                        return x, y
            # All positions tried - this shouldn't happen
            return 0, 0

        # Convert index to coordinates
        x = best_idx % self.size
        y = best_idx // self.size

        self.tried.add((x, y))
        return x, y


# ============================================================================
# API Models
# ============================================================================


class BoardRequest(BaseModel):
    game_id: str
    size: int = 10
    ships_spec: Optional[Dict[str, int]] = None


class BoardResponse(BaseModel):
    game_id: str
    board: Dict


class ShotResult(BaseModel):
    x: int
    y: int
    hit: bool
    sunk: Optional[str] = None


class NextMoveRequest(BaseModel):
    game_id: str
    result: Optional[ShotResult] = None


class NextMoveResponse(BaseModel):
    x: int
    y: int


class PlayerSession(BaseModel):
    """Stores state for a single game session."""

    model_config = {"arbitrary_types_allowed": True}

    game_id: str
    size: int
    board: Any  # Board object
    shooter: Any  # TorchScriptShooter object
    history: List[Dict] = Field(default_factory=list)


# ============================================================================
# FastAPI Application
# ============================================================================


def make_app(model_path: str) -> FastAPI:
    """
    Create the FastAPI application for Battleship RL player.

    Args:
        model_path: Path to a TorchScript model file (.pt)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Battleship RL Player API",
        description="Battleship player using TorchScript RL model for shot selection",
        version="1.0.0",
    )

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model path: {model_path}")

    # Load TorchScript model
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print("TorchScript model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

    # Session storage
    SESSIONS: Dict[str, PlayerSession] = {}

    @app.get("/health")
    def health() -> Dict:
        """Health check endpoint."""
        return {
            "ok": True,
            "model_path": model_path,
            "device": str(device),
        }

    @app.post("/board", response_model=BoardResponse)
    def board(req: BoardRequest) -> BoardResponse:
        """
        Create or return the board for this game_id (idempotent).
        """
        sess = SESSIONS.get(req.game_id)
        if sess is None:
            size = req.size
            ships_spec = req.ships_spec or DEFAULT_SHIPS

            # Generate board
            player_board = generate_board(size=size, ships_spec=ships_spec)

            # Create shooter with RL model
            shooter = TorchScriptShooter(size=size, model=model, device=device)

            sess = PlayerSession(
                game_id=req.game_id,
                size=size,
                board=player_board,
                shooter=shooter,
            )
            SESSIONS[req.game_id] = sess

        return BoardResponse(game_id=req.game_id, board=sess.board.to_payload())

    @app.post("/next-move", response_model=NextMoveResponse)
    def next_move(req: NextMoveRequest) -> NextMoveResponse:
        """
        Return the next shot coordinate using the RL model.
        """
        sess = SESSIONS.get(req.game_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="Unknown game_id")

        # Convert result to dict format if provided
        previous_result = None
        if req.result is not None:
            previous_result = {
                "x": req.result.x,
                "y": req.result.y,
                "hit": req.result.hit,
                "sunk": req.result.sunk,
            }
            sess.history.append({"type": "result", **previous_result})

        # Get next shot from RL model
        x, y = sess.shooter.next_shot(previous_result)
        sess.history.append({"type": "shot", "x": x, "y": y})

        return NextMoveResponse(x=x, y=y)

    return app


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Battleship RL player API using TorchScript model")
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to run the server on (default: 8001)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to a TorchScript model file (.pt)",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        exit(1)

    app = make_app(model_path=args.model)
    uvicorn.run(app, host=args.host, port=args.port)
