#!/usr/bin/env python
"""
Tron RL Player API using TorchScript Models

A FastAPI server that implements the Tron player protocol using a
TorchScript-exported RL model for action selection.

This is the launcher script that runs inside the sandbox. Miners submit
TorchScript models; this script loads the model and serves the HTTP API.

Models must be exported via torch.jit.trace() or torch.jit.script().
Expected model input:  tensor of shape (1, 5, height, width)
Expected model output: tensor of shape (4,) — Q-values/logits for [UP, RIGHT, DOWN, LEFT]
"""

import os
from typing import Any, Dict, List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================================
# State Encoding
# ============================================================================

EMPTY = 0
WALL = 1
PLAYER_TRAIL_START = 2


def encode_state(
    grid: List[List[int]],
    player_id: int,
    my_position: List[int],
    my_direction: int,
    opponent_positions: List[List[int]],
    opponent_alive: List[bool],
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Encode the game grid into a tensor for the model.

    Channels:
        0: Walls           (1.0 where grid == WALL)
        1: Your trail       (1.0 where grid == player_id + 2)
        2: Opponent trail   (1.0 where any opponent trail)
        3: Your head        (1.0 at your current position)
        4: Opponent heads   (1.0 at each living opponent position)

    Returns:
        torch.Tensor of shape (1, 5, height, width)
    """
    grid_arr = np.array(grid, dtype=np.int32)
    state = np.zeros((5, height, width), dtype=np.float32)

    state[0] = (grid_arr == WALL).astype(np.float32)
    state[1] = (grid_arr == PLAYER_TRAIL_START + player_id).astype(np.float32)

    # Opponent trails: any trail value that isn't ours
    opponent_mask = np.zeros_like(grid_arr, dtype=np.float32)
    for oid in range(8):  # max 8 players
        if oid != player_id:
            opponent_mask += (grid_arr == PLAYER_TRAIL_START + oid).astype(np.float32)
    state[2] = np.clip(opponent_mask, 0.0, 1.0)

    # Head positions
    my_y, my_x = my_position
    if 0 <= my_y < height and 0 <= my_x < width:
        state[3, my_y, my_x] = 1.0

    for i, (opp_pos, alive) in enumerate(zip(opponent_positions, opponent_alive)):
        if alive:
            oy, ox = opp_pos
            if 0 <= oy < height and 0 <= ox < width:
                state[4, oy, ox] = 1.0

    return torch.from_numpy(state).unsqueeze(0)  # (1, 5, H, W)


# ============================================================================
# API Models
# ============================================================================


class GameRequest(BaseModel):
    game_id: str
    player_id: int
    config: Dict[str, Any]
    grid: List[List[int]]
    your_position: List[int]
    your_direction: int
    opponent_positions: List[List[int]]


class MoveRequest(BaseModel):
    game_id: str
    step: int
    grid: List[List[int]]
    your_position: List[int]
    your_direction: int
    your_alive: bool
    opponent_positions: List[List[int]]
    opponent_alive: List[bool]
    valid_actions: List[int]


class MoveResponse(BaseModel):
    action: int


class GameSession(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    game_id: str
    player_id: int
    width: int
    height: int


# ============================================================================
# FastAPI Application
# ============================================================================


def make_app(model_path: str) -> FastAPI:
    app = FastAPI(title="Tron RL Player API")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model path: {model_path}")

    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print("TorchScript model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

    SESSIONS: Dict[str, GameSession] = {}

    @app.get("/health")
    def health() -> Dict:
        return {
            "ok": True,
            "model_path": model_path,
            "device": str(device),
        }

    @app.post("/game")
    def game(req: GameRequest) -> Dict:
        SESSIONS[req.game_id] = GameSession(
            game_id=req.game_id,
            player_id=req.player_id,
            width=req.config.get("width", 24),
            height=req.config.get("height", 24),
        )
        return {"ok": True}

    @app.post("/move", response_model=MoveResponse)
    def move(req: MoveRequest) -> MoveResponse:
        sess = SESSIONS.get(req.game_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="Unknown game_id")

        if not req.your_alive or not req.valid_actions:
            # Dead or no valid actions — return current direction as fallback
            return MoveResponse(action=req.your_direction)

        # Encode state and run inference
        state_tensor = encode_state(
            grid=req.grid,
            player_id=sess.player_id,
            my_position=req.your_position,
            my_direction=req.your_direction,
            opponent_positions=req.opponent_positions,
            opponent_alive=req.opponent_alive,
            height=sess.height,
            width=sess.width,
        ).to(device)

        with torch.no_grad():
            output = model(state_tensor)

        # Handle different output shapes
        if output.dim() == 1:
            q_values = output
        elif output.dim() == 2:
            q_values = output.squeeze(0)
        else:
            q_values = output.flatten()[:4]

        # Mask invalid actions with -inf, pick best valid action
        q_np = q_values.cpu().numpy()[:4]
        masked = np.full(4, float("-inf"))
        for a in req.valid_actions:
            if 0 <= a < 4:
                masked[a] = q_np[a]

        best_action = int(np.argmax(masked))

        # Fallback if all masked out
        if masked[best_action] == float("-inf"):
            best_action = req.valid_actions[0] if req.valid_actions else req.your_direction

        return MoveResponse(action=best_action)

    return app


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Tron RL player API using TorchScript model")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, required=True, help="Path to TorchScript model (.pt)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        exit(1)

    app = make_app(model_path=args.model)
    uvicorn.run(app, host=args.host, port=args.port)
