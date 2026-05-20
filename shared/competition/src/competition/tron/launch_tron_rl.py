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
# Layer 2 in-sandbox model validation
#
# Intentionally silent. The only externally observable failure signal is the
# bare sentinel `MODEL_VALIDATION_FAILED` written to stderr via SystemExit,
# which the evaluation runner detects in the sandbox log. No per-check
# details, thresholds, or check names are ever emitted.
# ============================================================================

_VALIDATION_SENTINEL = "MODEL_VALIDATION_FAILED"
_MIN_PARAMETER_COUNT = 1_000
_MIN_WEIGHT_STD = 1e-6
_INPUT_CHANNELS = 5
_NUM_ACTIONS = 4


def _check_parameter_count(model: torch.jit.ScriptModule) -> bool:
    return sum(p.numel() for p in model.parameters()) >= _MIN_PARAMETER_COUNT


def _check_parameter_distribution(model: torch.jit.ScriptModule) -> bool:
    chunks = [p.detach().cpu().numpy().flatten() for p in model.parameters()]
    if not chunks:
        return False
    w = np.concatenate(chunks)
    if np.any(np.isnan(w)) or np.any(np.isinf(w)):
        return False
    if np.allclose(w, 0, atol=1e-10):
        return False
    if np.allclose(w, w[0], atol=1e-10):
        return False
    return float(np.std(w)) >= _MIN_WEIGHT_STD


def _check_output_shape(model: torch.jit.ScriptModule, device: torch.device, board) -> bool:
    h, w = board
    inp = torch.rand(1, _INPUT_CHANNELS, h, w, device=device)
    try:
        with torch.no_grad():
            out = model(inp)
    except Exception:
        return False
    return out.flatten().numel() >= _NUM_ACTIONS


def _check_output_variation(model: torch.jit.ScriptModule, device: torch.device, board) -> bool:
    h, w = board
    inputs = [
        torch.zeros(1, _INPUT_CHANNELS, h, w, device=device),
        torch.ones(1, _INPUT_CHANNELS, h, w, device=device),
        torch.rand(1, _INPUT_CHANNELS, h, w, device=device),
    ]
    outputs = []
    with torch.no_grad():
        for inp in inputs:
            try:
                outputs.append(model(inp).cpu().numpy().flatten()[:_NUM_ACTIONS])
            except Exception:
                return False
    if len(outputs) != len(inputs):
        return False
    return not all(np.allclose(outputs[0], o, atol=1e-6) for o in outputs[1:])


def _check_gradient_flow(model: torch.jit.ScriptModule, device: torch.device, board) -> bool:
    h, w = board
    inp = torch.rand(1, _INPUT_CHANNELS, h, w, device=device)
    params = list(model.parameters())
    if not params:
        return False
    for p in params:
        p.requires_grad_(True)
    try:
        loss = model(inp).sum()
        loss.backward()
        live = 0
        for p in params:
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                live += p.numel()
        return live > 0
    except Exception:
        return False
    finally:
        for p in params:
            p.requires_grad_(False)
            p.grad = None


def _check_output_sensitivity(model: torch.jit.ScriptModule, device: torch.device, board) -> bool:
    h, w = board
    torch.manual_seed(42)
    base_input = torch.rand(1, _INPUT_CHANNELS, h, w, device=device)
    epsilons = (1e-3, 1e-2, 1e-1)
    diffs = []
    with torch.no_grad():
        try:
            base_out = model(base_input).cpu().numpy().flatten()[:_NUM_ACTIONS]
            for eps in epsilons:
                torch.manual_seed(0)
                perturbed = base_input + eps * torch.randn_like(base_input)
                perturbed_out = model(perturbed).cpu().numpy().flatten()[:_NUM_ACTIONS]
                diffs.append(float(np.abs(base_out - perturbed_out).mean()))
        except Exception:
            return False
    if not diffs:
        return False
    smallest, largest = diffs[0], diffs[-1]
    if largest < 1e-8:
        return False
    if smallest < 1e-10 and largest > 0.01:
        return False
    return True


def _validate_model(model: torch.jit.ScriptModule, device: torch.device, board: tuple[int, int]) -> None:
    """Run all Layer 2 checks. Raises SystemExit with the bare sentinel on any failure.

    `board` is the (height, width) the competition will actually play on, passed from the runner
    """
    if not _check_parameter_count(model):
        raise SystemExit(_VALIDATION_SENTINEL)
    if not _check_parameter_distribution(model):
        raise SystemExit(_VALIDATION_SENTINEL)
    if not _check_output_shape(model, device, board):
        raise SystemExit(_VALIDATION_SENTINEL)
    if not _check_output_variation(model, device, board):
        raise SystemExit(_VALIDATION_SENTINEL)
    if not _check_gradient_flow(model, device, board):
        raise SystemExit(_VALIDATION_SENTINEL)
    if not _check_output_sensitivity(model, device, board):
        raise SystemExit(_VALIDATION_SENTINEL)


# ============================================================================
# FastAPI Application
# ============================================================================


def make_app(model_path: str, board_height: int, board_width: int) -> FastAPI:
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

    _validate_model(model, device, (board_height, board_width))

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
    parser.add_argument("--board-height", type=int, required=True, help="Tron board height for this round")
    parser.add_argument("--board-width", type=int, required=True, help="Tron board width for this round")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        exit(1)

    app = make_app(model_path=args.model, board_height=args.board_height, board_width=args.board_width)
    uvicorn.run(app, host=args.host, port=args.port)
