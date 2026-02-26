import torch
import numpy as np
from typing import List, Dict, Any, Tuple


def encode_state(history: List[Dict], size: int = 10) -> torch.Tensor:
    """
    Encodes the game state into a tensor for the CNN.

    Args:
        history: List of history events from PlayerSession.
                 Events are dicts with keys like 'type', 'x', 'y', 'hit', 'sunk', and optionally 'sunk_cells'.
        size: Board size (default 10).

    Returns:
        torch.Tensor: Shape (3, size, size).
            Channel 0: 1.0 if cell was shot at (tried), 0.0 otherwise.
            Channel 1: 1.0 if cell was a HIT, 0.0 otherwise.
            Channel 2: 1.0 if cell belongs to a SUNK ship, 0.0 otherwise.
    """
    # Initialize 3 channels
    board_tensor = np.zeros((3, size, size), dtype=np.float32)

    # First pass: populate channel 0/1 and collect sunk events (with optional sunk_cells for touching ships)
    shots = set()
    hits = set()
    sunk_events: List[Tuple[Any, ...]] = []  # (x, y, sunk_cells?) per sunk

    for event in history:
        if event.get("type") == "result":
            x, y = event["x"], event["y"]
            if 0 <= x < size and 0 <= y < size:
                hit = event.get("hit", False)
                sunk = event.get("sunk")
                sunk_cells = event.get("sunk_cells")

                # Channel 0: Tried
                board_tensor[0, y, x] = 1.0
                shots.add((x, y))

                # Channel 1: Hit
                if hit:
                    board_tensor[1, y, x] = 1.0
                    hits.add((x, y))

                if sunk:
                    sunk_events.append((x, y, sunk_cells))

    # Channel 2: use exact sunk_cells when provided (handles touching ships), else flood-fill
    for item in sunk_events:
        sx, sy = item[0], item[1]
        sunk_cells = item[2] if len(item) > 2 else None
        if sunk_cells:
            for cell in sunk_cells:
                cx, cy = (cell[0], cell[1]) if isinstance(cell, (list, tuple)) else (cell[0], cell[1])
                if 0 <= cx < size and 0 <= cy < size:
                    board_tensor[2, cy, cx] = 1.0
        else:
            board_tensor[2, sy, sx] = 1.0
            stack = [(sx, sy)]
            visited = set([(sx, sy)])
            while stack:
                cx, cy = stack.pop()
                board_tensor[2, cy, cx] = 1.0
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        if (nx, ny) in hits and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            stack.append((nx, ny))

    return torch.from_numpy(board_tensor)
