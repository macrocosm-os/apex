import torch
import numpy as np
from typing import List, Dict


def encode_state(history: List[Dict], size: int = 10) -> torch.Tensor:
    """
    Encodes the game state into a tensor for the CNN.

    Args:
        history: List of history events from PlayerSession.
                 Events are dicts with keys like 'type', 'x', 'y', 'hit', 'sunk'.
        size: Board size (default 10).

    Returns:
        torch.Tensor: Shape (3, size, size).
            Channel 0: 1.0 if cell was shot at (tried), 0.0 otherwise.
            Channel 1: 1.0 if cell was a HIT, 0.0 otherwise.
            Channel 2: 1.0 if cell belongs to a SUNK ship, 0.0 otherwise.
    """
    # Initialize 3 channels
    board_tensor = np.zeros((3, size, size), dtype=np.float32)

    # We also need to track which shots belong to which sunk ship if possible,
    # but the history only gives us the "sunk" status on the shot that sunk it.
    # However, standard battleship rules say if a ship is sunk, all its hits are sunk.
    # But we don't know WHICH hits belong to that ship unless we tracked it or infer it.
    # For simplicity in this version:
    # - We mark the 'sunk' shot's location.
    # - Ideally we would perform connected component analysis or simple adjacency check
    #   to mark the whole ship as sunk.
    #   Since ships are linear, we can backtrack from the sunk shot along hits.

    # Let's do a first pass to populate hits and shots
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

        # Some history items might be just "shot" (the intent), but "result" has the outcome.
        # baseline.py appends "shot" then later "result" (which is from previous turn).
        # Actually baseline.py flow:
        # 1. /next-move called.
        # 2. if req.result provided (from previous shot), record it as "result".
        # 3. pick new shot, record as "shot".
        # So "result" entries are what we care about for board state.
        # "shot" entries without "result" are pending shots (usually just the last one if any).

    # Channel 2: Sunk inference
    # This is heuristic because we don't know for sure which hits belong to the sunk ship
    # without more complex logic, but usually they are adjacent.
    # A simple robust way for RL: just mark the sunk coordinate.
    # A better way: BFS/DFS from sunk coordinate to find connected hits and mark them as sunk.

    for sx, sy in sunk_events:
        # Mark the shot that caused sinking
        board_tensor[2, sy, sx] = 1.0

        # Flood fill to mark connected hits as sunk
        stack = [(sx, sy)]
        visited = set([(sx, sy)])

        while stack:
            cx, cy = stack.pop()
            board_tensor[2, cy, cx] = 1.0

            # Check neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if (nx, ny) in hits and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        stack.append((nx, ny))

    return torch.from_numpy(board_tensor)
