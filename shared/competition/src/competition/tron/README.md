# Tron DUEL Competition

Two players control light-cycles on a grid. Each cycle moves continuously and leaves a trail behind it. Collide with a wall, any trail, or another player's head and you die. Last player alive wins.

## Game Rules

### Grid

- Default size: **32x32** with a 1-cell wall border (30x30 playable area).
- Cell values: `0` = empty, `1` = wall, `2` = Player 0's trail, `3` = Player 1's trail.
- Walls are deadly. Moving off the grid or into a wall kills you.

### Movement

- Each step, both players simultaneously choose a direction: `0=UP, 1=RIGHT, 2=DOWN, 3=LEFT`.
- You cannot reverse (e.g. if facing RIGHT, you cannot choose LEFT). Attempting to reverse keeps your current direction.
- After choosing, both players move one cell in their chosen direction. The cell you leave becomes part of your trail.

### Collisions

All collisions are checked simultaneously after both players move:

| Collision type | What happens |
|---|---|
| **Wall** | Moving into a wall cell or off the grid kills you. |
| **Trail** | Moving into any trail cell (yours or opponent's) kills you. |
| **Head-on** | Both players moving to the same cell kills both (draw). |

### Winning

- **Last alive wins.** If one player dies and the other survives, the survivor wins.
- **Draw** if both die on the same step (head-on collision or both hit trails simultaneously).
- **Draw** if `max_steps` (default 500) is reached with both players still alive.

### Spawn Positions

Players spawn in opposite corners:
- Player 0: top-left area `(2, 2)`, facing DOWN
- Player 1: bottom-right area `(29, 29)`, facing UP

## Submission Format

Miners submit **TorchScript models** (`.pt` files) exported via `torch.jit.trace()` or `torch.jit.script()`.

### Model Interface

**Input:** Tensor of shape `(1, 5, height, width)` with these channels:

| Channel | Contents |
|---|---|
| 0 | Walls (`1.0` where wall, `0.0` elsewhere) |
| 1 | Your trail (`1.0` where your trail exists) |
| 2 | Opponent trail (`1.0` where opponent trail exists) |
| 3 | Your head position (`1.0` at your current cell) |
| 4 | Opponent head position (`1.0` at opponent's current cell) |

**Output:** Tensor of shape `(4,)` representing Q-values or logits for `[UP, RIGHT, DOWN, LEFT]`.

The launcher picks the highest-valued **valid** action (invalid actions like reversing are masked out). If all valid actions are masked or the model fails, the player continues in its current direction.

### Allowed Packages

See `dockerfiles/requirements.txt` for the exact versions available in the sandbox:

- `torch` (CPU-only)
- `fastapi`
- `uvicorn`
- `numpy`
- `pydantic`

## Scoring

This is a **DUEL** competition run as a **single-elimination bracket**. Submissions are randomly seeded at the start of the round; each match's loser is eliminated and the winner advances. With N submissions the round plays at most N−1 matches (vs. N·(N−1)/2 in a round-robin).

### Per-duel scoring

Each duel plays **3 games** with alternating spawn positions for fairness:

| Game | Submission A role | Submission B role |
|---|---|---|
| 1 | Player 0 (top-left) | Player 1 (bottom-right) |
| 2 | Player 1 (bottom-right) | Player 0 (top-left) |
| 3 | Player 0 (top-left) | Player 1 (bottom-right) |

Per-game scores:

| Outcome | Score |
|---|---|
| Win | `1.0` |
| Loss | `0.0` |
| Draw | `0.5` |
| API error / crash | `0.0` |

The per-duel score is the **average** across the 3 games (i.e. each submission's win rate over the 3 games). The submission with the higher per-duel score wins the match and advances; the loser is eliminated. If the per-duel score ties (legitimate draws or double-failure), the lower bracket seed advances.

### Round scoring

A submission's round score is its **win rate across the matches it actually played** in the bracket. Eliminated submissions stop accumulating matches at the round they're knocked out, so deeper survivors have more matches contributing to their score. The bracket winner is the last submission standing.

### Multiple submissions

If you submit multiple models in one round, only the **latest submission** is used for evaluation. Previous versions are automatically replaced.

## Game Configuration

| Parameter | Value |
|---|---|
| Grid size | 32x32 |
| Max steps | 500 |
| Wall wrap | No (walls are deadly) |
| Trail fade | 0 (permanent trails) |
| Items | Disabled |
| Move timeout | 0.5 seconds |
| Spawn mode | Corners |
