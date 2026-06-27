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

| Collision type | What happens                                                |
| -------------- | ----------------------------------------------------------- |
| **Wall**       | Moving into a wall cell or off the grid kills you.          |
| **Trail**      | Moving into any trail cell (yours or opponent's) kills you. |
| **Head-on**    | Both players moving to the same cell kills both (draw).     |

### Game Termination

- **One survivor.** If one player dies and the other survives, the game ends.
- **Both die same step.** Head-on collision or simultaneous trail-kills end the game.
- **Timeout.** If `max_steps` (default 500) is reached, the game ends with both players alive.

Note: the per-game *score* depends on **how** the game ended, not just who survived — see [Scoring](#scoring) below. A passive win, a clean kill, a head-on collision, and a timeout all produce different scores.

### Spawn Positions

Players spawn in opposite corners (one cell inside the wall border):
- Player 0: top-left `(1, 1)`, facing DOWN
- Player 1: bottom-right `(30, 30)`, facing UP

## Submission Format

Miners submit **TorchScript models** (`.pt` files) exported via `torch.jit.trace()` or `torch.jit.script()`.

### Model Interface

**Input:** Tensor of shape `(1, 5, height, width)` with these channels:

| Channel | Contents                                                  |
| ------- | --------------------------------------------------------- |
| 0       | Walls (`1.0` where wall, `0.0` elsewhere)                 |
| 1       | Your trail (`1.0` where your trail exists)                |
| 2       | Opponent trail (`1.0` where opponent trail exists)        |
| 3       | Your head position (`1.0` at your current cell)           |
| 4       | Opponent head position (`1.0` at opponent's current cell) |

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

| Game | Submission A role       | Submission B role       |
| ---- | ----------------------- | ----------------------- |
| 1    | Player 0 (top-left)     | Player 1 (bottom-right) |
| 2    | Player 1 (bottom-right) | Player 0 (top-left)     |
| 3    | Player 0 (top-left)     | Player 1 (bottom-right) |

Per-game scores follow a **death-cause cascade**. Rules apply in order; the first rule that matches your situation wins:

| #   | Your situation                                                                                                                      | Score  |
| --- | ----------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 1   | You killed your opponent (their `killed_by` is you) and you're still alive                                                          | `1.00` |
| 2   | You're still alive and your opponent self-destructed (hit a wall or their own trail)                                                | `0.80` |
| 3   | You killed your opponent but also died on the same step (head-on, mutual trail-kill, or you wall-died while your trail killed them) | `0.40` |
| 4   | Both players alive at `max_steps` (timeout draw)                                                                                    | `0.25` |
| 5   | Your opponent killed you and you did not kill them                                                                                  | `0.10` |
| 6   | You died alone (wall or your own trail), no kill credit                                                                             | `0.00` |
| —   | API error / crash / sandbox failure                                                                                                 | `0.00` |

The scoring rewards aggression: a clean kill (`1.00`) is worth more than waiting for your opponent to crash (`0.80`); taking the opponent down with you (`0.40`) beats timing out (`0.25`) or dying alone (`0.00`); even a losing engagement (`0.10`) beats a passive wall-death (`0.00`).

#### Match outcome and tiebreakers

The **per-duel score** is the average of the 3 per-game scores (`sum(per_game_scores) / num_games`). The submission with the higher per-duel score wins the match and advances; the loser is eliminated. This average only decides **who advances** in a given match — it is *not* the `eval_score` reported for the round (see [Round scoring](#round-scoring) below).

If per-duel scores tie, tiebreakers apply in order:

1. **Games won outright** — number of games in the duel where you survived and your opponent died.
2. **Kills caused** — number of games where your opponent's `killed_by` was you (covers Rule 1 and Rule 3).
3. **Fewer self-deaths** — number of games where you wall-died or hit your own trail without killing your opponent (Rule 6).
4. **Lower bracket seed** wins (final fallback).

### Round scoring

A submission's reported scores reflect **how deep it got in the bracket**, not its per-duel averages:

- **`eval_raw_score`** = the **number of bracket rounds it survived** — i.e. how many matches it won before being eliminated. The bracket winner survives every round.
- **`eval_score`** = `eval_raw_score / total_rounds`, where `total_rounds` is the bracket depth (the number of rounds the eventual winner played). This puts `eval_score` in `[0, 1]`, and the **tournament winner always scores `1.0`**.


### Multiple submissions

If you submit multiple models in one round, only the **latest submission** is used for evaluation. Previous versions are automatically replaced.

## Game Configuration

| Parameter    | Value                 |
| ------------ | --------------------- |
| Grid size    | 32x32                 |
| Max steps    | 500                   |
| Wall wrap    | No (walls are deadly) |
| Trail fade   | 0 (permanent trails)  |
| Items        | Disabled              |
| Move timeout | 0.1 seconds           |
| Spawn mode   | Corners               |

## Local Tooling

Two helper scripts let you evaluate submissions on your own machine — pit two
models against each other, or play one yourself in the terminal. Both use the
**same game engine** (`tron.py`) and the **same model inference path** as the
competition launcher (`launch_tron_rl.py`), so behaviour matches a real duel.

### Prerequisites

The model servers need the same runtime deps as the sandbox. Install the
CPU-only PyTorch build into your environment:

```bash
uv pip install torch==2.9.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

`fastapi`, `uvicorn`, `numpy`, and `pydantic` are already provided by the
repo environment. Run both scripts from this directory:

```bash
cd shared/competition/src/competition/tron
```

### `run_local_duel.py` — model vs. model

Plays a full duel between two TorchScript submissions and reports a winner using
the **exact production scoring** (death-cause cascade → mean per-duel score →
tiebreakers of games-won › kills › fewest self-deaths, see [Scoring](#scoring)).
Under the hood it launches each model behind its own `launch_tron_rl.py` HTTP
server, plays the games with alternating spawn corners for fairness, then tears
the servers down.

```bash
python run_local_duel.py --a modelA.pt --b modelB.pt --games 21
```

Key flags:

| Flag             | Default | Purpose                                                       |
| ---------------- | ------- | ------------------------------------------------------------- |
| `--a` / `--b`    | —       | Paths to the two `.pt` models (required unless `--replay`)    |
| `--games`        | `3`     | Number of games in the duel (use an **odd** number)           |
| `--seed`         | `42`    | Base seed; game *N* uses `seed + N`                           |
| `--move-timeout` | `0.5`   | Per-move deadline in seconds (matches production)             |
| `--max-steps`    | `500`   | Steps before a game is a timeout draw                         |
| `--save-replay`  | —       | Write a JSON replay artifact of the duel to this path         |
| `--replay`       | —       | Replay a saved artifact in the terminal (skips running a duel) |
| `--replay-game`  | `-1`    | With `--replay`: which game index to show (`-1` = all)        |
| `--tick`         | `0.12`  | Replay animation seconds per step                             |

**Generate a replay artifact**, then watch it back:

```bash
python run_local_duel.py --a modelA.pt --b modelB.pt --games 3 --save-replay duel.json
python run_local_duel.py --replay duel.json                 # all games
python run_local_duel.py --replay duel.json --replay-game 1 # just game 2, faster with --tick 0.08
```

The artifact stores each game's config, metadata (swap, seed, scores), and the
full per-step history. Replay reconstructs the board frame-by-frame and animates
it with curses: `@`/`o` (cyan) = model A, `X`/`x` (red) = model B, `#` = walls.
Controls: **q** quit · **n** next game · **space** pause.

### `play_vs_model.py` — human vs. model

Drive one light-cycle yourself in the terminal while a submission drives the
other. Real-time (one step per `--tick`), rendered with curses.

```bash
python play_vs_model.py --model modelA.pt
python play_vs_model.py --model modelA.pt --human-player 1 --tick 0.25
```

| Flag             | Default | Purpose                                                       |
| ---------------- | ------- | ------------------------------------------------------------- |
| `--model`        | —       | Path to the opponent `.pt` model (required)                   |
| `--human-player` | `0`     | `0` = you spawn top-left, `1` = you spawn bottom-right        |
| `--tick`         | `0.15`  | Seconds per step (lower = faster)                             |
| `--seed`         | `42`    | Seed (spawn corners are fixed; here for parity)               |

Controls: **arrow keys** / **WASD** to steer · **q** to quit. You are `@` (cyan),
the model is `X` (red); trails are `o`/`x`. Reversing into yourself is ignored,
as in-game.

> **Note:** unlike `run_local_duel.py`, this script applies **no per-move
> timeout** — the model gets unlimited time to decide each step, so a model slow
> enough to lose moves in production will run clean here.

Both scripts require a real terminal for the curses display (`run_local_duel.py`
only for `--replay`); they won't render inside a piped or non-interactive shell.
