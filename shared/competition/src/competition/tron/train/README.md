# Tron RL Training (Baseline)

A minimal DQN training scaffold for the Tron Duel competition. Produces a
TorchScript `.pt` file that can be submitted directly.

## Quick start

```bash
cd train
pip install -r requirements.txt

# trains against a random opponent for 500k steps (~30 min on CPU)
python train.py
```

When training finishes, your submission file is at `models/tron_model.pt`.

## What's here

| File               | Purpose                                                                          |
| ------------------ | -------------------------------------------------------------------------------- |
| `env.py`           | Gymnasium env wrapping `tron.TronGame`. Agent is Player 0, opponent is Player 1. |
| `train.py`         | DQN training loop + TorchScript export.                                          |
| `requirements.txt` | Python dependencies.                                                             |

## Model interface (must match)

The competition launcher (`launch_tron_rl.py`) expects:

- **Input**: tensor of shape `(1, 5, H, W)` — channels: walls, your trail, opponent trail, your head, opponent head
- **Output**: tensor of shape `(4,)` — Q-values / logits over `[UP, RIGHT, DOWN, LEFT]`

`TronEnv` produces observations in this exact format and `train.py` exports a
network that produces the right output shape, so a model trained here drops
straight into the sandbox.

## Reward shaping

`TronEnv` gives:
- `+0.01` per surviving step
- `+1.0` on win, `-1.0` on loss, `0.0` on draw

Tweak in `env.py:step()` if you want a different signal.

## Going beyond random opponents

`TronEnv(opponent_fn=...)` accepts any function `(game, player_id) -> action`.
For self-play, write a wrapper that loads a frozen TorchScript snapshot of
your own model and uses it as the opponent. A simple recipe:

1. Periodically during training, export `model.q_net` to TorchScript and save
   it under `snapshots/`.
2. At each `env.reset()`, sample a random snapshot (or stay random with some
   probability) and instantiate it as the opponent.

This is left as an extension — the bare scaffold above gets you a working
submission first.
