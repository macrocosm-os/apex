#!/usr/bin/env python3
"""
Play Tron against a submitted TorchScript model in the terminal.

You drive one light-cycle with the arrow keys (or WASD); the model drives the
other. Same game engine (competition.tron.tron.TronGame) and the same model
inference path as the competition launcher (launch_tron_rl.py), so the model
behaves exactly as it would in a real duel.

Controls:
    Arrow keys / WASD : steer
    q                 : quit
    (you cannot reverse into yourself — reversing is ignored, as in-game)

Usage:
    python play_vs_model.py --model submission_88165_round6_top.pt
    python play_vs_model.py --model M.pt --human-player 1 --tick 0.12
"""

import argparse
import curses
import os
import sys
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from competition.tron.launch_tron_rl import encode_state  # noqa: E402
from competition.tron.tron import (  # noqa: E402
    PLAYER_TRAIL_START,
    WALL,
    Direction,
    GameConfig,
    TronGame,
)

# key -> Direction
_KEYMAP = {
    curses.KEY_UP: Direction.UP,
    curses.KEY_RIGHT: Direction.RIGHT,
    curses.KEY_DOWN: Direction.DOWN,
    curses.KEY_LEFT: Direction.LEFT,
    ord("w"): Direction.UP,
    ord("d"): Direction.RIGHT,
    ord("s"): Direction.DOWN,
    ord("a"): Direction.LEFT,
    ord("W"): Direction.UP,
    ord("D"): Direction.RIGHT,
    ord("S"): Direction.DOWN,
    ord("A"): Direction.LEFT,
}


def model_action(model, device, game: TronGame, pid: int) -> int:
    """Pick the model's action exactly as launch_tron_rl.py's /move does."""
    player = game.get_player(pid)
    valid = game.get_valid_actions(pid)
    if player is None or not player.alive or not valid:
        return int(player.direction) if player else 0

    opponents = [p for p in game.players if p.id != pid]
    state = encode_state(
        grid=game.grid,
        player_id=pid,
        my_position=[player.y, player.x],
        my_direction=int(player.direction),
        opponent_positions=[[o.y, o.x] for o in opponents],
        opponent_alive=[o.alive for o in opponents],
        height=game.config.height,
        width=game.config.width,
    ).to(device)

    with torch.no_grad():
        output = model(state)

    if output.dim() == 1:
        q_values = output
    elif output.dim() == 2:
        q_values = output.squeeze(0)
    else:
        q_values = output.flatten()[:4]

    q_np = q_values.cpu().numpy()[:4]
    masked = np.full(4, float("-inf"))
    for a in valid:
        if 0 <= a < 4:
            masked[a] = q_np[a]
    best = int(np.argmax(masked))
    if masked[best] == float("-inf"):
        best = valid[0]
    return best


def _render(stdscr, game: TronGame, human_id: int, model_id: int, status: str) -> None:
    stdscr.erase()
    grid = game.grid
    h, w = grid.shape
    human_cell = PLAYER_TRAIL_START + human_id
    model_cell = PLAYER_TRAIL_START + model_id
    hp = game.get_player(human_id)
    mp = game.get_player(model_id)

    for y in range(h):
        for x in range(w):
            cell = grid[y, x]
            ch, attr = " ", curses.A_NORMAL
            if cell == WALL:
                ch, attr = "#", curses.color_pair(1)
            elif cell == human_cell:
                ch, attr = "o", curses.color_pair(2)
            elif cell == model_cell:
                ch, attr = "x", curses.color_pair(3)
            try:
                stdscr.addstr(y, x * 2, ch + " ", attr)
            except curses.error:
                pass
    # Draw heads on top
    if hp is not None:
        try:
            stdscr.addstr(hp.y, hp.x * 2, "@", curses.color_pair(2) | curses.A_BOLD)
        except curses.error:
            pass
    if mp is not None:
        try:
            stdscr.addstr(mp.y, mp.x * 2, "X", curses.color_pair(3) | curses.A_BOLD)
        except curses.error:
            pass

    info_row = h + 1
    try:
        stdscr.addstr(info_row, 0, f"step {game.step_count}   you=@(o)  model=X(x)   [arrows/WASD, q=quit]")
        stdscr.addstr(info_row + 1, 0, status)
    except curses.error:
        pass
    stdscr.refresh()


def _play(stdscr, model, device, config: GameConfig, human_id: int, tick: float, seed: int) -> str:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, -1)
    curses.init_pair(2, curses.COLOR_CYAN, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    model_id = 1 - human_id
    game = TronGame(config)
    game.reset(seed=seed)

    human_dir = game.get_player(human_id).direction
    _render(stdscr, game, human_id, model_id, "Get ready...")
    time.sleep(0.6)

    while not game.game_over:
        loop_start = time.time()
        # Drain input; last directional key wins, q quits.
        while True:
            key = stdscr.getch()
            if key == -1:
                break
            if key in (ord("q"), ord("Q")):
                return "You quit."
            if key in _KEYMAP:
                human_dir = _KEYMAP[key]

        actions = {}
        hp = game.get_player(human_id)
        if hp.alive:
            actions[human_id] = int(human_dir)  # engine ignores reversal
        mp = game.get_player(model_id)
        if mp.alive:
            actions[model_id] = model_action(model, device, game, model_id)

        game.step(actions)
        _render(stdscr, game, human_id, model_id, "")

        elapsed = time.time() - loop_start
        if elapsed < tick:
            time.sleep(tick - elapsed)

    # Outcome
    if game.winner == human_id:
        return "YOU WIN!"
    if game.winner == model_id:
        return "Model wins. You died."
    return "Draw."


def main() -> int:
    parser = argparse.ArgumentParser(description="Play Tron against a TorchScript model in the terminal.")
    parser.add_argument("--model", required=True, help="Path to TorchScript model (.pt)")
    parser.add_argument(
        "--human-player",
        type=int,
        choices=[0, 1],
        default=0,
        help="0 = you spawn top-left (default), 1 = you spawn bottom-right",
    )
    parser.add_argument("--tick", type=float, default=0.15, help="Seconds per step (lower = faster)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1

    device = torch.device("cpu")
    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    config = GameConfig(width=args.width, height=args.height, max_steps=args.max_steps, num_players=2)

    result = curses.wrapper(_play, model, device, config, args.human_player, args.tick, args.seed)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
