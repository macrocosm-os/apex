#!/usr/bin/env python3
"""
Run a local Tron duel between two TorchScript models and report a winner.

This mirrors the production duel scoring in
shared/backend/src/backend/eval/tron/runner.py:
  - 3 games (configurable) with alternating spawn corners for fairness
  - per-game death-cause score cascade
  - per-duel score = mean of per-game scores
  - tiebreakers: games won > kills caused > fewer self-deaths

It launches each model behind its own launch_tron_rl.py HTTP server, plays the
games in-process via run_duel_game, then tears the servers down.

Usage:
    python run_local_duel.py --a /path/to/modelA.pt --b /path/to/modelB.pt
    python run_local_duel.py --a A.pt --b B.pt --games 3 --seed 42 --max-steps 500
"""

import argparse
import curses
import json
import os
import subprocess
import sys
import tempfile
import time

# Make the competition package importable when run as a plain script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from competition.tron.tron import (  # noqa: E402
    DEATH_SELF,
    DEATH_WALL,
    GameConfig,
    TronGameResult,
    run_duel_game,
)

LAUNCHER = os.path.join(_THIS_DIR, "launch_tron_rl.py")


def _killed_by_includes(death_info: dict, player_id: int) -> bool:
    """True if player_id appears in the death's killed_by (int or list)."""
    killed_by = death_info.get("killed_by")
    if killed_by is None:
        return False
    if isinstance(killed_by, list):
        return player_id in killed_by
    return killed_by == player_id


def _score_for_player(this_id: int, opponent_id: int, death_causes: dict) -> float:
    """Death-cause cascade — identical to backend runner._score_for_player."""
    this_died = this_id in death_causes
    opp_died = opponent_id in death_causes
    this_alive = not this_died

    opp_death = death_causes.get(opponent_id, {})
    this_death = death_causes.get(this_id, {})

    this_killed_opp = _killed_by_includes(opp_death, this_id)
    opp_killed_this = _killed_by_includes(this_death, opponent_id)

    if this_killed_opp and this_alive:
        return 1.00
    if this_alive and opp_died and opp_death.get("cause") in (DEATH_WALL, DEATH_SELF):
        return 0.80
    if this_killed_opp and this_died:
        return 0.40
    if this_alive and not opp_died:
        return 0.25
    if opp_killed_this and not this_killed_opp:
        return 0.10
    return 0.00


def _launch_server(model_path: str, port: int):
    """Launch a model server, capturing its output to a log file.

    Returns (proc, log_path). The log is tailed if the server dies early.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    log_path = os.path.join(tempfile.gettempdir(), f"tron_server_{os.path.basename(model_path)}_{port}.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, LAUNCHER, "--model", model_path, "--port", str(port), "--host", "127.0.0.1"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc, log_path


def _check_alive(proc: subprocess.Popen, log_path: str, label: str) -> None:
    """Raise with the captured log tail if the server process has already exited."""
    if proc.poll() is None:
        return
    try:
        with open(log_path) as f:
            tail = f.read().strip()
    except OSError:
        tail = "(no output captured)"
    raise RuntimeError(f"{label} server exited early (code {proc.returncode}). Server output:\n{tail}")


def _frames_from_game(game: dict):
    """Reconstruct an ordered list of (positions, alive) frames for one game.

    positions / alive are dicts keyed by int player id. The first frame is the
    spawn state; each subsequent frame is a recorded step.
    """
    res = game["result"]
    spawn = res.get("spawn_positions") or {}
    history = res.get("history") or []

    frames = []
    spawn_pos = {int(k): [v["y"], v["x"]] for k, v in spawn.items()}
    frames.append((spawn_pos, {pid: True for pid in spawn_pos}))
    for rec in history:
        pos = {int(k): list(v) for k, v in rec["positions"].items()}
        alv = {int(k): bool(v) for k, v in rec["alive"].items()}
        frames.append((pos, alv))
    return frames


def _replay_loop(stdscr, games, label_a, label_b, tick) -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, -1)  # walls
    curses.init_pair(2, curses.COLOR_CYAN, -1)  # model A
    curses.init_pair(3, curses.COLOR_RED, -1)  # model B
    stdscr.nodelay(True)
    stdscr.keypad(True)

    def poll() -> str:
        """Return 'quit', 'next', or '' based on pending keys (also handles pause)."""
        while True:
            key = stdscr.getch()
            if key == -1:
                return ""
            if key in (ord("q"), ord("Q")):
                return "quit"
            if key in (ord("n"), ord("N")):
                return "next"
            if key in (ord("p"), ord("P"), ord(" ")):
                stdscr.nodelay(False)
                stdscr.getch()  # block until any key resumes
                stdscr.nodelay(True)

    for g in games:
        res = g["result"]
        cfg = res.get("config") or {}
        h, w = cfg.get("height", 32), cfg.get("width", 32)
        wall_wrap = cfg.get("wall_wrap", False)
        a_id, b_id = g["a_game_id"], g["b_game_id"]

        wall_cells = set()
        if not wall_wrap:
            for x in range(w):
                wall_cells.add((0, x))
                wall_cells.add((h - 1, x))
            for y in range(h):
                wall_cells.add((y, 0))
                wall_cells.add((y, w - 1))

        owner = {}  # (y, x) -> player id of trail
        frames = _frames_from_game(g)
        header = f"Game {g['index'] + 1}  |  @={label_a} (A)  X={label_b} (B)  " f"|  swap={g['swap']} seed={g['seed']}"
        footer_keys = "[q]uit  [n]ext game  [space]=pause"

        quit_all = False
        for i, (positions, alive) in enumerate(frames):
            action = poll()
            if action == "quit":
                return
            if action == "next":
                break

            stdscr.erase()
            for y, x in wall_cells:
                try:
                    stdscr.addstr(y, x * 2, "# ", curses.color_pair(1))
                except curses.error:
                    pass
            for (y, x), pid in owner.items():
                pair = 2 if pid == a_id else 3
                ch = "o" if pid == a_id else "x"
                try:
                    stdscr.addstr(y, x * 2, ch, curses.color_pair(pair))
                except curses.error:
                    pass
            # heads (only living players); record every head cell as future trail
            for pid, (y, x) in positions.items():
                if alive.get(pid, False):
                    pair = 2 if pid == a_id else 3
                    ch = "@" if pid == a_id else "X"
                    try:
                        stdscr.addstr(y, x * 2, ch, curses.color_pair(pair) | curses.A_BOLD)
                    except curses.error:
                        pass
                owner[(y, x)] = pid

            try:
                stdscr.addstr(h + 1, 0, header)
                stdscr.addstr(h + 2, 0, f"step {i}/{len(frames) - 1}   {footer_keys}")
            except curses.error:
                pass
            stdscr.refresh()
            time.sleep(tick)

        # End-of-game pause with outcome.
        outcome = res.get("game_result", "")
        score_line = f"score: A={g['score_a']:.2f}  B={g['score_b']:.2f}"
        try:
            stdscr.addstr(h + 3, 0, f"{outcome}   {score_line}   (press any key for next, q to quit)")
        except curses.error:
            pass
        stdscr.refresh()
        stdscr.nodelay(False)
        key = stdscr.getch()
        stdscr.nodelay(True)
        if key in (ord("q"), ord("Q")):
            return


def replay_artifact(path: str, which_game: int, tick: float) -> int:
    if not os.path.isfile(path):
        print(f"Replay artifact not found: {path}", file=sys.stderr)
        return 1
    with open(path) as f:
        art = json.load(f)
    if art.get("type") != "tron_duel_replay":
        print(f"Not a tron duel replay artifact: {path}", file=sys.stderr)
        return 1

    games = art.get("games", [])
    if which_game >= 0:
        games = [g for g in games if g["index"] == which_game]
        if not games:
            print(f"No game with index {which_game} in artifact.", file=sys.stderr)
            return 1

    label_a = art.get("model_a", "A")
    label_b = art.get("model_b", "B")
    curses.wrapper(_replay_loop, games, label_a, label_b, tick)

    s = art.get("summary", {})
    print(f"Replayed {art.get('model_a')} (A) vs {art.get('model_b')} (B)")
    if s:
        print(f"Duel: A={s.get('duel_score_a'):.3f}  B={s.get('duel_score_b'):.3f}  " f"Winner: {s.get('winner')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local Tron duel between two .pt models.")
    parser.add_argument("--a", help="Path to model A (.pt TorchScript)")
    parser.add_argument("--b", help="Path to model B (.pt TorchScript)")
    parser.add_argument("--games", type=int, default=3, help="Number of games (default 3)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; game N uses seed+N")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--move-timeout", type=float, default=0.5)
    parser.add_argument("--port-a", type=int, default=8001)
    parser.add_argument("--port-b", type=int, default=8002)
    parser.add_argument("--save-replay", metavar="PATH", help="Write a JSON replay artifact of the duel to PATH")
    parser.add_argument(
        "--replay", metavar="PATH", help="Replay a saved artifact in the terminal instead of running a duel"
    )
    parser.add_argument(
        "--replay-game",
        type=int,
        default=-1,
        help="With --replay: which game index to show (default -1 = all in sequence)",
    )
    parser.add_argument("--tick", type=float, default=0.12, help="Replay seconds per step (lower = faster)")
    args = parser.parse_args()

    # Replay mode: load an artifact and render it; no models/servers needed.
    if args.replay:
        return replay_artifact(args.replay, args.replay_game, args.tick)

    if not args.a or not args.b:
        parser.error("--a and --b are required unless using --replay")

    config = GameConfig(width=args.width, height=args.height, max_steps=args.max_steps, num_players=2)

    print(f"Launching model A ({args.a}) on :{args.port_a}")
    print(f"Launching model B ({args.b}) on :{args.port_b}")
    proc_a, log_a = _launch_server(args.a, args.port_a)
    proc_b, log_b = _launch_server(args.b, args.port_b)
    url_a = f"http://127.0.0.1:{args.port_a}"
    url_b = f"http://127.0.0.1:{args.port_b}"

    # Give servers a moment to import torch + load the model, then fail fast
    # with the captured server log if either died (e.g. missing deps, bad model).
    time.sleep(3)
    _check_alive(proc_a, log_a, "Model A")
    _check_alive(proc_b, log_b, "Model B")

    raw_a, raw_b = [], []
    won_a = won_b = kills_a = kills_b = self_a = self_b = 0
    games_data = []

    try:
        for game_idx in range(args.games):
            # Alternate who spawns top-left (player 0) vs bottom-right (player 1).
            swap = game_idx % 2 == 1
            p1_url, p2_url = (url_b, url_a) if swap else (url_a, url_b)
            a_game_id = 1 if swap else 0  # game-engine player id for model A
            b_game_id = 0 if swap else 1
            game_seed = args.seed + game_idx

            result: TronGameResult = run_duel_game(
                p1_url=p1_url,
                p2_url=p2_url,
                config=config,
                seed=game_seed,
                move_timeout=args.move_timeout,
            )

            death_causes = result.death_causes or {}
            failed = result.steps == 0 and result.winner is None and not death_causes
            if failed:
                sa = sb = 0.0
                a_killed = b_killed = a_alone = b_alone = a_alive = b_alive = False
            else:
                sa = _score_for_player(a_game_id, b_game_id, death_causes)
                sb = _score_for_player(b_game_id, a_game_id, death_causes)
                a_killed = _killed_by_includes(death_causes.get(b_game_id, {}), a_game_id)
                b_killed = _killed_by_includes(death_causes.get(a_game_id, {}), b_game_id)
                a_alone = death_causes.get(a_game_id, {}).get("cause") in (DEATH_WALL, DEATH_SELF)
                b_alone = death_causes.get(b_game_id, {}).get("cause") in (DEATH_WALL, DEATH_SELF)
                a_alive = a_game_id not in death_causes
                b_alive = b_game_id not in death_causes

            raw_a.append(sa)
            raw_b.append(sb)
            won_a += int(a_alive and not b_alive)
            won_b += int(b_alive and not a_alive)
            kills_a += int(a_killed)
            kills_b += int(b_killed)
            self_a += int(a_alone and not a_killed)
            self_b += int(b_alone and not b_killed)

            games_data.append(
                {
                    "index": game_idx,
                    "swap": swap,
                    "seed": game_seed,
                    "a_game_id": a_game_id,
                    "b_game_id": b_game_id,
                    "score_a": sa,
                    "score_b": sb,
                    "result": result.model_dump(),
                }
            )

            print(
                f"  Game {game_idx + 1} (swap={swap}, seed={game_seed}): "
                f"{result.game_result} in {result.steps} steps  ->  A={sa:.2f} B={sb:.2f}"
            )
    finally:
        for proc in (proc_a, proc_b):
            proc.terminate()
        for proc in (proc_a, proc_b):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    n = args.games
    duel_a = sum(raw_a) / n if n else 0.0
    duel_b = sum(raw_b) / n if n else 0.0
    key_a = (duel_a, won_a, kills_a, -self_a)
    key_b = (duel_b, won_b, kills_b, -self_b)

    print("\n=== Duel result ===")
    print(f"Model A: score={duel_a:.3f}  games_won={won_a}  kills={kills_a}  self_deaths={self_a}")
    print(f"Model B: score={duel_b:.3f}  games_won={won_b}  kills={kills_b}  self_deaths={self_b}")
    winner = "A" if key_a > key_b else "B" if key_b > key_a else "TIE"
    if winner == "TIE":
        print("Winner: TIE (production falls back to bracket seed)")
    else:
        print(f"Winner: {winner}")

    if args.save_replay:
        artifact = {
            "type": "tron_duel_replay",
            "version": 1,
            "model_a": os.path.basename(args.a),
            "model_b": os.path.basename(args.b),
            "config": {"width": args.width, "height": args.height, "max_steps": args.max_steps},
            "summary": {
                "duel_score_a": duel_a,
                "duel_score_b": duel_b,
                "winner": winner,
            },
            "games": games_data,
        }
        with open(args.save_replay, "w") as f:
            json.dump(artifact, f)
        print(f"\nReplay artifact saved to {args.save_replay}")
        print(f"Watch it with:  python {os.path.basename(__file__)} --replay {args.save_replay}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
