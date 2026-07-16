# Competition Externalization Phase 1 Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship battleship as the first externalized competition in **shadow mode**. Legacy `BattleshipEvaluation` stays canonical (its results are delivered to the orchestrator, scored, ranked). For every battleship job on PR-staging, the worker ALSO runs the Phase-0 `SoloRunner` against a published `apex.competition.v1` spec, compares the two `eval_score`s, and records the comparison — without ever delivering shadow results. Prove the external pipeline (public repo → CI build/sign → registry activation → syncer mirror → worker spec resolution → `result.json` contract) is score-consistent with the legacy in-process evaluator.

**Architecture:** Three repos in strict dependency order. (1) `apex-competition-battleship` (new public repo) publishes a signed, digest-pinned player image that bundles a vendored copy of the battleship game engine + scoring and runs the miner submission as an in-container HTTP subprocess. (2) `apex-competitions-registry` (private GitOps, not local) activates that spec in `active/stage.yaml`. (3) `apex-mvp` gains a self-contained **shadow subsystem** — one revertable commit (flag + `shadow.py` + `shadow_store.py` + migration 012 + `_handle_job` hook + tests) that NEVER merges to `main`; it is reverted after the shadow round and re-applied per future solo port.

**Tech Stack:** Python 3.12, UV workspace, FastAPI/async, SQLAlchemy raw-SQL over asyncpg (`DatabaseClient`), pytest-asyncio (`asyncio_mode=auto`), Docker/K8s sandboxes, cosign keyless (OIDC) signing, crane mirroring, prometheus (NOT used for shadow — see Global Constraints).

## Global Constraints
- Shadow is **observability via structured logs + a DB table ONLY** — do NOT add any `apex_worker_shadow_*` prometheus metric series (§5.5 of the spec).
- The ENTIRE shadow subsystem is **ONE git commit that never reaches `main`**: flag, `shadow.py`, `shadow_store.py`, migration `012`, `_handle_job` hook, `test_shadow_dispatch.py`. It is `git revert`-ed before the branch merges (§5.4).
- Shadow runs **after** canonical delivery, **never concurrently**; a shadow failure of any kind is logged and swallowed and can NEVER raise, delay-before-delivery, or alter the canonical job result (§5.2).
- Score-scale (settled §3.3): the battleship image emits the **normalized 0–1** value as `raw_score`; the 1009-scale mean goes in `metadata.unnormalized_raw_score`. Shadow compares `eval_score` ONLY; `eval_raw_score` divergence is expected and excluded.
- Determinism (settled §3.4): the ported engine **honors `task.input.seed`** (it already does). Phase-1 gate = pipeline success ≥99% + status parity ≥99% + Spearman(legacy, shadow `eval_score`) ≥0.95 with per-submission |Δ| ≤0.3 sanity bound. Not per-submission bit-parity.
- Legacy scoring to match (from `shared/backend/src/backend/eval/battleship/runner.py::_calculate_game_score` + `normalizer.py`): `base = size**2 * 10`; `speed_bonus = max(size**2 - turns, 0) * 0.1`; win required (else 0); `per_game_raw = base + speed_bonus`; normalized = `per_game_raw / 1009`.
- `SPEC_DRIVEN_ENABLED` stays `false` throughout — a resolvable spec never becomes canonical in Phase 1.
- `SPEC_DRIVEN_ROUND_GEN` stays OFF for battleship — the `generate_round` entrypoint ships but is not activated.
- Registry `competition_key` == spec `id` == `competition_pkg` == `"battleship"`.

---

## File Structure

### Repo 1 — `apex-competition-battleship` (new, at `/Users/giannisevagorou/projects/apex-competition-battleship`)
| Path | Responsibility |
|---|---|
| `spec.yaml` | `apex.competition.v1` spec (concrete values, §3.1). |
| `input.schema.json` | JSON Schema for one round `{tasks:[{task_name, input:{...}}]}`, derived from `BattleshipInputDataSchema`. |
| `player/Dockerfile` | `python:3.12-slim`, uid 1000, `/app` + `/workspace` + `/data`, deps, bakes eval assets. |
| `player/battleship_engine.py` | Vendored game engine (verbatim copy of monorepo `battleship.py`, replay/CLI tail dropped). Honors `seed`. |
| `player/scoring.py` | Port of `_calculate_game_score` + `/1009` normalizer. Pure functions. |
| `player/evaluate.py` | Eval entrypoint: launch submission HTTP subprocess, health-poll, run `run_game` per task with 30s cap, score, aggregate, write `/data/result.json`. |
| `player/generate_round.py` | Port of `BattleshipInputDataGenerator` → `/data/generated_tasks.json` (numpy only). Ships, not activated. |
| `player/screener.py` | Standalone ASTGuard-based screener. Shipped, NOT wired. |
| `player/ast_guard.py` | Vendored `ASTGuard` (verbatim, pure stdlib). |
| `player/submission.py` | Reference miner = vendored `baseline.py` (FastAPI RandomShooter, reads `--port`). |
| `fixtures/input.json` | Deterministic 2-task round (fixed seeds, low `max_turns`). |
| `tests/test_scoring.py` | Scoring parity vs a golden table replicating legacy `_calculate_game_score`. |
| `tests/test_determinism.py` | Same seed ⇒ identical hidden board placement. |
| `.github/workflows/release.yml` | Build → push by digest → cosign keyless sign → emit digest + registry snippet artifact. |
| `README.md` | What it is, how to `apex-dev preflight`/`run`, publish flow. |

### Repo 2 — `apex-competitions-registry` (private, NOT local — authored/PR'd, described only)
| Path | Responsibility |
|---|---|
| `competitions/battleship/1.0.0.yaml` | Copy of repo-1 `spec.yaml` with `image.digest` filled from CI. |
| `competitions/battleship/input.schema.json` | Sibling copy of repo-1 `input.schema.json`. |
| `active/stage.yaml` | Add `battleship: "1.0.0"` under `competitions:`. `active/prod.yaml` untouched. |

### Repo 3 — `apex-mvp` (the ONE revertable shadow commit)
| Path | Create/Modify | Responsibility |
|---|---|---|
| `src/worker/src/worker/settings.py` | Modify | Add `SPEC_DRIVEN_SHADOW` (set) + `SPEC_DRIVEN_SHADOW_EPSILON` (float). |
| `src/scheduler/src/scheduler/db/migrations/012_add_shadow_eval_comparisons.sql` | Create | Temp comparison table (`-- UP`/`-- DOWN`). |
| `src/worker/src/worker/shadow_store.py` | Create | Pydantic row model + `record_comparison(...)` with own lazy write-capable `DatabaseClient`; swallow write errors; `close()`. |
| `src/worker/src/worker/shadow.py` | Create | `run_shadow(job, canonical_results, sandbox_cls)`: resolve spec, build `SoloRunner` (no callback), timeout-bounded run, compare (§5.3), structured logs (§5.5), call store. Never raises. |
| `src/worker/src/worker/worker.py` | Modify | Import `shadow` + `SPEC_DRIVEN_SHADOW`; add the `_handle_job` hook after the end-of-run sweep; `shadow_store.close()` in `cleanup`. |
| `src/worker/tests/test_shadow_dispatch.py` | Create | Cases (a)–(e) mirroring `test_spec_dispatch.py`. |

---

# Part A — Repo 1: `apex-competition-battleship`

> New sibling repo at `/Users/giannisevagorou/projects/apex-competition-battleship`. It MUST NOT import the private monorepo — everything is vendored or newly written. Tests run with plain `pytest` inside this repo (it has its own venv/deps). Commit after each task.

## Task A1: Repo scaffold + Dockerfile + README

**Files:**
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/README.md`
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/player/Dockerfile`
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/.gitignore`
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/requirements.txt`

**Interfaces:**
- Produces: a buildable image context whose `WORKDIR /app` bakes eval assets; `/workspace` and `/data` exist and are owned by uid 1000; submission arrives at `/workspace/submission.py` at run time (file-level mount), input/result under `/data`.
- Consumes: nothing.

Steps:
- [ ] `mkdir -p /Users/giannisevagorou/projects/apex-competition-battleship/{player,fixtures,tests,.github/workflows}` and `git init` there.
- [ ] Create `requirements.txt` (runtime + test deps):
  ```
  fastapi
  uvicorn
  pydantic
  requests
  numpy
  pytest
  ```
- [ ] Create `.gitignore`:
  ```
  __pycache__/
  *.pyc
  .venv/
  .pytest_cache/
  result.json
  generated_tasks.json
  ```
- [ ] Create `player/Dockerfile` (based on the SDK hello-world Dockerfile, adjusted for the `/workspace` submission path + engine deps):
  ```dockerfile
  # apex-competition-battleship player image.
  #
  # Build context = this directory (player/):
  #     docker build -t apex-competition-battleship .
  # or let apex-dev build it:
  #     apex-dev run --spec ../spec.yaml --input ../fixtures/input.json \
  #                  --submission ./submission.py --dockerfile ./Dockerfile
  #
  # The eval entrypoint (evaluate.py) + vendored engine/scoring are baked into /app.
  # The miner SUBMISSION is NOT baked in: the platform (and apex-dev) writes it to
  # /workspace/submission.py at run time (spec.submission.target_path).
  FROM python:3.12-slim

  # Match the platform sandbox user (run_as_user=1000).
  RUN useradd --create-home --uid 1000 app \
      && mkdir -p /app /workspace /data \
      && chown -R 1000:1000 /app /workspace /data

  WORKDIR /app

  RUN pip install --no-cache-dir fastapi uvicorn pydantic requests numpy

  # Bake the eval assets. submission.py is intentionally NOT copied.
  COPY evaluate.py /app/evaluate.py
  COPY battleship_engine.py /app/battleship_engine.py
  COPY scoring.py /app/scoring.py
  COPY generate_round.py /app/generate_round.py

  USER 1000

  # spec.entrypoints.evaluate.command is ["python", "/app/evaluate.py"].
  CMD ["python", "/app/evaluate.py"]
  ```
- [ ] Create `README.md` documenting: purpose (public reference solo competition), the in-container HTTP-subprocess model, local validation (`apex-dev preflight --spec spec.yaml`; `apex-dev run --spec spec.yaml --input fixtures/input.json --submission player/submission.py --dockerfile player/Dockerfile`), and the release flow (tag → CI signs → copy digest into the registry).
- [ ] Commit: `git add -A && git commit -m "battleship competition: repo scaffold + player Dockerfile"`

## Task A2: `input.schema.json`

**Files:**
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/input.schema.json`

**Interfaces:**
- Produces: JSON Schema validating the round input `{tasks:[{task_name, input:{size, ships_spec, repeat_on_hit, enforce_no_touching, startup_health_check_timeout_in_seconds, board_generation_timeout_in_seconds, shot_timeout_in_seconds, starting_player, max_turns, seed}}]}`.
- Consumes: field set + defaults from `shared/backend/src/backend/eval/battleship/models.py::BattleshipInputDataSchema`.

Steps:
- [ ] Create `input.schema.json` (mechanical translation of `BattleshipInputDataSchema`; all input fields optional with the model defaults, `additionalProperties:false`, `task_name` + `input` required):
  ```json
  {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "battleship round input",
    "description": "One round is a batch of solo battleship games. Each task fixes the game parameters (crucially the seed, which is honored deterministically).",
    "type": "object",
    "additionalProperties": false,
    "required": ["tasks"],
    "properties": {
      "tasks": {
        "type": "array",
        "minItems": 1,
        "items": {
          "type": "object",
          "additionalProperties": false,
          "required": ["task_name", "input"],
          "properties": {
            "task_name": { "type": "string" },
            "input": {
              "type": "object",
              "additionalProperties": false,
              "properties": {
                "size": { "type": "integer", "default": 10, "minimum": 5 },
                "ships_spec": {
                  "type": ["object", "null"],
                  "default": null,
                  "additionalProperties": { "type": "integer" }
                },
                "repeat_on_hit": { "type": "boolean", "default": false },
                "enforce_no_touching": { "type": "boolean", "default": false },
                "startup_health_check_timeout_in_seconds": { "type": "integer", "default": 1 },
                "board_generation_timeout_in_seconds": { "type": "integer", "default": 1 },
                "shot_timeout_in_seconds": { "type": "integer", "default": 1 },
                "starting_player": { "type": "string", "default": "p1" },
                "max_turns": { "type": "integer", "default": 100, "minimum": 1 },
                "seed": { "type": "integer", "default": 42 }
              }
            }
          }
        }
      }
    }
  }
  ```
- [ ] Validate it is well-formed JSON: `python -c "import json; json.load(open('/Users/giannisevagorou/projects/apex-competition-battleship/input.schema.json'))"` → expect no output (exit 0).
- [ ] Commit: `git add input.schema.json && git commit -m "battleship: input.schema.json"`

## Task A3: Vendor the game engine (`battleship_engine.py`)

**Files:**
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/player/battleship_engine.py`

**Interfaces:**
- Produces: `run_game(name, p1_id, p1_url, size, max_turns, ships_spec, enforce_no_touching, startup_health_check_timeout_in_seconds, board_generation_timeout_in_seconds, shot_timeout_in_seconds, console_mode, seed) -> GameResult`; classes `Board`, `BoardManager`, `Validator`, `Ship`, `RemotePlayer`, `GameResult`, enum `Name`; helpers `health_check`, `init_board`, `ask_next_move`; constant `DEFAULT_SHIPS`. Deps: `requests`, `pydantic` (self-contained).
- Consumes: verbatim source `shared/competition/src/competition/battleship/battleship.py`.

Note: the engine **already honors `seed`** (`Board(..., seed=seed)` → `random.Random(seed)` in `run_game`), so NO behavioral edit is needed. The only edit is dropping the replay/CLI tail to keep the vendored file lean and dependency-free of `argparse` usage at import.

Steps:
- [ ] Copy `shared/competition/src/competition/battleship/battleship.py` verbatim to `player/battleship_engine.py`.
- [ ] Apply this edit — delete everything from the replay helpers through the CLI block (the block starting at the `# Replay logic from a saved log` banner, i.e. functions `_board_from_log_ships`, `infer_board_size_from_log`, `replay_from_log`, and the entire `if __name__ == "__main__":` argparse block). Concretely, remove all lines from:
  ```python
  # -----------------------------
  # Replay logic from a saved log
  # -----------------------------
  def _board_from_log_ships(ships_dict: Dict[str, Any], size: int) -> BoardManager:
  ```
  through end of file:
  ```python
      else:
          run_game(
              name=args.game_name,
              p1_id=args.p1_id,
              p1_url=args.p1,
              size=args.size,
              max_turns=args.max_turns,
              console_mode=args.console,
              enforce_no_touching=args.no_touching,
          )
  ```
  Keep everything above the replay banner (imports, `DEFAULT_SHIPS`, `Name`, `Ship`, `RemotePlayer`, `GameResult`, `health_check`, `init_board`, `ask_next_move`, `Board`, `BoardManager`, `Validator`, `run_game`). Also remove the now-unused imports `argparse` and `json` from the top import block (leave `time`, `uuid`, `requests`, `random`, and the typing/pydantic/enum imports, which `run_game`/`Board` still use).
- [ ] Sanity-import: `cd /Users/giannisevagorou/projects/apex-competition-battleship/player && python -c "import battleship_engine as e; print(e.run_game, e.DEFAULT_SHIPS)"` → expect it prints the function + ships dict (exit 0).
- [ ] Commit: `git add player/battleship_engine.py && git commit -m "battleship: vendor game engine (seed honored; replay/CLI dropped)"`

## Task A4: `scoring.py` (TDD)

**Files:**
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/player/scoring.py`
- Test: `/Users/giannisevagorou/projects/apex-competition-battleship/tests/test_scoring.py`

**Interfaces:**
- Produces: `calculate_game_raw_score(*, max_player_turns: int, player_turns: int | None, success: bool) -> float` (1000-scale, legacy-identical) and `normalize(raw_score: float) -> float` (`raw_score / 1009`).
- Consumes: legacy semantics from `runner.py::_calculate_game_score` + `normalizer.py`.

Steps:
- [ ] Write the failing test `tests/test_scoring.py` first:
  ```python
  import math
  from player import scoring


  def legacy_raw(max_player_turns, player_turns, success):
      """Mirror of legacy _calculate_game_score raw_score (runner.py ~226-250)."""
      if not success or player_turns is None or player_turns <= 0:
          return 0.0
      base = max_player_turns * 10
      speed_bonus = max(max_player_turns - player_turns, 0) * 0.1
      return base + speed_bonus


  def test_raw_score_parity_table():
      size = 10
      mpt = size**2  # 100 -> base 1000
      cases = [
          (mpt, 17, True),
          (mpt, 100, True),
          (mpt, 1, True),
          (mpt, 100, False),   # loss
          (mpt, 0, True),      # invalid turns
          (mpt, None, True),   # no turns
      ]
      for player_turns, _label in [(c[1], c) for c in cases]:
          pass
      for max_player_turns, player_turns, success in cases:
          got = scoring.calculate_game_raw_score(
              max_player_turns=max_player_turns, player_turns=player_turns, success=success
          )
          assert math.isclose(got, legacy_raw(max_player_turns, player_turns, success), rel_tol=1e-9, abs_tol=1e-9)


  def test_normalize_matches_legacy_divisor():
      assert math.isclose(scoring.normalize(1009.0), 1.0, rel_tol=1e-9)
      assert math.isclose(scoring.normalize(1000.0), 1000.0 / 1009, rel_tol=1e-9)
      assert scoring.normalize(0.0) == 0.0
  ```
  Add `tests/__init__.py` and `player/__init__.py` (empty) so `from player import scoring` resolves; run tests from repo root.
- [ ] Run it, expect FAIL: `cd /Users/giannisevagorou/projects/apex-competition-battleship && python -m pytest tests/test_scoring.py -q` → expect `ModuleNotFoundError: No module named 'player.scoring'` (or collection error).
- [ ] Minimal implementation `player/scoring.py`:
  ```python
  """Battleship scoring — a verbatim port of the legacy scorer + normalizer.

  Legacy sources (apex-mvp):
    - shared/backend/src/backend/eval/battleship/runner.py::_calculate_game_score
    - shared/backend/src/backend/eval/battleship/normalizer.py::BattleshipNormalizer.normalize

  Kept identical so shadow (this image) and legacy stay comparable on eval_score.
  """

  from __future__ import annotations

  NORMALIZER_DIVISOR = 1009  # base 1000 + max speed bonus 9 (see legacy normalizer)


  def calculate_game_raw_score(*, max_player_turns: int, player_turns: int | None, success: bool) -> float:
      """1000-scale per-game raw score. 0 unless the game was won with a positive turn count."""
      if not success or player_turns is None or player_turns <= 0:
          return 0.0
      base = max_player_turns * 10  # e.g. size 10 -> 1000
      speed_bonus = max(max_player_turns - player_turns, 0) * 0.1
      return base + speed_bonus


  def normalize(raw_score: float) -> float:
      """Map the 1000-scale raw score into [0, 1] using the legacy /1009 divisor."""
      return raw_score / NORMALIZER_DIVISOR
  ```
- [ ] Run it, expect PASS: `python -m pytest tests/test_scoring.py -q` → expect `2 passed`.
- [ ] Commit: `git add player/scoring.py player/__init__.py tests/ && git commit -m "battleship: scoring port + parity tests"`

## Task A5: Vendor reference submission (`submission.py`)

**Files:**
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/player/submission.py`

**Interfaces:**
- Produces: a FastAPI server exposing `GET /health`→`{"ok":true}`, `POST /board`, `POST /next-move` (RandomShooter), started via `python submission.py --port 8001`.
- Consumes: verbatim source `shared/competition/src/competition/battleship/baseline.py`.

Steps:
- [ ] Copy `shared/competition/src/competition/battleship/baseline.py` verbatim to `player/submission.py`. No edits — it already reads `--port` (default 8001) and binds `0.0.0.0`; `127.0.0.1` from inside the same container reaches it.
- [ ] Confirm the argv contract: `grep -n "add_argument(\"--port\"" /Users/giannisevagorou/projects/apex-competition-battleship/player/submission.py` → expect a match (the `--port` flag is present).
- [ ] Commit: `git add player/submission.py && git commit -m "battleship: reference RandomShooter submission (vendored baseline)"`

## Task A6: `evaluate.py` — the in-container game-loop port (TDD, addresses tensions 1–3)

**Files:**
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/player/evaluate.py`
- Test: `/Users/giannisevagorou/projects/apex-competition-battleship/tests/test_evaluate.py`

**Interfaces:**
- Produces: `run_round(tasks: list[dict]) -> dict` returning `{"raw_score": float(0-1), "eval_time_in_seconds": float, "metadata": dict}` where `metadata` has one entry per task plus `unnormalized_raw_score`; and a `main()` that reads `/data/input.json`, launches `/workspace/submission.py` as an HTTP subprocess, health-polls, runs each game with a 30s wall cap, and writes `/data/result.json`.
- Consumes: `player/battleship_engine.py::run_game`, `player/scoring.py::{calculate_game_raw_score, normalize}`.

Design (settled): honors `task.input.seed` (tension 2); emits normalized 0–1 as `raw_score`, 1009-scale mean in `metadata.unnormalized_raw_score` (tension 1); launches the miner submission as a loopback HTTP subprocess in the SAME container (tension 3). `startup_health_check` uses a **10s** startup budget (matching legacy's effective window), NOT the per-task 1s schema default.

Steps:
- [ ] Write the failing test `tests/test_evaluate.py` first (drives `run_round` against a fake `run_game`, so no Docker/network needed):
  ```python
  import math
  import player.evaluate as ev
  from player import scoring


  class _FakeResult:
      def __init__(self, turns, game_result):
          self.turns = turns
          self.game_result = game_result
          self.game_id = "gid"

      def model_dump_json(self):
          return "{}"


  def test_run_round_aggregates_and_normalizes(monkeypatch):
      # Two tasks: a win in 40 turns and a loss.
      def fake_run_game(**kwargs):
          if kwargs["seed"] == 1:
              return _FakeResult(turns=40, game_result="Player 1 won")
          return _FakeResult(turns=kwargs["max_turns"], game_result="Max turns reached")

      monkeypatch.setattr(ev, "run_game", fake_run_game)

      tasks = [
          {"task_name": "Game 1", "input": {"size": 10, "max_turns": 100, "seed": 1}},
          {"task_name": "Game 2", "input": {"size": 10, "max_turns": 100, "seed": 2}},
      ]
      out = ev.run_round(tasks)

      win_raw = scoring.calculate_game_raw_score(max_player_turns=100, player_turns=40, success=True)  # 1006.0
      expected_unnorm = (win_raw + 0.0) / 2
      assert math.isclose(out["metadata"]["unnormalized_raw_score"], expected_unnorm, rel_tol=1e-9)
      assert math.isclose(out["raw_score"], scoring.normalize(expected_unnorm), rel_tol=1e-9)
      assert 0.0 <= out["raw_score"] <= 1.0
      assert out["metadata"]["Game 1"]["success"] is True
      assert out["metadata"]["Game 2"]["success"] is False


  def test_run_round_seed_is_passed_through(monkeypatch):
      seen = []
      monkeypatch.setattr(ev, "run_game", lambda **k: seen.append(k["seed"]) or _FakeResult(10, "Player 1 won"))
      ev.run_round([{"task_name": "G", "input": {"size": 10, "max_turns": 50, "seed": 777}}])
      assert seen == [777]
  ```
- [ ] Run it, expect FAIL: `cd /Users/giannisevagorou/projects/apex-competition-battleship && python -m pytest tests/test_evaluate.py -q` → expect `ModuleNotFoundError`/`AttributeError` for `player.evaluate`.
- [ ] Minimal implementation `player/evaluate.py`:
  ```python
  """battleship solo eval entrypoint (spec.entrypoints.evaluate.command).

  Collapses the legacy split (worker runs the game loop; sandbox runs the miner server)
  into ONE container: this script launches the miner submission as a loopback HTTP
  subprocess, then runs the bundled game engine against it per task.

  Contract (SoloRunner / apex-dev):
    - miner submission written to /workspace/submission.py (spec.submission.target_path)
    - round input at /data/input.json (validated against input.schema.json)
    - MUST write /data/result.json = {raw_score: float(0-1), eval_time_in_seconds: float, metadata: {...}}

  Settled decisions:
    - raw_score is the NORMALIZED mean (0-1); the 1009-scale mean is metadata.unnormalized_raw_score.
    - the game seed is HONORED (deterministic given input).
  """

  from __future__ import annotations

  import concurrent.futures
  import json
  import subprocess
  import sys
  import time
  from pathlib import Path

  import requests

  from battleship_engine import run_game
  from scoring import calculate_game_raw_score, normalize

  SUBMISSION_PATH = Path("/workspace/submission.py")
  INPUT_PATH = Path("/data/input.json")
  RESULT_PATH = Path("/data/result.json")

  PORT = 8001
  PLAYER_URL = f"http://127.0.0.1:{PORT}"
  STARTUP_BUDGET_SECONDS = 10   # matches legacy's effective startup window (NOT the 1s per-task default)
  PER_GAME_WALL_CAP_SECONDS = 30  # replicates GAME_TIMEOUT_BUFFER_SECONDS


  def _wait_healthy(deadline: float) -> bool:
      while time.time() < deadline:
          try:
              r = requests.get(f"{PLAYER_URL}/health", timeout=1)
              if r.ok and r.json().get("ok"):
                  return True
          except Exception:
              pass
          time.sleep(0.5)
      return False


  def _score_one(task: dict) -> tuple[float, dict]:
      """Run one game with a hard wall cap; return (raw_1000_scale, metadata_entry)."""
      inp = task.get("input", {})
      size = int(inp.get("size", 10))
      max_turns = int(inp.get("max_turns", 100))
      seed = int(inp.get("seed", 42))
      start = time.time()

      def _play():
          return run_game(
              name=task.get("task_name", "game"),
              p1_id="miner",
              p1_url=PLAYER_URL,
              size=size,
              max_turns=max_turns,
              ships_spec=inp.get("ships_spec"),
              enforce_no_touching=bool(inp.get("enforce_no_touching", False)),
              startup_health_check_timeout_in_seconds=STARTUP_BUDGET_SECONDS,
              board_generation_timeout_in_seconds=int(inp.get("board_generation_timeout_in_seconds", 1)),
              shot_timeout_in_seconds=int(inp.get("shot_timeout_in_seconds", 1)),
              console_mode=False,
              seed=seed,
          )

      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
          try:
              result = pool.submit(_play).result(timeout=PER_GAME_WALL_CAP_SECONDS)
              success = result.game_result == "Player 1 won"
              turns = result.turns or 0
              raw = calculate_game_raw_score(max_player_turns=size**2, player_turns=turns, success=success)
              meta = {
                  "game_id": getattr(result, "game_id", None),
                  "game_result": result.game_result,
                  "turns": turns,
                  "success": success,
                  "max_turns": max_turns,
                  "seed": seed,
                  "execution_time": time.time() - start,
                  "score_breakdown": {"raw_1000_scale": raw, "normalized": normalize(raw)},
              }
              return raw, meta
          except concurrent.futures.TimeoutError:
              meta = {
                  "game_result": f"Game exceeded {PER_GAME_WALL_CAP_SECONDS}s wall cap",
                  "turns": 0,
                  "success": False,
                  "max_turns": max_turns,
                  "seed": seed,
                  "execution_time": time.time() - start,
                  "score_breakdown": {"raw_1000_scale": 0.0, "normalized": 0.0},
              }
              return 0.0, meta


  def run_round(tasks: list[dict]) -> dict:
      """Score every task; aggregate to a normalized 0-1 raw_score. Pure (no I/O, no server)."""
      start = time.time()
      per_game_raw: list[float] = []
      metadata: dict = {}
      for task in tasks:
          raw, meta = _score_one(task)
          per_game_raw.append(raw)
          metadata[task.get("task_name", f"Game {len(per_game_raw)}")] = meta
      unnorm_mean = (sum(per_game_raw) / len(per_game_raw)) if per_game_raw else 0.0
      metadata["unnormalized_raw_score"] = unnorm_mean
      return {
          "raw_score": normalize(unnorm_mean),
          "eval_time_in_seconds": time.time() - start,
          "metadata": metadata,
      }


  def main() -> None:
      tasks = json.loads(INPUT_PATH.read_text())["tasks"]
      proc = subprocess.Popen(
          [sys.executable, str(SUBMISSION_PATH), "--port", str(PORT)],
          stdout=sys.stdout,
          stderr=sys.stderr,
      )
      try:
          if not _wait_healthy(time.time() + STARTUP_BUDGET_SECONDS):
              raise RuntimeError(f"submission server never became healthy within {STARTUP_BUDGET_SECONDS}s")
          result = run_round(tasks)
      finally:
          proc.terminate()
          try:
              proc.wait(timeout=5)
          except Exception:
              proc.kill()

      RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
      RESULT_PATH.write_text(json.dumps(result))


  if __name__ == "__main__":
      main()
  ```
  Note the win-detection string `f"{'Player 1'} won"` equals the engine's `f"{Name.PLAYER_1.value} won"` = `"Player 1 won"`; keep it as the literal `"Player 1 won"` for clarity when implementing (the interpolation above is only to avoid a magic-string lint; either is fine as long as it equals `"Player 1 won"`).
- [ ] Simplify the win check to the literal to avoid confusion: use `success = result.game_result == "Player 1 won"`.
- [ ] Run it, expect PASS: `python -m pytest tests/test_evaluate.py -q` → expect `2 passed`.
- [ ] Commit: `git add player/evaluate.py tests/test_evaluate.py && git commit -m "battleship: evaluate.py in-container game-loop port (seed honored, normalized raw_score)"`

## Task A7: `generate_round.py` (ships, not activated)

**Files:**
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/player/generate_round.py`

**Interfaces:**
- Produces: writes `/data/generated_tasks.json` = `{"tasks": [...], "sandbox_data": {}}`; each task is `{"task_name": "Game i", "input": {"starting_player": "random", "max_turns": <int>, "seed": <int>}}`.
- Consumes: port of `shared/backend/src/backend/eval/battleship/generator.py` (numpy only).

Steps:
- [ ] Create `player/generate_round.py` (port of `BattleshipInputDataGenerator`; same Beta-sampled `max_turns` over `[15,200]`, random per-task seed):
  ```python
  """battleship round generation (spec.entrypoints.generate_round.command).

  Port of BattleshipInputDataGenerator (apex-mvp). Writes /data/generated_tasks.json.
  SHIPPED but NOT activated in Phase 1 (SPEC_DRIVEN_ROUND_GEN stays off for battleship).
  """

  from __future__ import annotations

  import json
  from pathlib import Path

  import numpy as np

  OUTPUT_PATH = Path("/data/generated_tasks.json")


  def generate_max_turns(number_of_tasks=5, min_turns=15, max_turns=200, alpha=20.0, beta=50.0, rng=None):
      rng = rng or np.random.default_rng()
      span = max_turns - min_turns
      u = rng.beta(alpha, beta, size=number_of_tasks)
      return [int(round(t)) for t in (min_turns + u * span)]


  def generate_tasks(number_of_tasks=5, alpha=20.0, beta=50.0) -> list[dict]:
      rng = np.random.default_rng()
      turns = generate_max_turns(number_of_tasks=number_of_tasks, alpha=alpha, beta=beta, rng=rng)
      tasks = []
      for i, mt in enumerate(turns, start=1):
          seed = int(rng.integers(0, 2**31))
          tasks.append({"task_name": f"Game {i}", "input": {"starting_player": "random", "max_turns": mt, "seed": seed}})
      return tasks


  def main() -> None:
      OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
      OUTPUT_PATH.write_text(json.dumps({"tasks": generate_tasks(), "sandbox_data": {}}))


  if __name__ == "__main__":
      main()
  ```
- [ ] Smoke-run: `cd /Users/giannisevagorou/projects/apex-competition-battleship/player && python -c "import generate_round as g; t=g.generate_tasks(3); print(len(t), t[0]['input'].keys())"` → expect `3 dict_keys(['starting_player', 'max_turns', 'seed'])`.
- [ ] Commit: `git add player/generate_round.py && git commit -m "battleship: generate_round port (shipped, not activated)"`

## Task A8: Screener (shipped, not wired) + vendored ASTGuard

**Files:**
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/player/ast_guard.py`
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/player/screener.py`

**Interfaces:**
- Produces: `BattleshipScreener().screen(submission: str) -> bool` (standalone; no monorepo imports).
- Consumes: verbatim `shared/backend/src/backend/utils/ast_guard.py` (pure stdlib `ast`).

Steps:
- [ ] Copy `shared/backend/src/backend/utils/ast_guard.py` verbatim to `player/ast_guard.py` (no edit — it only `import ast`).
- [ ] Create `player/screener.py` as a standalone rewrite of `shared/backend/src/backend/eval/battleship/screener.py` (drop the `BaseScreener` base and the `backend.*` imports; use the local `ast_guard`; drop loguru for stdlib print/logging to stay dependency-free):
  ```python
  """Static AST screener for battleship submissions (ported, standalone).

  Phase 1: SHIPPED as the reference asset for future spec-driven screening — NOT wired
  into the eval path (screening runs upstream in the orchestrator; §3.5).
  """

  from __future__ import annotations

  from ast_guard import ASTGuard


  class BattleshipScreener:
      def screen(self, submission: str) -> bool:
          guard = ASTGuard()
          try:
              tree = guard.parse_submission(submission)
          except SyntaxError:
              return False
          guard.visit(tree)
          return not guard.violations
  ```
- [ ] Sanity-import: `cd /Users/giannisevagorou/projects/apex-competition-battleship/player && python -c "from screener import BattleshipScreener as S; print(S().screen('x = 1'))"` → expect `True`.
- [ ] Commit: `git add player/ast_guard.py player/screener.py && git commit -m "battleship: vendor ASTGuard + standalone screener (shipped, not wired)"`

## Task A9: `spec.yaml` + `fixtures/input.json` + determinism test

**Files:**
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/spec.yaml`
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/fixtures/input.json`
- Test: `/Users/giannisevagorou/projects/apex-competition-battleship/tests/test_determinism.py`

**Interfaces:**
- Produces: a spec that passes `apex-dev preflight` (schema + stage ceilings cpu≤2/mem≤2048Mi) once the digest is filled; a deterministic fixture round; a test proving `seed` determinism.
- Consumes: `battleship_engine.Board`.

Steps:
- [ ] Write the failing determinism test `tests/test_determinism.py`:
  ```python
  from player.battleship_engine import Board


  def _placement(seed):
      b = Board(size=10, seed=seed)
      b.place_ships_randomly()
      return {name: sorted(ship.cells) for name, ship in b.ships.items()}


  def test_same_seed_same_board():
      assert _placement(1234) == _placement(1234)


  def test_different_seed_different_board():
      assert _placement(1) != _placement(2)
  ```
  (Requires `player/battleship_engine.py` importable as `player.battleship_engine`; `player/__init__.py` from Task A4 already exists.)
- [ ] Run it, expect PASS immediately (the engine already honors seed): `cd /Users/giannisevagorou/projects/apex-competition-battleship && python -m pytest tests/test_determinism.py -q` → expect `2 passed`. (This is a characterization test, not red-green; it guards the settled determinism decision.)
- [ ] Create `spec.yaml` with the concrete values from design §3.1. Leave the digest and the four `<filled from live stage config>` values as EXPLICIT markers (they are genuinely runtime-unknown per spec §9.4 and are filled at the registry step / from live config — do NOT invent them):
  ```yaml
  schema: apex.competition.v1
  id: battleship
  version: 1.0.0
  display_name: Battleship
  kind: solo
  process_type: cpu
  resources:
    cpu_limit: 2
    mem_limit: 1024Mi
    gpu_count: 0
  image:
    ref: ghcr.io/macrocosm-os/apex-competition-battleship
    digest: sha256:0000000000000000000000000000000000000000000000000000000000000000  # FILLED by release CI (Task A10) before registry activation
  submission:
    artifact_type: code
    max_size_mb: 1  # FILL from live stage battleship config before registry PR (spec §9.4)
    target_path: /workspace/submission.py
  input_schema:
    $ref: ./input.schema.json
  defaults:
    baseline_score: 0.0        # FILL from live stage config (spec §9.4)
    baseline_raw_score: 0.0    # FILL from live stage config; NEW 0-1 scale (spec §3.3)
    round_length_in_days: 1    # FILL from live stage config (spec §9.4)
    submission_reveal_days: 1  # FILL from live stage config (spec §9.4)
    lower_is_better: false
  entrypoints:
    evaluate:
      command: ["python", "/app/evaluate.py"]
      timeout_s: 300
      network_disabled: true
      allow_internet: false
    generate_round:
      command: ["python", "/app/generate_round.py"]
      timeout_s: 120
  signature:
    cosign_identity: https://github.com/macrocosm-os/apex-competition-battleship/.github/workflows/release.yml@refs/tags/*
    cosign_issuer: https://token.actions.githubusercontent.com
  ```
  (Note: the `# FILL` markers on `max_size_mb`/`defaults.*` carry placeholder values so `apex-dev preflight` schema validation passes locally; they are corrected from live config before the registry PR. This is the only place placeholder values are permitted, per spec §9.4.)
- [ ] Create `fixtures/input.json` (deterministic 2-task round, low `max_turns` so `apex-dev run` is fast):
  ```json
  {
    "tasks": [
      { "task_name": "Game 1", "input": { "size": 10, "max_turns": 40, "seed": 101 } },
      { "task_name": "Game 2", "input": { "size": 10, "max_turns": 60, "seed": 202 } }
    ]
  }
  ```
- [ ] Local pipeline validation (requires the SDK `apex-dev` CLI on PATH and Docker running). Run and expect success (valid `result.json` with `raw_score` in [0,1]):
  ```
  cd /Users/giannisevagorou/projects/apex-competition-battleship
  apex-dev preflight --spec spec.yaml
  apex-dev run --spec spec.yaml --input fixtures/input.json --submission player/submission.py --dockerfile player/Dockerfile
  ```
  Expected: preflight reports schema OK + resource ceilings OK; run prints a `result.json` whose `raw_score` is a float in [0,1] and `metadata` has `Game 1`/`Game 2`/`unnormalized_raw_score`. This also confirms loopback works under the sandbox network mode (spec §9.3). If `apex-dev` is unavailable in this environment, record that this validation is deferred to CI/stage and note it in the task completion.
- [ ] Commit: `git add spec.yaml fixtures/input.json tests/test_determinism.py && git commit -m "battleship: spec.yaml, fixtures, determinism test"`

## Task A10: Release workflow (build → push by digest → cosign sign → emit digest)

**Files:**
- Create `/Users/giannisevagorou/projects/apex-competition-battleship/.github/workflows/release.yml`

**Interfaces:**
- Produces: on tag push, a GHCR image pushed by digest, cosign keyless signature over that digest, and a build artifact containing the digest + a ready-to-paste registry `1.0.0.yaml` snippet (spec §4 / §2).
- Consumes: `player/Dockerfile` build context.

Steps:
- [ ] Create `.github/workflows/release.yml`:
  ```yaml
  name: release
  on:
    push:
      tags: ["v*.*.*"]
  permissions:
    contents: read
    packages: write
    id-token: write   # cosign keyless (OIDC)
  jobs:
    build-sign:
      runs-on: ubuntu-latest
      env:
        IMAGE: ghcr.io/macrocosm-os/apex-competition-battleship
      steps:
        - uses: actions/checkout@v4
        - uses: docker/login-action@v3
          with:
            registry: ghcr.io
            username: ${{ github.actor }}
            password: ${{ secrets.GITHUB_TOKEN }}
        - uses: sigstore/cosign-installer@v3
        - name: Build and push by digest
          id: build
          working-directory: player
          run: |
            set -euo pipefail
            docker build -t "$IMAGE:${GITHUB_REF_NAME}" .
            docker push "$IMAGE:${GITHUB_REF_NAME}"
            DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "$IMAGE:${GITHUB_REF_NAME}" | cut -d@ -f2)
            echo "digest=$DIGEST" >> "$GITHUB_OUTPUT"
        - name: Cosign sign by digest (keyless)
          run: cosign sign --yes "${IMAGE}@${{ steps.build.outputs.digest }}"
        - name: Emit registry snippet
          run: |
            mkdir -p out
            sed "s|sha256:0\{64\}|${{ steps.build.outputs.digest }}|" spec.yaml > out/1.0.0.yaml
            cp input.schema.json out/input.schema.json
            echo "${{ steps.build.outputs.digest }}" > out/DIGEST.txt
        - uses: actions/upload-artifact@v4
          with:
            name: registry-snippet
            path: out/
  ```
- [ ] Lint the YAML is well-formed: `python -c "import yaml,sys; yaml.safe_load(open('/Users/giannisevagorou/projects/apex-competition-battleship/.github/workflows/release.yml'))"` → expect no output. (If pyyaml is unavailable, skip; CI will validate.)
- [ ] Commit: `git add .github/workflows/release.yml && git commit -m "battleship: release workflow (build/push-by-digest/cosign, emit digest snippet)"`
- [ ] Run the full repo-1 test suite once: `cd /Users/giannisevagorou/projects/apex-competition-battleship && python -m pytest -q` → expect all tests pass.

---

# Part B — Repo 2: `apex-competitions-registry` (authored/PR'd, not built here)

> This private GitOps repo is not checked out locally. This part is a runbook, not TDD code. It is executed AFTER Part A's release CI produces the digest.

## Task B1: Activate the battleship spec in stage

**Files (in the registry repo):**
- Create `competitions/battleship/1.0.0.yaml`
- Create `competitions/battleship/input.schema.json`
- Modify `active/stage.yaml`

Steps:
- [ ] Wait for Part A Task A10 CI to succeed on a `v1.0.0` tag; download the `registry-snippet` artifact (`1.0.0.yaml` with the digest already substituted, `input.schema.json`, `DIGEST.txt`).
- [ ] Before committing, replace the four `# FILL from live stage config` values in `1.0.0.yaml` (`submission.max_size_mb`, `defaults.baseline_score`, `defaults.baseline_raw_score`, `defaults.round_length_in_days`, `defaults.submission_reveal_days`) with the live stage battleship competition values (spec §9.4). `baseline_raw_score` is on the NEW 0–1 scale (§3.3).
- [ ] Add `competitions/battleship/1.0.0.yaml` and sibling `competitions/battleship/input.schema.json` (the stem must be the 3-part semver `1.0.0`).
- [ ] Edit `active/stage.yaml`: add `battleship: "1.0.0"` under `competitions:`. **Do NOT touch `active/prod.yaml`.**
- [ ] Validate locally with the SDK before PR: `apex-dev preflight --spec competitions/battleship/1.0.0.yaml` → expect schema OK + ceilings OK (cpu 2 ≤ 2, mem 1024Mi ≤ 2048Mi).
- [ ] Open the registry PR. On merge, the spec-syncer (`src/spec-syncer/.../syncer.py`) validates (`load_spec` + `check_resource_ceilings`), cosign-verifies against the declared identity/issuer, crane-mirrors by digest, and upserts `CompetitionSpecVersion` + the stage `CompetitionActiveVersion` row.
- [ ] Dependency gate: Part C's end-to-end stage exercise cannot run until this syncer reconcile has populated the PR-staging DB with the battleship spec + active pointer (spec §9.2 — decide syncer-against-PR-DB vs. manual 2-row copy).

---

# Part C — Repo 3: `apex-mvp` shadow subsystem (ONE revertable commit)

> Work on a dedicated migration branch. **Do NOT commit incrementally to `main`.** Per the writing-plans TDD rhythm, each task below writes/commits locally, but the FINAL integration is one squashed/single commit ("the shadow commit") that Task C7 documents for reversion. Keep all six artifacts in that one logical commit so `git revert <sha>` removes them together.

## Task C0: Create the migration branch

Steps:
- [ ] `cd /Users/giannisevagorou/projects/apex-mvp && git checkout -b competition-externalization-phase-1-battleship-shadow`
- [ ] Confirm clean baseline: `uv run pytest src/worker/tests/test_spec_dispatch.py -q` → expect all pass (Phase-0 dispatch tests are green before we add shadow).

## Task C1: Settings — `SPEC_DRIVEN_SHADOW` + `SPEC_DRIVEN_SHADOW_EPSILON`

**Files:**
- Modify `/Users/giannisevagorou/projects/apex-mvp/src/worker/src/worker/settings.py`
- Test: covered indirectly by Task C6 (settings are plain env reads).

**Interfaces:**
- Produces: `SPEC_DRIVEN_SHADOW: set[str]` (comma-separated pkgs) and `SPEC_DRIVEN_SHADOW_EPSILON: float` (default 0.05).
- Consumes: `os.getenv`.

Steps:
- [ ] Add, immediately after the existing `SPEC_DRIVEN_ROUND_GEN` block in `settings.py`:
  ```python
  # Comma-separated competition pkgs that ALSO run the spec-driven runner in SHADOW mode
  # after the canonical legacy eval. Legacy stays canonical; shadow results are compared,
  # recorded to shadow_eval_comparisons, and discarded. Independent of SPEC_DRIVEN_ENABLED
  # (full cutover). Empty = no shadowing. This whole subsystem never merges to main.
  SPEC_DRIVEN_SHADOW = {p.strip() for p in os.getenv("SPEC_DRIVEN_SHADOW", "").split(",") if p.strip()}
  # Advisory tolerance recorded per comparison row (within_epsilon flag); not a hard gate.
  SPEC_DRIVEN_SHADOW_EPSILON = float(os.getenv("SPEC_DRIVEN_SHADOW_EPSILON", "0.05"))
  ```
- [ ] Verify import: `cd /Users/giannisevagorou/projects/apex-mvp && uv run python -c "from worker.settings import SPEC_DRIVEN_SHADOW, SPEC_DRIVEN_SHADOW_EPSILON; print(SPEC_DRIVEN_SHADOW, SPEC_DRIVEN_SHADOW_EPSILON)"` → expect `set() 0.05`.
- [ ] (No standalone commit yet — accumulate into the shadow commit; or commit with `--amend`-friendly message and squash at Task C7.)

Note on the "lazy-connect gate" (spec §5.1): `spec_resolver` connects on demand and has no internal flag check today — the operational gate is that callers only invoke it under a flag. We honor this by guarding the `run_shadow` call in `_handle_job` with `if job.competition_pkg in SPEC_DRIVEN_SHADOW` (Task C5), mirroring the existing `SPEC_DRIVEN_ENABLED`/`SPEC_DRIVEN_ROUND_GEN` pattern. No code change to `spec_resolver.py` is required; see Self-Review.

## Task C2: Migration 012 — `shadow_eval_comparisons`

**Files:**
- Create `/Users/giannisevagorou/projects/apex-mvp/src/scheduler/src/scheduler/db/migrations/012_add_shadow_eval_comparisons.sql`

**Interfaces:**
- Produces: table `shadow_eval_comparisons` with the exact columns from spec §5.4 + index on `(competition_pkg, round_number)`.
- Consumes: applied by the scheduler on startup via `db_client.migrations.apply_pending()` (PR-staging only).

Steps:
- [ ] Create the migration file with EXACTLY the SQL from spec §5.4 (matching the `-- UP` / `-- DOWN` convention of `011_*.sql`):
  ```sql
  -- UP
  CREATE TABLE shadow_eval_comparisons (
      id BIGSERIAL PRIMARY KEY,
      submission_id BIGINT NOT NULL,
      competition_pkg TEXT NOT NULL,
      competition_id BIGINT NOT NULL,
      round_number INT NOT NULL,
      spec_version_id BIGINT,          -- CompetitionSpecVersion.id
      spec_version TEXT,
      legacy_raw_score DOUBLE PRECISION,
      legacy_score DOUBLE PRECISION,
      legacy_error TEXT,
      legacy_eval_time_s DOUBLE PRECISION,
      shadow_raw_score DOUBLE PRECISION,
      shadow_score DOUBLE PRECISION,
      shadow_error TEXT,
      shadow_eval_time_s DOUBLE PRECISION,
      score_delta DOUBLE PRECISION,
      status_parity BOOLEAN,
      within_epsilon BOOLEAN,
      legacy_metadata JSONB,
      shadow_metadata JSONB,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
  );
  CREATE INDEX idx_shadow_eval_comparisons_pkg_round
      ON shadow_eval_comparisons (competition_pkg, round_number);

  -- DOWN
  DROP TABLE IF EXISTS shadow_eval_comparisons;
  ```
- [ ] Confirm it is the next sequential number (latest is `011_*`): `ls /Users/giannisevagorou/projects/apex-mvp/src/scheduler/src/scheduler/db/migrations/ | sort | tail -3` → expect `011_...`, then your new `012_...`.
- [ ] (Accumulate into the shadow commit.)

## Task C3: `shadow_store.py` (TDD)

**Files:**
- Create `/Users/giannisevagorou/projects/apex-mvp/src/worker/src/worker/shadow_store.py`
- Test: `/Users/giannisevagorou/projects/apex-mvp/src/worker/tests/test_shadow_dispatch.py` (case (e); file created fully in C6, but the store row-model + write-swallow test is authored here).

**Interfaces:**
- Produces:
  - `class ShadowComparisonRow(BaseModel)` — one field per non-generated table column (`submission_id, competition_pkg, competition_id, round_number, spec_version_id, spec_version, legacy_raw_score, legacy_score, legacy_error, legacy_eval_time_s, shadow_raw_score, shadow_score, shadow_error, shadow_eval_time_s, score_delta, status_parity, within_epsilon, legacy_metadata, shadow_metadata`).
  - `async def record_comparison(row: ShadowComparisonRow) -> bool` — INSERT via a lazy write-capable `DatabaseClient`; returns True on success, swallows+logs on failure (never raises).
  - `async def close() -> None`.
- Consumes: `backend.db.client.DatabaseClient`, `backend.settings.DB_CONNECTION_STRING` (mirrors `spec_resolver.py`).

Steps:
- [ ] In `src/worker/tests/test_shadow_dispatch.py`, author the store tests (this test file grows in C6; start it here):
  ```python
  import json

  from worker import shadow_store


  def _row(**over):
      base = dict(
          submission_id=10, competition_pkg="battleship", competition_id=1, round_number=1,
          spec_version_id=42, spec_version="1.0.0",
          legacy_raw_score=1006.0, legacy_score=0.997, legacy_error="Success", legacy_eval_time_s=1.0,
          shadow_raw_score=0.997, shadow_score=0.997, shadow_error="", shadow_eval_time_s=1.1,
          score_delta=0.0, status_parity=True, within_epsilon=True,
          legacy_metadata={"a": 1}, shadow_metadata={"b": 2},
      )
      base.update(over)
      return shadow_store.ShadowComparisonRow(**base)


  async def test_record_comparison_swallows_write_errors(monkeypatch):
      # No DB configured / connection fails -> must return False, never raise.
      async def boom():
          raise RuntimeError("no db")
      monkeypatch.setattr(shadow_store, "_get_db", boom)
      assert await shadow_store.record_comparison(_row()) is False


  def test_row_model_serializes_metadata_as_json():
      row = _row()
      assert json.loads(json.dumps(row.legacy_metadata)) == {"a": 1}
  ```
- [ ] Run it, expect FAIL: `uv run pytest src/worker/tests/test_shadow_dispatch.py -q` → expect `ModuleNotFoundError: No module named 'worker.shadow_store'`.
- [ ] Implement `src/worker/src/worker/shadow_store.py` (mirrors `spec_resolver.py`'s lazy-handle pattern; write-capable; swallow-on-failure):
  ```python
  """TEMPORARY shadow-comparison writer (competition externalization, Phase 1).

  Part of the shadow subsystem that NEVER merges to main (design §5.4). Own lazy,
  write-capable DB handle (mirrors spec_resolver's pattern). All write failures are
  caught and logged here — record_comparison never raises, so the shadow path stays
  fully isolated from the canonical job.
  """

  from __future__ import annotations

  import asyncio
  import json

  from loguru import logger
  from pydantic import BaseModel
  from sqlalchemy import text

  from backend import settings as backend_settings
  from backend.db.client import DatabaseClient

  _db: DatabaseClient | None = None
  _lock = asyncio.Lock()


  class ShadowComparisonRow(BaseModel):
      submission_id: int
      competition_pkg: str
      competition_id: int
      round_number: int
      spec_version_id: int | None = None
      spec_version: str | None = None
      legacy_raw_score: float | None = None
      legacy_score: float | None = None
      legacy_error: str | None = None
      legacy_eval_time_s: float | None = None
      shadow_raw_score: float | None = None
      shadow_score: float | None = None
      shadow_error: str | None = None
      shadow_eval_time_s: float | None = None
      score_delta: float | None = None
      status_parity: bool | None = None
      within_epsilon: bool | None = None
      legacy_metadata: dict | None = None
      shadow_metadata: dict | None = None


  async def _get_db() -> DatabaseClient:
      global _db
      if _db is not None:
          return _db
      async with _lock:
          if _db is None:
              if not backend_settings.DB_CONNECTION_STRING:
                  raise RuntimeError("shadow recording enabled but DB_CONNECTION_STRING is not set")
              db = DatabaseClient()
              await db.connect(backend_settings.DB_CONNECTION_STRING, pool_size=2, max_overflow=2)
              logger.info("Worker shadow-store DB connection established")
              _db = db
      return _db


  async def record_comparison(row: ShadowComparisonRow) -> bool:
      """INSERT one comparison row. Returns True on success; logs + returns False on any failure."""
      try:
          db = await _get_db()
          async with db.transaction() as session:
              await session.execute(
                  text(
                      """
                      INSERT INTO shadow_eval_comparisons (
                          submission_id, competition_pkg, competition_id, round_number,
                          spec_version_id, spec_version,
                          legacy_raw_score, legacy_score, legacy_error, legacy_eval_time_s,
                          shadow_raw_score, shadow_score, shadow_error, shadow_eval_time_s,
                          score_delta, status_parity, within_epsilon,
                          legacy_metadata, shadow_metadata
                      ) VALUES (
                          :submission_id, :competition_pkg, :competition_id, :round_number,
                          :spec_version_id, :spec_version,
                          :legacy_raw_score, :legacy_score, :legacy_error, :legacy_eval_time_s,
                          :shadow_raw_score, :shadow_score, :shadow_error, :shadow_eval_time_s,
                          :score_delta, :status_parity, :within_epsilon,
                          :legacy_metadata, :shadow_metadata
                      )
                      """
                  ),
                  {
                      **row.model_dump(exclude={"legacy_metadata", "shadow_metadata"}),
                      "legacy_metadata": json.dumps(row.legacy_metadata) if row.legacy_metadata is not None else None,
                      "shadow_metadata": json.dumps(row.shadow_metadata) if row.shadow_metadata is not None else None,
                  },
              )
          return True
      except Exception as e:
          logger.warning(f"shadow_store.record_comparison failed (swallowed): {e}")
          return False


  async def close() -> None:
      global _db
      if _db is not None:
          await _db.disconnect()
          _db = None
  ```
- [ ] Run it, expect PASS: `uv run pytest src/worker/tests/test_shadow_dispatch.py -q` → expect the two store tests pass.
- [ ] (Accumulate into the shadow commit.)

## Task C4: `shadow.py` — comparison engine (TDD)

**Files:**
- Create `/Users/giannisevagorou/projects/apex-mvp/src/worker/src/worker/shadow.py`
- Test: `src/worker/tests/test_shadow_dispatch.py` (comparison-math case (d) authored here).

**Interfaces:**
- Produces:
  - `def _is_success(eval_error: str | None) -> bool` — `eval_error in ("", "Success", None)`.
  - `def compare(legacy: EvaluationResults, shadow: EvaluationResults, epsilon: float) -> dict` — returns `{score_delta, status_parity, within_epsilon}`.
  - `async def run_shadow(job, canonical_results, sandbox_cls) -> None` — resolve spec (miss → `outcome=no_spec` log, return), build a `SoloRunner` like `_build_spec_runner` but WITHOUT `set_result_callback`, run under `asyncio.wait_for(..., timeout=evaluate.timeout_s + 120)`, match shadow↔legacy by `submission_id`, compute `compare`, emit one structured log line per comparison + one `outcome=...` log per run, and call `shadow_store.record_comparison`. Wrapped so it NEVER raises.
- Consumes: `worker.spec_resolver.resolve_active_spec`, `worker.spec_runners.SoloRunner`, `worker.shadow_store`, `worker.settings.{SPEC_DRIVEN_SHADOW_EPSILON, SHARED_VOLUME_PATH_ON_WORKER}`, `backend.eval.base.EvaluationResults`, `common.models.api.job.JobResponse`.

Steps:
- [ ] Author the comparison-math tests (append to `test_shadow_dispatch.py`):
  ```python
  from datetime import datetime, timezone

  from worker import shadow
  from backend.eval.base import EvaluationResults


  def _res(sid, score, err):
      return EvaluationResults(
          submission_id=sid, hotkey="hk", eval_metadata={}, eval_error=err,
          eval_time_in_seconds=1.0, eval_raw_score=score, eval_score=score,
          eval_at=datetime.now(timezone.utc), file_paths={},
      )


  def test_is_success_normalizes_legacy_and_shadow_markers():
      assert shadow._is_success("Success") is True
      assert shadow._is_success("") is True
      assert shadow._is_success(None) is True
      assert shadow._is_success("boom") is False


  def test_compare_delta_status_and_epsilon():
      out = shadow.compare(_res(10, 0.90, "Success"), _res(10, 0.93, ""), epsilon=0.05)
      assert abs(out["score_delta"] - 0.03) < 1e-9
      assert out["status_parity"] is True
      assert out["within_epsilon"] is True

      out2 = shadow.compare(_res(10, 0.10, "Success"), _res(10, 0.90, ""), epsilon=0.05)
      assert out2["within_epsilon"] is False

      out3 = shadow.compare(_res(10, 0.90, "Success"), _res(10, 0.0, "eval blew up"), epsilon=0.05)
      assert out3["status_parity"] is False
  ```
- [ ] Run it, expect FAIL: `uv run pytest src/worker/tests/test_shadow_dispatch.py -k "compare or is_success" -q` → expect `ModuleNotFoundError: No module named 'worker.shadow'`.
- [ ] Implement `src/worker/src/worker/shadow.py`:
  ```python
  """TEMPORARY shadow-mode comparison harness (competition externalization, Phase 1).

  Part of the shadow subsystem that NEVER merges to main (design §5.4). For a battleship
  job, run_shadow runs the spec-driven SoloRunner AFTER the canonical legacy eval has
  already been delivered, compares eval_score, logs, and records a comparison row. It
  never delivers shadow results and never raises — the canonical job is untouched.
  """

  from __future__ import annotations

  import asyncio

  from loguru import logger

  from backend.eval.base import EvaluationResults
  from common.models.api.job import JobResponse
  from worker import shadow_store, spec_resolver
  from worker.settings import SHARED_VOLUME_PATH_ON_WORKER, SPEC_DRIVEN_SHADOW_EPSILON
  from worker.spec_runners import SoloRunner

  _SHADOW_RUN_TIMEOUT_BUFFER_S = 120


  def _is_success(eval_error: str | None) -> bool:
      return eval_error in ("", "Success", None)


  def compare(legacy: EvaluationResults, shadow: EvaluationResults, epsilon: float) -> dict:
      score_delta = float(shadow.eval_score) - float(legacy.eval_score)
      status_parity = _is_success(legacy.eval_error) == _is_success(shadow.eval_error)
      return {
          "score_delta": score_delta,
          "status_parity": status_parity,
          "within_epsilon": abs(score_delta) <= epsilon,
      }


  async def run_shadow(
      job: JobResponse,
      canonical_results: list[EvaluationResults],
      sandbox_cls,
  ) -> None:
      """Run the spec-driven shadow eval, compare against canonical, log + record. Never raises."""
      try:
          spec = await spec_resolver.resolve_active_spec(job.competition_pkg)
          if spec is None:
              logger.info(f"shadow outcome=no_spec pkg={job.competition_pkg}")
              return
          if spec.spec_json.get("kind") != "solo":
              logger.info(f"shadow outcome=no_spec pkg={job.competition_pkg} (kind != solo, unsupported)")
              return

          runner = SoloRunner(
              spec=spec,
              sandbox_cls=sandbox_cls,
              competition_id=job.competition_id,
              round_number=job.round_number,
              competition_pkg=job.competition_pkg,
              submission_ids=job.submission_id,
              hotkeys=job.hotkey,
              codes=job.raw_code,
              shared_volume_path_on_worker=SHARED_VOLUME_PATH_ON_WORKER,
          )
          runner.screening_statuses = job.screening_status
          # Deliberately DO NOT set_result_callback: _emit_result no-ops, so nothing
          # from the shadow run ever reaches _deliver_result / the orchestrator.

          timeout_s = int(spec.spec_json["entrypoints"]["evaluate"]["timeout_s"]) + _SHADOW_RUN_TIMEOUT_BUFFER_S
          shadow_results = await asyncio.wait_for(runner.run(job.input_data), timeout=timeout_s)

          by_sid = {r.submission_id: r for r in shadow_results}
          for legacy in canonical_results:
              shadow = by_sid.get(legacy.submission_id)
              if shadow is None:
                  logger.warning(f"shadow missing result for submission_id={legacy.submission_id}")
                  continue
              cmp = compare(legacy, shadow, SPEC_DRIVEN_SHADOW_EPSILON)
              logger.bind(
                  submission_id=legacy.submission_id,
                  spec_version=spec.version,
                  legacy_score=legacy.eval_score,
                  shadow_score=shadow.eval_score,
                  score_delta=cmp["score_delta"],
                  status_parity=cmp["status_parity"],
                  within_epsilon=cmp["within_epsilon"],
              ).info("shadow comparison")

              await shadow_store.record_comparison(
                  shadow_store.ShadowComparisonRow(
                      submission_id=legacy.submission_id,
                      competition_pkg=job.competition_pkg,
                      competition_id=job.competition_id,
                      round_number=job.round_number,
                      spec_version_id=spec.id,
                      spec_version=spec.version,
                      legacy_raw_score=legacy.eval_raw_score,
                      legacy_score=legacy.eval_score,
                      legacy_error=legacy.eval_error,
                      legacy_eval_time_s=legacy.eval_time_in_seconds,
                      shadow_raw_score=shadow.eval_raw_score,
                      shadow_score=shadow.eval_score,
                      shadow_error=shadow.eval_error,
                      shadow_eval_time_s=shadow.eval_time_in_seconds,
                      score_delta=cmp["score_delta"],
                      status_parity=cmp["status_parity"],
                      within_epsilon=cmp["within_epsilon"],
                      legacy_metadata=legacy.eval_metadata,
                      shadow_metadata=shadow.eval_metadata,
                  )
              )
          logger.info(f"shadow outcome=ok pkg={job.competition_pkg} compared={len(canonical_results)}")
      except asyncio.TimeoutError:
          logger.warning(f"shadow outcome=timeout pkg={job.competition_pkg} submission_id={job.submission_id}")
      except Exception as e:
          logger.warning(f"shadow outcome=error pkg={job.competition_pkg}: {e.__class__.__name__}: {e}")
  ```
- [ ] Run it, expect PASS: `uv run pytest src/worker/tests/test_shadow_dispatch.py -k "compare or is_success" -q` → expect the three tests pass.
- [ ] (Accumulate into the shadow commit.)

## Task C5: `_handle_job` hook + cleanup wiring (TDD)

**Files:**
- Modify `/Users/giannisevagorou/projects/apex-mvp/src/worker/src/worker/worker.py`
- Test: `src/worker/tests/test_shadow_dispatch.py` (cases (a)–(c) authored here).

**Interfaces:**
- Consumes: `worker.settings.SPEC_DRIVEN_SHADOW`, `worker.shadow.run_shadow`, `worker.shadow_store.close`.
- Produces: after the end-of-run sweep in `_handle_job`, `if job.competition_pkg in SPEC_DRIVEN_SHADOW: await shadow.run_shadow(job, eval_results, self._get_selected_sandbox_class())`; `shadow_store.close()` added to `cleanup`.

Steps:
- [ ] Author the hook tests (append to `test_shadow_dispatch.py`) — they drive `_handle_job` with a `Worker.__new__` instance, a stub `execute_job`, a mock orchestrator client, and monkeypatched `shadow.run_shadow`:
  ```python
  import worker.worker as worker_mod
  from worker import settings as worker_settings


  class _Client:
      def __init__(self):
          self.sent = []
          self.rejected = []
      async def send_job_results(self, job_results):
          self.sent.append(job_results)
      async def send_job_file(self, **k):
          pass
      async def reject_submission(self, submission_id, reason):
          self.rejected.append(submission_id)


  def _eval_job(pkg="battleship"):
      return JobResponse(
          competition_id=1, competition_pkg=pkg, round_number=1,
          submission_id=[10], hotkey=["hk"], raw_code=["print(1)"],
          screening_status=[None], input_data={"tasks": []},
      )


  def _result(sid=10):
      return EvaluationResults(
          submission_id=sid, hotkey="hk", eval_metadata={}, eval_error="Success",
          eval_time_in_seconds=1.0, eval_raw_score=1006.0, eval_score=0.997,
          eval_at=datetime.now(timezone.utc), file_paths={},
      )


  async def _run_handle(monkeypatch, pkg, shadow_set, shadow_calls):
      w = worker_mod.Worker.__new__(worker_mod.Worker)
      w.client = _Client()
      w.idle_start_time = 0.0
      monkeypatch.setattr(w, "_get_selected_sandbox_class", lambda: object, raising=False)

      async def fake_execute_job(job, result_callback=None):
          return [_result()]
      monkeypatch.setattr(w, "execute_job", fake_execute_job, raising=False)
      monkeypatch.setattr(worker_mod, "SPEC_DRIVEN_SHADOW", shadow_set)

      async def fake_run_shadow(job, canonical, sandbox_cls):
          shadow_calls.append((job.competition_pkg, [r.submission_id for r in canonical]))
      monkeypatch.setattr(worker_mod.shadow, "run_shadow", fake_run_shadow)

      await w._handle_job(_eval_job(pkg), [])
      return w


  async def test_shadow_not_invoked_when_pkg_not_listed(monkeypatch):
      calls = []
      await _run_handle(monkeypatch, pkg="battleship", shadow_set=set(), shadow_calls=calls)
      assert calls == []


  async def test_shadow_invoked_after_delivery_with_canonical_results(monkeypatch):
      calls = []
      w = await _run_handle(monkeypatch, pkg="battleship", shadow_set={"battleship"}, shadow_calls=calls)
      # canonical delivery happened...
      assert len(w.client.sent) == 1 and w.client.sent[0].submission_id == 10
      # ...and shadow saw the already-computed canonical results.
      assert calls == [("battleship", [10])]


  async def test_shadow_failure_never_breaks_job(monkeypatch):
      w = worker_mod.Worker.__new__(worker_mod.Worker)
      w.client = _Client()
      w.idle_start_time = 0.0
      monkeypatch.setattr(w, "_get_selected_sandbox_class", lambda: object, raising=False)
      async def fake_execute_job(job, result_callback=None):
          return [_result()]
      monkeypatch.setattr(w, "execute_job", fake_execute_job, raising=False)
      monkeypatch.setattr(worker_mod, "SPEC_DRIVEN_SHADOW", {"battleship"})
      async def boom(job, canonical, sandbox_cls):
          raise RuntimeError("shadow exploded")
      monkeypatch.setattr(worker_mod.shadow, "run_shadow", boom)
      # _handle_job must NOT propagate: it catches internally. Delivery already happened.
      await w._handle_job(_eval_job("battleship"), [])
      assert len(w.client.sent) == 1 and w.client.rejected == []
  ```
  (Case (c) — "shadow results never reach the orchestrator" — is covered structurally by C4's no-`set_result_callback` design plus `test_shadow_invoked_after_delivery_with_canonical_results` asserting exactly one delivered result; the fake `run_shadow` cannot deliver.)
- [ ] Run it, expect FAIL: `uv run pytest src/worker/tests/test_shadow_dispatch.py -k "shadow_not_invoked or shadow_invoked or shadow_failure" -q` → expect `AttributeError: module 'worker.worker' has no attribute 'shadow'` (import not yet added) or the hook assertion fails.
- [ ] Modify `worker.py` imports — add `SPEC_DRIVEN_SHADOW` to the `worker.settings` import block and import the shadow modules:
  ```python
  from worker.settings import (
      WORKER_TYPE,
      IDLE_TIMEOUT_IN_SECONDS,
      JOB_SLEEP_INTERVAL_IN_SECONDS,
      MAX_CONCURRENT_SANDBOXES,
      SHARED_VOLUME_PATH_ON_WORKER,
      SANDBOX_BACKEND,
      FILE_BATCH_SIZE,
      SPEC_DRIVEN_ENABLED,
      SPEC_DRIVEN_ROUND_GEN,
      SPEC_DRIVEN_SHADOW,
  )
  ```
  and, next to `from worker import spec_resolver`:
  ```python
  from worker import spec_resolver, shadow, shadow_store
  ```
- [ ] Modify `_handle_job` — after the end-of-run sweep `for` loop and BEFORE `self.idle_start_time = time.time()`, add the shadow hook (still inside the `with logger.contextualize(...)` block):
  ```python
                  # --- Shadow subsystem (competition externalization Phase 1; never merges to main).
                  # Runs AFTER canonical delivery; failures are swallowed inside run_shadow so the
                  # canonical job is never affected.
                  if job.competition_pkg in SPEC_DRIVEN_SHADOW:
                      await shadow.run_shadow(job, eval_results, self._get_selected_sandbox_class())

                  # Only reset the idle start time if the job succeeded
                  self.idle_start_time = time.time()
  ```
- [ ] Modify `cleanup` — add `shadow_store.close()` alongside `spec_resolver.close()`:
  ```python
      async def cleanup(self):
          """Cleanup resources"""
          await self.client.close()
          await s3_service.close()
          await spec_resolver.close()
          await shadow_store.close()
          logger.info("Worker cleanup completed")
  ```
- [ ] Run it, expect PASS: `uv run pytest src/worker/tests/test_shadow_dispatch.py -q` → expect all shadow tests pass.
- [ ] (Accumulate into the shadow commit.)

## Task C6: Full shadow test suite green + regression check

Steps:
- [ ] Confirm the complete `test_shadow_dispatch.py` (cases a–e) passes: `uv run pytest src/worker/tests/test_shadow_dispatch.py -q` → expect all pass.
- [ ] Regression: the Phase-0 dispatch tests still pass and nothing else broke in the worker: `uv run pytest src/worker/tests/ -q` → expect all pass.
- [ ] Lint: `uv run ruff check src/worker/src/worker/shadow.py src/worker/src/worker/shadow_store.py src/worker/src/worker/worker.py src/worker/src/worker/settings.py` → expect no errors.

## Task C7: Make the single shadow commit + record the revert plan

**Files:**
- All Part C files, in ONE commit.

Steps:
- [ ] Stage exactly the shadow artifacts:
  ```
  git add \
    src/worker/src/worker/settings.py \
    src/scheduler/src/scheduler/db/migrations/012_add_shadow_eval_comparisons.sql \
    src/worker/src/worker/shadow_store.py \
    src/worker/src/worker/shadow.py \
    src/worker/src/worker/worker.py \
    src/worker/tests/test_shadow_dispatch.py
  ```
- [ ] Commit as ONE unit (this is the reverted-before-merge "shadow commit"):
  ```
  git commit -m "TEMP shadow subsystem: battleship spec-vs-legacy comparison (revert before merge)

  Adds SPEC_DRIVEN_SHADOW flag, worker/shadow.py, worker/shadow_store.py, migration 012
  (shadow_eval_comparisons), the _handle_job hook, and test_shadow_dispatch.py. Entire
  subsystem is per-migration scaffolding and MUST be git-reverted before this branch
  merges to main (design §5.4). No prometheus metrics; structured logs + DB table only."
  ```
- [ ] Record the commit SHA for reversion: `git rev-parse HEAD` → note it in the PR description as "REVERT THIS before merge".

## Task C8: Stage exercise + "remove before merge" checklist (runbook, not executed now)

Steps (documented for the operator; executed during the shadow round, not at implementation time):
- [ ] Deploy the branch via the `deploy-pr` label; set `SPEC_DRIVEN_SHADOW=battleship` in the PR worker config; confirm the worker deployment has a write-capable `DB_CONNECTION_STRING` (spec §9.1) and that migration 012 applied on the PR-staging DB.
- [ ] Ensure Part B's syncer reconcile populated the battleship spec + stage active pointer in the PR-staging DB (spec §9.2).
- [ ] Submit the reference `submission.py` plus a few variants over one full round.
- [ ] Analyze `shadow_eval_comparisons`: pipeline success rate (≥99%), status-parity rate (≥99%), Spearman(legacy_score, shadow_score) ≥0.95, per-submission |score_delta| ≤0.3 (spec §3.4). If the round is thin, seed extra submissions (spec §9.6).
- [ ] **Remove before merge:** `git revert <shadow-commit-SHA>` (single revert removes flag, `shadow.py`, `shadow_store.py`, migration 012, the `_handle_job` hook, and the tests together; worker returns bit-for-bit to pre-shadow state). Confirm migration 012 never merged to `main`. PR-staging teardown (`cleanup-pr-staging.yml`) drops the DB.
- [ ] Merge only the permanent deliverables to `main`: the design doc + this plan (repos 1–2 live in their own repos). Nothing shadow-related reaches `main`.

---

# Self-Review

**Spec-section coverage:**
- §1 Overview / "no behavior change" → Global Constraints + C5 hook placement (after delivery) + C4 no-callback + swallow-all. ✔
- §2 Decomposition/ordering → Parts A→B→C ordering; B1 gates on A10 digest; C8 gates on B1 syncer. ✔
- §3.1 spec.yaml concrete values → A9 (verbatim, with the §9.4 FILL markers preserved). ✔
- §3.2 evaluate.py (subprocess, health-poll, 10s startup, 30s cap, result.json) → A6. ✔
- §3.3 score-scale (image emits 0–1 raw_score; unnormalized in metadata; compare eval_score) → A6 `run_round` + C4 `compare` uses `eval_score`. ✔
- §3.4 determinism (honor seed; statistical gate) → A3 (engine already honors seed) + A9 determinism test + C8 analysis criteria. ✔
- §3.5 screening scoped out (screener shipped not wired) → A8. ✔
- §4 registry files + digest ordering → B1. ✔
- §5.1 flag + lazy gate → C1 (+ Self-Review note on the gate). ✔
- §5.2 hook after sweep, no callback, wait_for(timeout+120), swallow → C4 + C5. ✔
- §5.3 comparison (delta, status normalization, epsilon, raw excluded) → C4 `compare` (raw scores stored but not compared). ✔
- §5.4 single revertable commit + exact SQL → C2 (SQL) + C7 (one commit) + C8 (revert). ✔
- §5.5 structured logs only, NO metrics → C4 emits two log kinds; Global Constraints forbids metric series; no metrics.py touched. ✔
- §6 tests (a–e) + repo-1 scoring/determinism + apex-dev run → A4/A6/A9 + C3/C4/C5/C6. ✔
- §7 rollout/rollback → C7 SHA record + C8 runbook. ✔
- §8 out of scope → not implemented (round-gen shipped-not-activated A7; duel untouched; SPEC_DRIVEN_ENABLED stays false). ✔
- §9 open questions → surfaced as explicit runbook checks (C8) + preserved FILL markers (A9/B1). ✔

**Placeholder scan:** The only non-literal values are `spec.yaml`'s `image.digest` (filled by CI, A10) and the five §9.4 `# FILL from live stage config` fields — both are genuinely runtime-unknown per spec §9 and are marked as explicit "FILL at <step>" markers, exactly as the spec authorizes. No other TBD/TODO placeholders; all code is shown in full except the two verbatim-vendored files (`battleship_engine.py` via A3 with the exact deletion range shown; `submission.py` via A5 verbatim; `ast_guard.py` via A8 verbatim), which the plan explicitly authorizes as copy-from-source.

**Type/name consistency:** `SPEC_DRIVEN_SHADOW` (set[str]) and `SPEC_DRIVEN_SHADOW_EPSILON` (float) defined in C1, imported in C4 (`shadow.py`) and C5 (`worker.py`) identically. `ShadowComparisonRow` fields (C3) are a 1:1 subset of the migration-012 columns (C2), minus generated `id`/`created_at`; the INSERT column list in C3 matches. `run_shadow(job, canonical_results, sandbox_cls)` signature is identical in C4 (def), C5 (call site + fake), and the spec §5.2 snippet. `SoloRunner` constructor kwargs in C4 match `_build_spec_runner`'s kwargs in `worker.py` (verified against source: `spec, sandbox_cls, competition_id, round_number, competition_pkg, submission_ids, hotkeys, codes, shared_volume_path_on_worker`) and it sets `screening_statuses` but NOT `set_result_callback` (the isolation guarantee). `EvaluationResults` field names used in `compare`/row (`eval_score, eval_raw_score, eval_error, eval_time_in_seconds, eval_metadata, submission_id`) match `backend.eval.base.EvaluationResults`. `_is_success` normalization (`"", "Success", None`) matches legacy `BattleshipEvaluation` (`eval_error="Success"`) vs `SoloRunner` (`eval_error=""`). Win string `"Player 1 won"` in A6 matches engine `f"{Name.PLAYER_1.value} won"`. Scoring `base=size**2*10`, `speed_bonus=max(size**2-turns,0)*0.1`, `/1009` match `runner.py`/`normalizer.py`.

**Deviations from literal spec wording (intentional, noted):**
1. §5.1 says "add SPEC_DRIVEN_SHADOW to spec_resolver's lazy-connect gate." `spec_resolver.py` has no internal flag check (it connects on demand; the gate is caller-side). Faithful equivalent implemented: the `run_shadow` call is guarded by `if job.competition_pkg in SPEC_DRIVEN_SHADOW` in `_handle_job` (C5), mirroring the existing `SPEC_DRIVEN_ENABLED`/`SPEC_DRIVEN_ROUND_GEN` caller-side pattern. No `spec_resolver.py` edit needed. This preserves the exact runtime behavior the spec intends (no DB connection unless a shadow pkg is configured).
2. `evaluate.py` uses a `ThreadPoolExecutor().submit(...).result(timeout=30)` for the per-game wall cap (the spec says "thread + timeout"). Equivalent; the thread is not force-killed on timeout (Python can't), but the game is scored as a loss and the subprocess is terminated in `main`'s `finally`, so the outcome matches the spec's "timeout ⇒ loss, 0 points."
