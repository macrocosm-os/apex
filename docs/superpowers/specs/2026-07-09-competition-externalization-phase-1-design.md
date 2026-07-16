# Competition Externalization — Phase 1 Design: Battleship Shadow Mode

**Status:** Approved — ready for implementation planning
**Date:** 2026-07-09
**Scope:** Three repos — `apex-competition-battleship` (new, public), `apex-competitions-registry` (private GitOps), `apex-mvp` (this repo).

## 1. Overview & Goals

Phase 1 ships the first real externalized competition — battleship — in **shadow mode**. For every battleship eval job on stage, the worker runs the legacy `BattleshipEvaluation` (canonical: its results are what get delivered to the orchestrator, scored, and ranked) **and** the Phase-0 `SoloRunner` against a published `apex.competition.v1` spec (shadow: results are compared against legacy, recorded, and discarded).

"No behavior change" means, concretely:

- The orchestrator receives exactly the results it receives today — same scores, same files, same timing semantics. Shadow results are never delivered.
- A shadow failure of any kind (spec resolution, image pull, sandbox crash, malformed `result.json`, DB write failure) is logged and swallowed; it can never fail, delay-before-delivery, or alter the canonical job.
- Prod is untouched: the spec is activated in `active/stage.yaml` only, and `SPEC_DRIVEN_SHADOW` defaults empty everywhere.

What Phase 1 proves: the full external pipeline (public competition repo → CI build/sign → registry activation → syncer mirror → worker spec resolution → generic `SoloRunner` → `result.json` contract) produces scores consistent with the legacy in-process evaluator, for a competition that is deliberately *hard* to port (miner submission is an HTTP server, host-side game loop, stochastic scoring).

## 2. Decomposition & Dependency Order

Three repos, strict ordering because each stage consumes an artifact of the previous:

```
[1] apex-competition-battleship (NEW, public)
    build image in CI → push by digest → cosign-sign
        │  (digest known only after this)
        ▼
[2] apex-competitions-registry (private GitOps)
    competitions/battleship/1.0.0.yaml (embeds digest) + input.schema.json
    active/stage.yaml: battleship: "1.0.0"
        │  syncer validates + verifies + crane-mirrors + upserts DB rows
        ▼
[3] apex-mvp
    SPEC_DRIVEN_SHADOW=battleship → worker resolves spec → runs shadow → compares
```

Repo 3's code can be written and unit-tested in parallel (mocked specs), but end-to-end shadow on stage requires 1→2 done first.

## 3. Repo 1 — `apex-competition-battleship` (new public repo)

Copy the SDK's `examples/hello-world/` layout and extend it:

```
apex-competition-battleship/
├── spec.yaml
├── input.schema.json          # derived from BattleshipEvalInputDataSchema
├── player/
│   ├── Dockerfile
│   ├── evaluate.py            # in-image orchestration (the hard part)
│   ├── battleship_engine.py   # vendored copy of shared/competition .../battleship.py
│   ├── scoring.py             # port of _calculate_game_score + /1009 normalizer
│   ├── generate_round.py      # port of BattleshipInputDataGenerator (numpy only)
│   ├── screener.py            # ported ASTGuard-based screener (shipped, not wired — see §3.5)
│   └── submission.py          # reference miner = vendored baseline.py (FastAPI RandomShooter)
├── fixtures/
│   └── input.json             # small deterministic round: 2 tasks, fixed seeds, low max_turns
├── tests/                     # scoring parity + determinism unit tests
├── .github/workflows/release.yml   # build → push by digest → cosign sign (OIDC keyless)
└── README.md
```

### 3.1 `spec.yaml` (concrete values)

```yaml
schema: apex.competition.v1
id: battleship                      # MUST equal competition_pkg; resolver keys on this
version: 1.0.0
display_name: Battleship
kind: solo
process_type: cpu
resources:
  cpu_limit: 2                      # server + game loop share one container now
  mem_limit: 1024Mi                 # FastAPI + engine are light; under stage 2048Mi ceiling
  gpu_count: 0
image:
  ref: ghcr.io/macrocosm-os/apex-competition-battleship
  digest: sha256:<filled by release CI at repo-1 tag time>
submission:
  artifact_type: code
  max_size_mb: <filled from live stage battleship config before writing spec — §9.4>
  target_path: /workspace/submission.py
input_schema:
  $ref: input.schema.json
defaults:
  baseline_score: <filled from live stage config — §9.4>         # 0–1 scale
  baseline_raw_score: <filled from live stage config — §9.4>     # NEW 0–1 scale, see §3.3
  round_length_in_days: <filled from live stage config — §9.4>
  submission_reveal_days: <filled from live stage config — §9.4>
  lower_is_better: false
entrypoints:
  evaluate:
    command: ["python", "/app/evaluate.py"]
    timeout_s: 300                  # 5 tasks × 30s/game cap + startup + margin
    network_disabled: true          # loopback only — see §3.2
    allow_internet: false
  generate_round:
    command: ["python", "/app/generate_round.py"]
    timeout_s: 120
signature:
  cosign_identity: https://github.com/macrocosm-os/apex-competition-battleship/.github/workflows/release.yml@refs/tags/*
  cosign_issuer: https://token.actions.githubusercontent.com
```

`input.schema.json`: top-level `{tasks: [{task_name, input: {size, ships_spec, repeat_on_hit, enforce_no_touching, startup_health_check_timeout_in_seconds, board_generation_timeout_in_seconds, shot_timeout_in_seconds, starting_player, max_turns, seed}}]}` with the defaults from `BattleshipInputDataSchema` — a mechanical translation of the pydantic models.

### 3.2 `player/evaluate.py` — the port of the host-side game loop (tension 3)

Legacy splits responsibilities: the sandbox runs the miner's HTTP server; the *worker process* runs `run_game` against the pod IP. The port collapses both into one sandbox. `evaluate.py`:

1. Read `/data/input.json` → `tasks[]`.
2. `subprocess.Popen(["python", "/workspace/submission.py", "--port", "8001"])` — same argv contract as legacy (`["python","solution.py","--port","8001"]`), so existing miner submissions work unmodified. Stdout/stderr inherit to container logs (captured by the sandbox log file, same as today).
3. Health-poll `GET http://127.0.0.1:8001/health` up to a 10s startup budget (matching the legacy runner's effective 10s startup health-check window; note that the per-task schema default of 1s is *not* the startup budget).
4. Per task: run vendored `run_game(p1_url="http://127.0.0.1:8001", ..., seed=task.input.seed)` with a hard **30s per-game wall cap** (replicating `GAME_TIMEOUT_BUFFER_SECONDS` via a thread + timeout; timeout ⇒ loss, 0 points), score via `scoring.py`.
5. Aggregate: `final = mean(per_game_raw) / 1009.0` (see §3.3), write `/data/result.json`:
   ```json
   {"raw_score": <normalized mean, 0–1>, "eval_time_in_seconds": <sum of game wall times>,
    "metadata": {"Game 1": {"game_result": "...", "turns": 0, "success": false, "seed": 0, "score_breakdown": {}}, "unnormalized_raw_score": 0.0}}
   ```
   Per-game metadata mirrors legacy `eval_metadata` for debuggability; full board histories are **not** emitted (legacy uploads them as HISTORY files — out of the `result.json` contract; tracked as a Phase-2 gap, see §9.5).
6. Terminate the subprocess. Exit 0 whenever `result.json` was written — a lost game is a valid 0-score result, not an error. Exit non-zero only on infra failure (server never healthy, engine crash), which `SoloRunner._read_result` surfaces as `eval_error`.

**Bundling:** `battleship_engine.py` is a vendored copy of `shared/competition/src/competition/battleship/battleship.py` (deps: requests, pydantic — self-contained), with one behavioral edit: it honors the passed `seed` (§3.4). `scoring.py` ports `_calculate_game_score` and the `/1009` normalizer verbatim; a unit test asserts numeric parity against a table of (turns, success) cases generated from the legacy code. The public repo must not import the private monorepo, and the engine is ~1 file, so vendoring is the correct trade-off over a shared dependency.

**Networking:** the miner server is a *subprocess in the same container*, so all traffic is loopback within one network namespace. `network_disabled: true` / `network_mode: none` still provides `lo`, on both Docker and K8s backends. This is strictly *better* isolation than legacy (which needed `network_disabled=False` + bridge for worker→pod traffic). The `apex-dev run` validation step confirms loopback-under-`none` locally; if a backend quirk surfaces, the fallback is `network_disabled: false, allow_internet: false` (still no egress).

`player/Dockerfile`: `python:3.12-slim`, non-root uid 1000 (matching the legacy sandbox Dockerfile), `pip install fastapi uvicorn pydantic requests numpy`, copy `evaluate.py`/`battleship_engine.py`/`scoring.py`/`generate_round.py` into `/app`. Nothing baked under `/data`; `/workspace/submission.py` arrives as a file-level mount (Phase 0 SoloRunner convention), so `evaluate.py` living in `/app` avoids any mount shadowing.

`generate_round.py`: mechanical port of `BattleshipInputDataGenerator` (numpy-only, already self-contained), writing `generated_tasks.json = {tasks, sandbox_data}` to the shared mount per the Phase 0 round-gen contract. The entrypoint ships, but `SPEC_DRIVEN_ROUND_GEN` stays **off** for battleship in Phase 1 — flipping round-gen is a real behavior change and dilutes the "shadow only" guarantee; it is a one-line flag flip in a later phase.

### 3.3 Score-scale mismatch — settled

Legacy emits two scales: `eval_raw_score ≈ 0..1009` and `eval_score = raw/1009 ∈ [0,1]`. `SoloRunner` has one number: `eval_raw_score = eval_score = raw_score`.

**The image emits the normalized 0–1 value as `raw_score`.** Rationale (one line each):

- Ranking is computed from `eval_score`; making the image emit the ranking-relevant number means Phase-2 cutover changes *nothing* rank-visible.
- Option A (no platform normalization) is already locked in Phase 0; normalization therefore lives inside the image.
- The 1009-scale value is preserved in `metadata.unnormalized_raw_score` for dashboards/forensics.

**Comparison scope:** shadow compares `eval_score` only. `eval_raw_score` divergence (legacy ~1009-scale vs shadow 0–1) is *expected and excluded* — both values are still recorded in the comparison table for inspection.

**Phase-2 prerequisite:** at cutover, `eval_raw_score` in the DB changes scale for battleship; `defaults.baseline_raw_score` and any raw-score dashboard/endpoint consumers (e.g., the `/miners` raw-score field) must be audited and re-baselined then. This is a Phase-2 prerequisite, not Phase-1 work.

### 3.4 Non-determinism — settled

Legacy **ignores** `task.input.seed` and rolls `random.randint(0, 2**32-1)` per game; boards (and the baseline shooter) are stochastic. Bit-exact per-submission parity is unachievable regardless of what the port does.

**The port HONORS `task.input.seed` (deterministic given input), and Phase 1's gate is pipeline validation with a statistical score check, not per-submission parity.** Rationale: honoring the seed is the *correct* long-term behavior (reproducible evals, and the generator already emits per-task seeds that legacy silently drops); replicating a bug into a public reference repo is worse than accepting distribution-level comparison.

**Success criteria** (these replace the design doc's "≥99% within epsilon"):

1. **Pipeline:** ≥99% of shadow runs over one full stage round complete without infra error (spec resolved, image ran, `result.json` valid).
2. **Status parity:** legacy and shadow agree on success-vs-error/forfeit for ≥99% of submissions (normalizing legacy's `eval_error="Success"` vs shadow's `""`).
3. **Score consistency (statistical, manual gate):** over the round, Spearman rank correlation between legacy and shadow `eval_score` ≥ 0.95, and mean |Δeval_score| within the noise band estimated from legacy's own game-to-game variance (computed from the comparison table; a per-submission |Δ| ≤ 0.3 sanity bound catches gross porting bugs such as a broken normalizer).

The `SPEC_DRIVEN_SHADOW_EPSILON` threshold (§5.3) remains a *generic* comparison field (default 0.05) recorded per row and logged for future deterministic competitions; for battleship the `within_epsilon` flag is advisory and the comparison-table analysis is the gate.

### 3.5 Screening-verdict parity — scoped out (structural)

Screening runs *upstream*: the orchestrator screens at job build time and ships verdicts as `job.screening_status`; both `BattleshipEvaluation` and the shadow `SoloRunner` consume the **same** list (SoloRunner forfeits on `"failed"`), so worker-level verdict parity holds by construction and there is nothing to compare in Phase 1. The spec schema has no screener entrypoint yet — spec-driven screening is a later phase. The repo ships `player/screener.py` (ported ASTGuard rules) as the reference asset the future phase will wire. Design open-Q#2 is answered as "deferred: screening is not externalized in Phase 1; parity is structural."

## 4. Repo 2 — `apex-competitions-registry`

Files added (mirroring hello_world):

- `competitions/battleship/1.0.0.yaml` — copy of repo 1's `spec.yaml` with `image.digest` filled in.
- `competitions/battleship/input.schema.json` — sibling copy (syncer requires it next to the version file).
- `active/stage.yaml` — add `battleship: "1.0.0"` under `competitions:`. **`active/prod.yaml` untouched.**

**Ordering (digest chicken-and-egg):** the digest cannot exist before repo 1's release CI runs. Flow: (1) tag repo 1 → CI builds, pushes by digest, cosign-signs, prints digest; (2) open registry PR embedding that digest; (3) syncer validates via SDK `load_spec` + `check_resource_ceilings` (spec fits stage ceilings: cpu 2 ≤ 2, mem 1024Mi ≤ 2048Mi), cosign-verifies against the declared identity/issuer, crane-mirrors by digest, upserts `CompetitionSpecVersion` + stage `CompetitionActiveVersion`. Repo 1's CI job emits a ready-to-paste registry snippet (yaml with digest) as a build artifact to make step 2 mechanical.

## 5. Repo 3 — `apex-mvp` shadow machinery

### 5.1 Flag

`src/worker/src/worker/settings.py`, next to the Phase-0 flags:

```python
# Comma-separated competition pkgs that ALSO run the spec-driven runner in shadow mode
# after the canonical legacy eval. Legacy stays canonical; shadow results are compared,
# recorded, and discarded. Independent of SPEC_DRIVEN_ENABLED (full cutover).
SPEC_DRIVEN_SHADOW = {p.strip() for p in os.getenv("SPEC_DRIVEN_SHADOW", "").split(",") if p.strip()}
```

`spec_resolver`'s "only connect when a spec flag is on" lazy gate is extended to include this flag (one-line condition change).

### 5.2 Execution hook (`worker.py`)

No change to `_resolve_runner` / `execute_job` / `_deliver_result`. The hook goes in **`_handle_job`**, *after* the canonical delivery loop completes (after the end-of-run sweep, before the idle-clock reset):

```python
if job.competition_pkg in SPEC_DRIVEN_SHADOW:
    await shadow.run_shadow(job, eval_results, self._get_selected_sandbox_class())
```

Design points:

- **After delivery, not concurrent.** Canonical results are already sent to the orchestrator before the shadow starts, so shadow cannot delay or corrupt delivery even in pathological cases. It also serializes sandbox usage (legacy battleship cleans its sandbox in `finally`), avoiding pod-name collisions and keeping `MAX_CONCURRENT_SANDBOXES` meaningful — the job slot is simply held longer, which is acceptable on stage. Simplicity and isolation are chosen over latency.
- **`worker/shadow.py` (new module — part of the shadow subsystem, which never merges to `main`; see §5.4):** `run_shadow(job, canonical_results, sandbox_cls)` — (1) resolve spec via `spec_resolver.resolve_active_spec` (miss → debug log, return); (2) build a `SoloRunner` exactly as `_build_spec_runner` does, but **never call `set_result_callback`** (so `_emit_result` no-ops — nothing reaches `_deliver_result`); (3) `await asyncio.wait_for(runner.run(job.input_data), timeout=evaluate.timeout_s + 120)` so a hung shadow cannot pin the slot; (4) compare, log, record. The whole body is wrapped in `try/except Exception` — any failure emits an `outcome=error` structured log line and returns. This is the hard isolation guarantee.
- Non-eval job types (`ROUND_GENERATION`, `ONNX_CONVERSION`) return from `_handle_job` before the hook; shadow applies to eval jobs only.

### 5.3 Comparison computation

Per canonical result, matched to the shadow result by `submission_id`:

- `score_delta = shadow.eval_score - legacy.eval_score`; `abs_delta = |score_delta|`.
- `status_parity`: both "success" (`eval_error in ("", "Success", None)`) or both non-success.
- `within_epsilon = abs_delta <= SPEC_DRIVEN_SHADOW_EPSILON` (new env, float, default `0.05`).
- Raw scores are recorded but **not** compared (§3.3).

### 5.4 The shadow subsystem is per-migration scaffolding — never merges to `main`

**The entire shadow feature — the `SPEC_DRIVEN_SHADOW` flag, `worker/shadow.py`, the comparison logic, the DB table, `shadow_store.py`, migration 012, and the shadow unit tests — lands as ONE self-contained commit ("the shadow commit") on the migration branch and is `git revert`-ed before that branch merges to `main`.** Nothing shadow-related is ever resident on `main`.

Rationale (decided during review): shadow is a one-time validation ritual per competition, not a standing platform feature. Keeping it off `main` means (a) no dormant, flag-gated code drifting out of sync across the Phase 2–4 releases; (b) nothing shadow-related left to clean up at Phase 5; (c) the harness is still reused for the 5 remaining solo ports by cherry-picking this commit onto each migration branch. (Tron/duel needs a *different* per-game comparison against `DuelRunnerGymV1` anyway, so a resident solo harness wouldn't serve it.) The one accepted cost is re-applying the commit per competition instead of it sitting ready on `main`.

The experiment runs on a **PR-staging deployment** (`deploy-pr` label), whose scheduler applies migration 012 to the PR-staging DB — the table (and every other shadow artifact) exists only there, never in stage/prod/main. The shadow commit contains exactly:

1. `src/scheduler/src/scheduler/db/migrations/012_add_shadow_eval_comparisons.sql` (`-- UP` creates, `-- DOWN` drops):

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

2. `src/worker/src/worker/settings.py` — the `SPEC_DRIVEN_SHADOW` flag (§5.1) and `SPEC_DRIVEN_SHADOW_EPSILON` (§5.3), plus the one-line addition of `SPEC_DRIVEN_SHADOW` to `spec_resolver`'s lazy-connect gate.
3. `src/worker/src/worker/shadow.py` — `run_shadow` (§5.2) + the comparison computation (§5.3) + the structured logging (§5.5).
4. The `_handle_job` hook (the two lines in §5.2).
5. `src/worker/src/worker/shadow_store.py` — pydantic row model + `record_comparison(...)` with its **own** lazy write-capable `DatabaseClient` (mirroring `spec_resolver`'s pattern; `close()` on shutdown). Write failures are caught inside `record_comparison` and logged — never raised. Called from `shadow.py` inside the comparison step's own `try/except`.
6. `src/worker/tests/test_shadow_dispatch.py` — the shadow unit tests (§6).

**"Remove before merge" checklist:**

- Revert the shadow commit (`git revert <sha>`) — one revert removes the flag, `shadow.py`, `shadow_store.py`, migration 012, the `_handle_job` hook, and the tests together. The worker returns bit-for-bit to its pre-shadow state and still compiles.
- Confirm migration 012 never merged to `main` (renumber risk moot; the next real migration reclaims `012`).
- The PR-staging DB is torn down by `cleanup-pr-staging.yml`.

### 5.5 Observability — structured logs only (no dedicated metrics)

Because the shadow subsystem never merges to `main` (§5.4), we deliberately do **not** add Prometheus metric series for it: standing up `apex_worker_shadow_*` counters/histograms (and the Grafana panels/alerts to read them) is infrastructure investment in throwaway per-migration scaffolding, and the DB comparison table (§5.4) is already the analysis surface.

Instead, `run_shadow` emits exactly two things:

- **One structured log line per comparison** (`logger.bind`: `submission_id`, `spec_version`, `legacy_score`, `shadow_score`, `score_delta`, `status_parity`, `within_epsilon`) — grep/Loki-able, and the primary operational signal during the shadow round.
- **One `outcome=...` structured log line per shadow run** (`ok` / `error` / `timeout` / `no_spec`) — a simple Loki query over `outcome="error"` gives the failure rate without any metric series.

If, during a shadow round, we find we want a live rate panel, a Loki log-metric query covers it without code — no exported metrics are added by this work.

**What reaches `main` from this work:** nothing shadow-related. Only the permanent, non-shadow deliverables merge — for Phase 1 that is the design doc and the reference competition/registry artifacts (repos 1–2). The whole shadow subsystem is reverted first (§5.4). Each later solo migration re-applies this shadow commit onto its own branch, validates, and reverts it the same way.

## 6. Testing

**Repo 1:** unit tests for scoring parity (ported `scoring.py` vs a golden table generated from legacy `_calculate_game_score`, including the `/1009` normalizer) and determinism (same seed ⇒ identical hidden board); `apex-dev preflight` (schema + ceilings) and `apex-dev run --submission player/submission.py --input fixtures/input.json` asserting a valid `result.json` with `raw_score` in the RandomShooter's expected band — this also validates the loopback-under-`network:none` question locally.

**Repo 3:** `src/worker/tests/test_shadow_dispatch.py` mirroring `test_spec_dispatch.py`:

- (a) pkg not in `SPEC_DRIVEN_SHADOW` ⇒ shadow never invoked;
- (b) shadow runner exception / timeout / resolution failure ⇒ job result unchanged, an `outcome=error`/`timeout`/`no_spec` log line emitted, no raise;
- (c) shadow results never reach `_deliver_result` / the orchestrator client mock;
- (d) comparison math (delta, status normalization incl. `"Success"` vs `""`, epsilon);
- (e) `record_comparison` row shape + write-failure swallowing.

**Stage exercise:** deploy the branch via the `deploy-pr` label; set `SPEC_DRIVEN_SHADOW=battleship` in the PR worker config; seed the battleship spec + active pointer into the PR-staging DB (run the syncer against it, or copy the two rows from stage — operational open item, §9.2); submit the reference `submission.py` plus a few variants over a full round; then run the analysis SQL over `shadow_eval_comparisons` (match rate, mean |Δ|, Spearman via a small script) against the §3.4 criteria.

## 7. Rollout & Rollback

**Rollout order:** (1) repo 1 tag → image built/signed, digest captured; (2) registry PR → syncer mirrors + activates in stage DB; (3) apex-mvp migration branch PR-staging deploy with `SPEC_DRIVEN_SHADOW=battleship`; (4) run one full round; (5) analyze against §3.4 criteria; (6) revert the shadow commit (§5.4) so nothing shadow-related merges; (7) merge only the permanent, non-shadow deliverables to main.

**Rollback (any point, independent levers):**

- Unset `SPEC_DRIVEN_SHADOW` (immediate, next job pull — canonical path was never touched).
- Remove `battleship` from `active/stage.yaml` (spec resolution returns None, shadow no-ops with `outcome=no_spec`).
- PR-staging teardown drops the DB.

`SPEC_DRIVEN_ENABLED` remains `false` throughout, so even a resolvable spec never becomes canonical.

## 8. Out of Scope

Phase 2 cutover (`SPEC_DRIVEN_ENABLED` for battleship, raw-score re-baselining, HISTORY-file gap, legacy code deletion); duel competitions / `DuelRunnerGymV1` (Phase 3); spec-driven screening; porting any other competition; prod activation; `SPEC_DRIVEN_ROUND_GEN` flip for battleship (the entrypoint ships, the flag stays off).

## 9. Open Questions / Risks

1. **Worker DB write access** — `shadow_store` assumes the worker's conn string can write; verify on PR-staging, else add a scoped write credential (temporary, part of the shadow commit's config).
2. **Spec seeding into PR-staging DB** — the syncer currently targets stage; decide between pointing a syncer run at the PR DB vs. a manual row copy. Small, but blocks the e2e test.
3. **Loopback under `network: none` on the K8s sandbox backend** — expected fine (single container), validated by `apex-dev run` locally but confirm once on the cluster; fallback documented (§3.2).
4. **`max_size_mb` / `defaults.*` values** — copy from the live stage battleship competition config before writing the spec (the `<filled from live stage config>` markers in §3.1).
5. **HISTORY files** — the shadow/spec path emits per-game metadata but not the `game_*.json` history uploads; acceptable for shadow, a real gap to close before Phase 2 cutover.
6. **Round size vs. statistics** — Spearman ≥ 0.95 needs a reasonable submission count; if the stage round is thin, seed extra submissions (the existing load-test tooling can help).
7. **Slot occupancy** — sequential shadow roughly doubles battleship job wall time on stage; fine at stage volume, but note it if `MAX_CONCURRENT_SANDBOXES` tuning matters during the round.

## 10. Platform follow-ups (found during rollout)

Surfaced while wiring battleship through the real spec-syncer on PR-staging. These are **platform (apex-mvp) fixes**, tracked here so they aren't lost:

- [ ] **(PHASE 2 PREREQUISITE) Spec-syncer cosign verify can't match identity globs.** `spec-syncer/images.py::verify_signature` runs `cosign verify --certificate-identity <spec.signature.cosign_identity>` — an **exact string match**. But specs are authored (per the SDK/schema/design) with a glob like `…/release.yml@refs/tags/*`, while the real keyless certificate identity is the concrete tag (`…@refs/tags/v1.0.0`). cosign does **not** expand `*` for `--certificate-identity`, so verification fails with *"none of the expected identities matched"* and the spec version is skipped (never inserted into `competition_spec_versions`). hello_world never caught this — its spec points at a bare `unsigned-test-image` identity, so real glob-matching was never exercised; battleship is the first real signed image.
  - **Fix:** switch `verify_signature` to `--certificate-identity-regexp`, translating the spec's glob to a regexp (`*` → `.*`), so specs keep the version-agnostic `@refs/tags/*` form the design intends. Rebuild + redeploy the `spec-syncer` image. Verified locally: `--certificate-identity-regexp '…@refs/tags/.*'` **passes** on the battleship image, while the exact glob fails.
  - **Why it's a Phase 2 prerequisite:** until this lands, every spec-driven competition must pin an exact per-version tag identity (as battleship 1.0.0 does via the §4.2 stopgap). That's tolerable for one hand-driven Phase 1 competition, but Phase 2 (stage→prod cutover + porting the remaining competitions) should not rely on the manual per-version pin — do the regexp fix first so specs can use the intended `@refs/tags/*` glob.
  - **Interim stopgap (in use):** `competitions/battleship/1.0.0.yaml` pins the exact identity `…/release.yml@refs/tags/v1.0.0`, which the current exact-match syncer accepts. Once the regexp fix lands, registry files can return to `@refs/tags/*` and the battleship repo's `spec.yaml` template (which still uses the glob, correctly) needs no per-version override.

- [ ] **`get_job` poisons a submission into an infinite loop when its code object is missing.** In `orchestrator/worker/job.py::get_job`, the submission is dequeued from Redis and committed to `EVALUATING` **before** the S3 fetch of `submission.code_path` (`s3_service.retrieve_text_file`). If the object is missing (`NoSuchKey`) or unreadable, `get_job` raises → the worker's `GET /worker/job` returns **HTTP 500**, but the Redis item is already popped and the row is already `EVALUATING`. Nothing rolls it back or rejects it, so the submission is stuck `EVALUATING`, the scheduler re-queues it after `JOB_QUEUE_TIMEOUT` (600s), and it loops forever — one 500 per cycle, no sandbox ever created. Found on staging when a demo seeded rows whose `code_path` objects weren't in the `apex-competitions-stage` bucket.
  - **Fix:** treat a missing/unreadable code artifact as a terminal failure of that submission — reject it (or roll back to a re-queueable state with a bounded retry count) inside `get_job`, instead of raising an unhandled 500. Fetching the code before committing `EVALUATING` (or rejecting on `NoSuchKey`) both avoid the poison loop.
  - **Impact / priority:** any bad `code_path` (missing object, wrong bucket/env, lifecycle-pruned) wedges a worker slot on a re-queue treadmill and spams 500s. Not spec-driven-specific (the S3 read is shared, before the runner is chosen) — squeeze into a Phase 2/3 platform PR.

- [x] **Competition image package must be pullable by the syncer.** The syncer's `crane`/`cosign` must pull the competition image to mirror + verify it; a **private** GHCR package fails the pull (401). Resolved by making the `apex-competition-battleship` container package public (design intends public competition images anyway); alternative would be granting the syncer read access to the package.

- **Resolved during rollout (no action):** getting the spec into the PR-staging DB (design §9.2) is handled by the `pr` env itself — the `pr` spec-syncer reconciles `active/pr.yaml` into the per-PR Neon branch DB (`ENV=pr`); no manual row copy needed. Migration 012 lands in that per-PR Neon branch, not the stage DB.
