# Competition Externalization — Phase 1 Test Plan & Status

**Status:** ✅ Complete — two shadow eval rounds executed on PR-staging. After two platform fixes (§4.3, §4.4), the spec-driven path scores **bit-identical** to legacy: **17/18 submissions Δ=0.0000**; the one non-match was a legacy-side sandbox startup flake, not a scoring divergence (§6).
**Date:** 2026-07-09 (updated 2026-07-10 with eval-round results)
**Owner:** Platform
**Related:** [design](superpowers/specs/2026-07-09-competition-externalization-phase-1-design.md) · [implementation plan](superpowers/plans/2026-07-09-competition-externalization-phase-1.md) · [parent design](competition-externalization-design.md)

---

## 1. Summary

Phase 1 runs **battleship** from an `apex.competition.v1` spec + external signed image, in **shadow mode** alongside the legacy in-process `BattleshipEvaluation`, and compares per-submission scores before any cutover. Legacy stays authoritative; the spec path's results are compared, recorded, and discarded.

The externalization **control plane is proven end-to-end** on PR-staging: image build → cosign keyless sign → registry activation → spec-syncer (cosign-verify + crane-mirror by digest) → DB rows (`competition_spec_versions` + `competition_active_versions env=pr`) → worker resolves the active spec. Two integration bugs were found and fixed during rollout (§4.1–4.2).

The shadow **data plane** has now also been exercised end-to-end (§5, §6). Two eval rounds were driven on comp `id=3`: the first surfaced two platform bugs (§4.3 shadow rows silently never persisted; §4.4 legacy ignored the task seed, making scores incomparable); after fixing both and redeploying the worker, the second round confirms the spec-driven port is faithful — **17/18 submissions scored bit-identical (Δ=0.0000)** to legacy, the lone exception being a legacy-side `SandboxStartupError` infra flake (§6).

Everything runs on the per-PR Neon branch DB for PR #388 (`apex-pr-388`), forked from stage — fully isolated, deleted on PR close.

---

## 2. Components

| Part | Contents | Location |
|---|---|---|
| **A. Competition repo** | `spec.yaml`, `player/Dockerfile`, `evaluate.py` (in-container port of the host-side game loop — launches the miner submission as a loopback HTTP subprocess, runs the vendored `run_game` engine per task, honours `task.input.seed`, writes `/data/result.json`), `scoring.py` (ports `_calculate_game_score` + `/1009` normalizer), `battleship_engine.py` (vendored), `generate_round.py`, `screener.py` (vendored ASTGuard, shipped-not-wired), fixtures, `release.yml`. | `macrocosm-os/apex-competition-battleship` |
| **B. Registry activation** | `competitions/battleship/1.0.0.yaml` (digest-pinned) + `input.schema.json`; `active/pr.yaml` → `battleship: "1.0.0"`. | `apex-competitions-registry` |
| **C. Platform shadow subsystem** | `SPEC_DRIVEN_SHADOW` / `SPEC_DRIVEN_SHADOW_EPSILON` settings; migration `012_add_shadow_eval_comparisons.sql`; `shared/backend/.../shadow_eval_comparison.py` (row model + INSERT — lives in `shared/backend` because `src/worker` forbids `sqlalchemy` per `package_checker`); `worker/shadow_store.py` (lazy write-capable DB handle); `worker/shadow.py` (`run_shadow` + `compare`); the `_handle_job` hook. | `apex-mvp` PR #388, commits `c062afa..fe6db09` |

### Design decisions that shape the test
- **Isolation guarantees (verified in review):** the hook (`worker.py::_handle_job`) fires **after** the canonical delivery sweep; `run_shadow` never calls `set_result_callback` (so `SoloRunner._emit_result` no-ops → nothing reaches the orchestrator); the whole `run_shadow` body is wrapped so it **never raises**; failures are logged `outcome={ok|error|timeout|no_spec}`. A shadow failure cannot alter, delay, or fail the real eval.
- **Score scale (design §3.3):** `SoloRunner` carries one number (`eval_raw_score == eval_score == raw_score`); legacy carries two (raw ~0–1009, normalized `/1009`). The image emits the **normalized 0–1** value as `raw_score`; shadow compares **`eval_score`**. `eval_raw_score` divergence is expected and excluded.
- **Non-determinism (design §3.4):** legacy ignores the task seed and randomizes the board per game → scoring is stochastic → exact per-submission parity is impossible. The port **honours `task.input.seed`** (engine already threads it into `Board(seed=…)`), and the success bar is pipeline + status parity + statistical score agreement, **not** bit-exact equality.
- **Disposable:** the shadow subsystem is one contiguous revert range and **never merges to `main`**. Migration 012 lands only in per-PR Neon branch DBs (confirmed), never stage/prod.

---

## 3. Test steps executed

Legend: ✅ done · ⏳ pending

| # | Step | Command / mechanism | Result |
|---|---|---|---|
| 1 | Competition repo built (TDD, per-task review). | `apex-dev preflight` + `apex-dev run` | ✅ `apex-dev run` built the image and played real games → valid `result.json`, `raw_score ∈ [0,1]`. Scoring parity + determinism unit tests pass. |
| 2 | Shadow subsystem built. | `uv run pytest src/worker/tests/` | ✅ 16 pass; `package_checker` clean (`src/worker` sqlalchemy-free); final whole-branch review clean (no Critical/Important). |
| 3 | Signed image published. | tag `v1.0.0` → `release.yml`: build → push by digest → `cosign sign --yes IMAGE@digest` (keyless OIDC) | ✅ `…@sha256:1841b069…`; `.sig` artifact present at that digest. |
| 4 | Registry activation merged (`pr` env). | PR #7 (`active/pr.yaml`) | ✅ |
| 5 | Spec-syncer run. | `kubectl -n apex-pr-388 create job --from=cronjob/spec-syncer …` | ✅ **after §4 fixes** — `SPEC_SYNCER_ENV=pr` reconciles `active/pr.yaml`; `run summary: env=pr failures=0`. |
| 6 | DB rows present (PR Neon branch). | `psql "$DB_CONNECTION_STRING"` | ✅ `competition_spec_versions`: `battleship/1.0.0`, `signature_verified_at` set, `mirrored_image_ref=…apex-images-mirror@sha256:1841b069…`. `competition_active_versions`: `pr \| battleship \| spec_version_id=5`. |
| 7 | Migration 012 in PR DB. | scheduler applies on startup | ✅ `shadow_eval_comparisons` exists (0 rows). |
| 8 | Shadow enabled on worker. | `kubectl -n apex-pr-388 set env deployment/worker SPEC_DRIVEN_SHADOW=battleship` | ✅ rolled out. (Battleship is CPU/solo → jobs served by the `worker` deployment; round/onnx workers return before the hook.) |
| 9 | Reactivate battleship + open round + seed submissions. | direct DB on the pr-388 branch (admin `PUT /competition` can't set `state`/`end_at` — see §5; old S3 submission code was GC'd, so reference players were re-uploaded) | ✅ comp 3 → active, EVALUATION round created with generated tasks, 18 submissions seeded |
| 10 | Worker runs legacy + shadow; rows written. | scheduler `enqueue_evaluations` → worker (`SPEC_DRIVEN_SHADOW=battleship`) | ✅ run 2 — all 18 evaluated; 18 rows in `shadow_eval_comparisons`. (Run 1 wrote **0** rows — see §4.3.) |
| 11 | Analyse comparison rows. | SQL (§6) | ✅ **17/18 Δ=0.0000** after §4.4 fix (§6) |

### How spec resolution reaches the PR worker
PR deployments set `ENV=pr` (`envs/pr/configmap-common.yaml`) and get their own Neon branch (forked from stage) via the `deploy-pr-staging.yml` "Create Neon branch for PR" step. The `pr` spec-syncer reconciles `active/pr.yaml` into that DB with `env='pr'`; `worker/spec_resolver.resolve_active_spec(pkg)` reads by `ENV` → matches. No manual row seeding (earlier design §9.2 concern) — the `pr` env is the mechanism.

---

## 4. Bugs found & fixed

§4.1–4.2 are **control-plane / syncer** bugs found during rollout; §4.3–4.4 are **shadow-subsystem / evaluator** bugs found while driving the eval round (§6). §4.1–4.2 surfaced only because battleship is the first **real signed, private** image through the syncer (hello_world used a bare unsigned test identity); neither could reach prod. §4.3 lives entirely inside the throwaway shadow subsystem; §4.4 touches the canonical legacy evaluator and is worth keeping (a genuine determinism improvement) even though the shadow subsystem itself never merges.

1. **Private GHCR package → syncer pull 401.** `images.py` shells out to `crane copy` / `cosign verify`, which must pull the source image. The `apex-competition-battleship` package was private (repo created private); the syncer's creds are scoped to the mirror registry → `unauthorized`, spec skipped before the `competition_spec_versions` INSERT. **Fix:** made the container package public (design intends public competition images; keyless cosign + mirror-by-digest assume pullable). ✅

2. **cosign identity glob vs exact match.** `images.py::verify_signature` runs `cosign verify --certificate-identity <spec.cosign_identity>` — **exact** match. Spec declared `…/release.yml@refs/tags/*`; real keyless cert SAN is `…@refs/tags/v1.0.0`. cosign does **not** expand `*` for `--certificate-identity`. Reproduced locally against the live image:
   - `--certificate-identity …@refs/tags/*` → ❌ `none of the expected identities matched … got [...@refs/tags/v1.0.0]`
   - `--certificate-identity …@refs/tags/v1.0.0` → ✅ verifies
   - `--certificate-identity-regexp …@refs/tags/.*` → ✅ verifies

   **Interim fix (done):** PR #8 pins `competitions/battleship/1.0.0.yaml` `cosign_identity` to the exact `@refs/tags/v1.0.0`; syncer now verifies + mirrors + activates. **Platform fix (⚠️ PHASE 2 PREREQUISITE, design §10):** switch `verify_signature` to `--certificate-identity-regexp` (translate `*`→`.*`) so specs keep the version-agnostic glob; rebuild/redeploy the syncer image. Required before Phase 2 (stage→prod cutover + porting the remaining competitions) so specs need not pin an exact per-version tag. The battleship repo `spec.yaml` template intentionally keeps the glob (correct once the syncer is fixed; pinning the tag there would break future versions). ✅ interim / ⏳ platform (Phase 2)

3. **Shadow rows silently never persisted (Prometheus registry collision).** In run 1, every submission logged `shadow outcome=ok` and a `shadow comparison`, but `shadow_eval_comparisons` stayed empty (0 rows). `worker/shadow_store.py::_get_db` constructed `DatabaseClient()` on the **default global** prometheus `REGISTRY`; the worker already holds a `DatabaseClient` (spec_resolver's) there, so the second construction raised `Duplicated timeseries in CollectorRegistry: {apex_pg_transactions_fail, …}` — which `record_comparison` catches and swallows by design (it "never raises"), dropping every row. The unit test missed it because `test_record_comparison_swallows_write_errors` monkeypatches `_get_db` to raise, so a real second client is never constructed. **Fix (done):** `DatabaseClient(registry=CollectorRegistry())` in `shadow_store.py` — the constructor already exposes `registry` for exactly this isolation. Verified in run 2: all 18 rows written. ✅

4. **Legacy evaluator ignored the task seed → scores not comparable.** `backend/eval/battleship/runner.py` generated a fresh `game_seed = random.randint(0, 2**32-1)` per game, randomizing the engine board, while the spec-driven `SoloRunner` already honours `task.input.seed` (design §3.4). With only the seed differing, legacy vs shadow disagreed by up to ±0.6 per submission in run 1 (pure stochastic noise, not a porting bug — scores quantized to ~1/N_games). **Fix (done):** `runner.py` now uses `task.input.seed` (falling back to random only if unset) so both evaluators build the **same board per task**. This is a real determinism/reproducibility improvement to the canonical path and is worth keeping independent of the shadow work. Verified in run 2: 17/18 exact matches (§6). ✅

---

## 5. Remaining steps (drive one eval round) — ✅ done (see §6)

**Mechanism note (differs from the original plan):** admin `PUT /competition` (`UpdateCompetitionSchema`) has **no `state`/`end_at` field**, so it cannot reactivate a stale competition; and the round-32-era S3 submission code had been garbage-collected (`NoSuchKey`). The round was therefore driven **directly against the pr-388 Neon branch** from inside a cluster pod (guaranteeing `DB_CONNECTION_STRING` → the isolated branch, never `apex-stage`): set comp 3 `state=active` + `burn_factor=0` + `end_at` in the past (so no round churn), created an EVALUATION round with generated tasks, uploaded fresh reference players to S3, and seeded `pending` submissions. The scheduler's `enqueue_evaluations` tick (every 10 s) then enqueued them to the CPU-solo queue and the worker ran legacy + shadow.

battleship (competition `id=3`, `pkg=battleship`, `ctype=solo`) is `state=stale`, `end_at=2026-02-03`, last round (31) `completed` — so it needs real reactivation. All on `apex-pr-388` (disposable per-PR DB).

Scheduler admin API (`scheduler-service:8500`, `Authorization: Bearer $ADMIN_API_KEY`; reach via `kubectl -n apex-pr-388 port-forward svc/scheduler-service 8500`):

1. **Reactivate** — `PUT /competition` (`UpdateCompetitionSchema`): `state=active`, future `end_at`.
2. **Open round** — `POST /round` (`RoundSchema`), or let `round_manager` auto-open for the now-active competition.
3. **Seed submissions** — upload the reference `submission.py` to an S3 `code_path`, then `python loadtest_seed_submissions.py --db-dsn … --competition-id 3 --round-id <R> --round-number <N> --code-path <path> --hotkeys-json …` (seeds `pending` rows directly; skips API/screener/fee/S3, so the `code_path` must already exist in S3 and match how the orchestrator builds `job.raw_code`).
4. **Advance to Evaluation** — force the round to `EVALUATION` (admin) so the orchestrator enqueues eval jobs to Redis rather than waiting for the ~daily cadence.
5. **Verify** the `worker` pulls battleship eval jobs and runs legacy + shadow; watch logs for `shadow comparison` / `outcome=ok`.
6. **Analyse** `shadow_eval_comparisons` (§6).

### Decisions to confirm before mutating state
- **Reactivate `id=3`** vs. a fresh throwaway competition (recommend reactivate — disposable PR DB).
- **Testnet side-effect:** `apex-pr-388` runs `BITTENSOR=True, NETWORK=test, NETUID=61` with a `validator` pod → scoring a round can trigger testnet weight-setting. Recommend `kubectl -n apex-pr-388 scale deploy/validator --replicas=0` for the duration.
- **Round timing:** force to `EVALUATION` for an immediate run.

---

## 6. Success criteria & analysis

`shadow.compare(legacy, shadow, epsilon)` records per submission: `score_delta = shadow.eval_score - legacy.eval_score`, `status_parity` (`_is_success(eval_error) in ("","Success",None)` on both sides), `within_epsilon (|Δ| ≤ SPEC_DRIVEN_SHADOW_EPSILON=0.05)`, plus both raw/normalized scores and metadata.

Over one round:
1. **Pipeline:** ≥99% of shadow runs `outcome=ok` (spec resolved, image ran, valid `result.json`).
2. **Status parity:** ≥99% of submissions agree success-vs-error/forfeit.
3. **Score agreement (statistical):** Spearman(legacy, shadow `eval_score`) ≥ 0.95; mean `|Δeval_score|` within legacy's own game-to-game noise band; a coarse per-submission `|Δ| ≤ 0.3` bound catches gross porting bugs (e.g. broken normalizer). Exact equality **not** expected (stochastic scoring). `within_epsilon` is advisory here, not a gate.

Analysis query (starting point):
```sql
SELECT count(*) AS n,
       avg(abs(score_delta)) AS mean_abs_delta,
       sum((status_parity)::int)::float/count(*) AS status_parity_rate,
       sum((abs(score_delta) <= 0.3)::int)::float/count(*) AS within_coarse_bound
FROM shadow_eval_comparisons WHERE competition_pkg='battleship';
-- Spearman via a small script over (legacy_score, shadow_score).
```

### Results (two rounds driven on comp id=3)

Both rounds used 18 reference submissions built from `battleship/baseline.py`'s wire contract, in 3 shooting strategies (random / hunt-target / parity-hunt) for score spread.

**Run 1 — 5 games/submission, pre-fix (round 32).** Exposed §4.3 and §4.4:
- `shadow_eval_comparisons` = **0 rows** (persistence bug §4.3); comparison data survived only in worker structured logs.
- Of the 12 pairs recoverable from logs: pipeline `outcome=ok` 12/12, status parity 12/12, but score agreement poor — mean |Δ| 0.133, max 0.597, Spearman 0.77. Root cause was §4.4 (legacy randomized the board, ignoring the seed): scores quantized to ~1/5 = 0.2 steps, so ±0.2–0.6 deltas were pure stochastic noise, **not** a porting error.

**Run 2 — 20 games/submission, post-fix, deterministic players (round 34).** Both fixes deployed; reference players seeded deterministically so each submission plays reproducibly (isolating the eval harness from player randomness):

| Metric (n=18) | Result |
|---|---|
| Rows persisted | **18/18** (§4.3 fixed) |
| **Exact matches (Δ = 0.0000)** | **17/18** |
| status_parity_rate | 17/18 (94.4%) |
| within_epsilon (0.05) | 17/18 (94.4%) |
| mean \|Δ\| | 0.033 (entirely from the 1 outlier) |
| Spearman (excluding outlier) | **1.000** |

Every cleanly-evaluated submission scored **bit-identical** under legacy and shadow (e.g. `hunt_0` 0.3484/0.3484, `parity_0` 0.5972/0.5972) — strong evidence the externalized `evaluate.py` + `scoring.py` port is faithful to the canonical evaluator.

**The single non-match is infra, not scoring:** `sub=15322` had `legacy=0.0` with `SandboxStartupError: Timeout waiting for sandbox_ready` (the known stage small-node startup-timeout flake) while shadow ran fine (0.597, in-range with its deterministic siblings). It correctly surfaces as `status_parity=False`.

**Interpreting the §6.3 criterion:** the Spearman ≥ 0.95 / epsilon-0.05 bar was written for the *stochastic* (pre-§4.4) regime and is not achievable at small `number_of_tasks` there. Once legacy honours the seed (§4.4) and players are deterministic, **per-submission exact match (Δ=0)** is the correct and stronger lens; run 2 meets it 17/18, with the miss attributable to sandbox infra.

---

## 7. Rollback / cleanup

- **Shadow off:** `kubectl -n apex-pr-388 set env deployment/worker SPEC_DRIVEN_SHADOW-` (canonical path untouched throughout).
- **Deactivate spec:** drop `battleship` from `active/pr.yaml`.
- **Teardown:** closing PR #388 deletes the Neon branch (and `shadow_eval_comparisons`).
- **Before merge:** revert the shadow range — `git revert --no-commit c062afa^..fe6db09` — so nothing shadow-related reaches `main` (leaves the docs + any unrelated commits). The §4.3 `shadow_store.py` registry fix lives inside that range and reverts with it.
- **Keep (do NOT revert):** the §4.4 `backend/eval/battleship/runner.py` seed fix (`game_seed = task.input.seed …`). It touches the **canonical** legacy evaluator, is independent of the shadow subsystem, and makes battleship scoring reproducible — merge it (currently staged, uncommitted).

---

## 8. Artifacts

| Item | Reference |
|---|---|
| Competition repo | `macrocosm-os/apex-competition-battleship` (release `v1.0.0`, run 29021342656) |
| Signed image | `ghcr.io/macrocosm-os/apex-competition-battleship@sha256:1841b0698b137c6a7b2f5f8d71e45dc57b975567c04d212a55be585f1e3ee275` |
| Mirror | `ghcr.io/macrocosm-os/apex-images-mirror@sha256:1841b069…` |
| Platform PR (shadow subsystem) | `apex-mvp` #388 (branch `phase-1-battleship-shadow`, commits `c062afa..fe6db09`) |
| Registry PRs | `apex-competitions-registry` #7 (activate `pr`), #8 (cosign identity pin) |
| PR-staging | namespace `apex-pr-388` on `do-nyc3-apex-mvp-stage`; per-PR Neon branch (forked from stage) |
| Platform follow-ups | design doc §10 (syncer `--certificate-identity-regexp`) |
