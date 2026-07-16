# Competition Externalization — Phase 3 + 4: Tron duel

**Status:** apex-mvp platform code landed on `phase-3-4-externalization`; companion repo,
registry activation, shadow validation, and prod cutover are the remaining (mostly operator)
steps.
**Owner:** Platform
**Related:** [Phase 1 design](../superpowers/specs/2026-07-09-competition-externalization-phase-1-design.md)
· [Phase 2 runbook](../phase-2/README.md)

---

## 1. What Phase 3/4 is

Externalize the **Tron** duel competition: move it off the legacy in-process `TronEvaluation`
(host-side game engine + `run_duel_game` driving two player sandboxes) onto the spec-driven
**`DuelRunnerGymV1`** — N player sandboxes plus a competition-owned **referee** image that holds
the game logic, driven over the gym_v1 HTTP protocol. Tron is the **first and only** duel
competition, so this phase de-risks the entire duel architecture.

- **Phase 3** — implement the platform runtime, publish the `apex-competition-tron` images,
  activate on **stage**, and **shadow-validate** the spec path against the legacy runner.
- **Phase 4** — **prod cutover** (per-competition, round-boundary), verify one full bracket,
  then delete the legacy Tron eval runner + engine.

The single-elimination **bracket** (seeding, pairing, advancement, recovery) and the
duel-result scoring in `round_manager` / `scoring_utils` are **unchanged** — they consume the
per-duel `EvaluationResults`, and `DuelRunnerGymV1` reproduces that exact contract.

Unlike battleship (Phase 2), Tron has **no score-scale change** at cutover: `TronNormalizer`
is identity, and the round score (`eval_raw_score` = bracket rounds survived) is recomputed
scheduler-side after each duel, so nothing rank- or weight-visible changes.

---

## 2. Architecture: gym_v1 referee + player sandboxes

```
                       per-job pod network (K8s)
   ┌───────────────────────────────────────────────────────────┐
   │  referee sandbox (ephemeral, one per GAME)                  │
   │   env: MATCH_ID / SEED / CONFIG_JSON / PLAYER_URLS / N      │
   │   POST /reset, POST /act  ─────────┐        ┌───────────────┤
   │   writes /data/result.json         ▼        ▼               │
   └────────────────────────┬───────────────────────────────────┘
                            │ (sandbox-peer-egress NetworkPolicy)
              ┌─────────────┘             └─────────────┐
              ▼                                         ▼
      player sandbox A (long-lived)            player sandbox B (long-lived)
      gym_v1 server on http_api.port           gym_v1 server on http_api.port
      miner model.pt at target_path            miner model.pt at target_path
```

Per duel job (two submissions):

1. Launch one **long-lived player sandbox** per submission from the mirrored player image
   (`exit_after_startup=True`, kubelet readiness probe on `http_api.readiness_path`, artifact
   written to `submission.target_path`). Obtain each player's pod-IP URL via `get_connection_url`.
2. For each of `duel.num_games_default` games: reorder `PLAYER_URLS` when `swap_sides` (odd
   games), launch an **ephemeral referee sandbox** from the mirrored referee image with the
   gym_v1 env contract, wait for it to write `result.json`, map the per-game `raw_scores` /
   `player_stats` back to canonical submission order.
3. Aggregate into one `EvaluationResults` per submission (see §4).

**Concurrency:** a single duel holds up to **3** sandbox slots at once (2 players + 1 referee),
plus one referee churned per game. Size `MAX_CONCURRENT_SANDBOXES` accordingly.

**K8s-only.** The Docker backend gives each sandbox its own isolated network, so a referee
sandbox cannot reach the players. Local development uses the SDK `apex-dev` harness instead.
`SANDBOX_BACKEND=k8s` on stage/prod, so this is not a production constraint.

---

## 3. apex-mvp platform changes (this branch — permanent, merges to main)

| Area | Change | File |
|---|---|---|
| **Sandbox env injection** | `SandboxRunRules.env: dict[str,str]` injected into the K8s runner container. Carries the referee's gym_v1 env contract. | `shared/common/.../models/sandbox.py`, `src/worker/.../k8s_sandbox.py` |
| **Referee egress** | `SandboxRunRules.sandbox_peer_egress` labels the referee pod; new `sandbox-peer-egress` NetworkPolicy permits egress **only** to in-namespace `app: sandbox` pods (the players). Additive with `sandbox-block-internet` (egress `[]`) → referee reaches players, internet stays blocked; players keep egress-denied and only reply on established connections. | `src/worker/.../k8s_sandbox.py` |
| **DuelRunnerGymV1** | Full match loop: player launch, per-game referee launch, `swap_sides` reorder, PLAYER_URLS→canonical mapping, per-game aggregation, exact legacy match-outcome tuple, forfeit + referee-crash handling. | `src/worker/.../spec_runners.py` |
| **Per-competition cutover flag** | `SPEC_DRIVEN_EVAL` (set-of-pkgs, like `SPEC_DRIVEN_ROUND_GEN`). A pkg uses the spec eval path iff `SPEC_DRIVEN_ENABLED` OR pkg ∈ `SPEC_DRIVEN_EVAL`. Lets Tron cut over in prod without the global flag cutting over every activated spec (Phase 2 §6). | `src/worker/.../settings.py`, `worker.py` |
| **Cosign identity regexp** | `spec-syncer` verifies the signer identity as an anchored regexp (`*`→`.*`), so specs can use the intended `@refs/tags/*` glob instead of an exact per-version pin (Phase 1 §10 follow-up). | `src/spec-syncer/.../images.py` |

Tests: `src/worker/tests/test_spec_dispatch.py` (duel dispatch, swap mapping, aggregation,
exact tiebreak, forfeit, referee-crash); `src/spec-syncer/tests/test_syncer_logic.py`
(identity-regexp translation).

### Referee command convention

The `duel` spec block pins the referee **image** but carries no command (the schema has no
field for it). The referee logic is competition-owned, so the platform launches it by a fixed
convention: **every duel referee image exposes its entrypoint at `/app/referee.py`**
(`DuelRunnerGymV1.REFEREE_COMMAND`). A future `apex.competition.v1` bump could make this a spec
field; until then the tron referee image satisfies the convention.

---

## 4. Per-duel result contract (what the bracket consumes)

`DuelRunnerGymV1` produces one `EvaluationResults` per submission, byte-compatible with the
legacy `TronEvaluation` so bracket advancement is unchanged:

- `eval_raw_score` = **mean per-game score**; `eval_score` = `eval_raw_score` (identity).
- `eval_metadata`:
  - `opponent_submission_id`
  - `match_outcome` ∈ {`won`, `lost`, `tied`} — decided by the **exact legacy tuple**
    `(mean_raw, games_won, kills_caused, -self_deaths)`; equal tuples → `tied` (the scheduler
    then breaks ties by bracket seed, as today).
  - `win_rate`, `games_won`, `kills_caused`, `self_deaths`, `screening_status`
  - `Game N` per-game breakdown (`game_id`, `swap`, `seed`, `terminal_reason`, `steps`, `score`)
- `file_paths[HISTORY]` = referee `trace.jsonl` per game (best-effort, → S3).

### Referee → platform contract (gym_v1 `result.json`)

The referee writes, per game:

```json
{"raw_scores": [1.0, 0.0], "winner": 0, "terminal_reason": "...", "steps": 42,
 "metadata": {"player_stats": [{"won": true, "killed_opponent": true, "self_death": false},
                               {"won": false, "killed_opponent": false, "self_death": true}]}}
```

`raw_scores` and `player_stats` are in **PLAYER_URLS order**; the platform maps them back to
canonical submission order (so `swap_sides` is transparent). The per-game score cascade
(1.00 clean kill … 0.00 self-death) and `player_stats` are computed **inside the referee** —
the platform aggregates generically.

### Failure semantics

- **Screening/startup failure** → that player forfeits (0.0, `lost`); the opponent wins
  (1.0, `won`). Both fail → both `lost`. Mirrors legacy `_build_validation_failure_results`.
- **Referee crash / no `result.json`** → that **game** scores 0 for everyone and is attributed
  to the referee (not the submissions); the duel continues.

### Seeding

Per-game `SEED = base_seed + game_idx`; `base_seed` is read from the round input
(`tasks[0].input.seed`, matching the legacy generator). Because shadow runs the same job input,
seeds align between the legacy and spec paths — Tron is deterministic given seed + models, so
per-game parity is achievable (a tighter gate than battleship's stochastic comparison).

---

## 5. Companion repo — `apex-competition-tron` (private, first-party)

Published to [`macrocosm-os/apex-competition-tron`](https://github.com/macrocosm-os/apex-competition-tron) (private) — that repo is the source of truth (player, referee, and the Layer-2 screen image at `v1.1.0`).

**Package visibility (recommended: public referee / private player).** GHCR package visibility
is independent of repo visibility, and the two images carry different sensitivity:

- **Referee image → publish public.** It contains the game engine + the per-game death-cause
  scoring, all of which is already public (the scoring cascade is in the competition README and
  the engine lives in `shared/competition`). No new leak; the syncer pulls it anonymously.
- **Player image → keep private.** It bundles the Layer-2 behavioural anti-cheat screener
  (`screen.py`) and its thresholds. Publishing that would let miners tune models to evade the
  checks. Grant the spec-syncer a `read:packages` token (via external-secrets) so its
  `crane`/`cosign` can pull the private player image; otherwise it 401s and the spec is skipped.

(The public-competition / external-designer flow keeps *both* images public — acceptable there
because those competitions externalize their screener too. Not the case for first-party Tron.)

```
apex-competition-tron/
├── spec.yaml                 # kind: duel, protocol: gym_v1 (both image digests)
├── input.schema.json         # from TronEvalInputDataSchema
├── player/
│   ├── Dockerfile            # CPU torch; entrypoint = gym_v1 server
│   ├── launch.py             # gym_v1 Player: /health(is_ready), /reset, /act
│   ├── encode.py             # 5-channel state encoding (verbatim from launch_tron_rl)
│   └── screen.py             # Layer-2 behavioral screener, run in is_ready()
├── referee/
│   ├── Dockerfile            # entrypoint at /app/referee.py (platform convention)
│   ├── referee.py            # SDK Referee.play_game: engine + death-cause cascade
│   └── tron_engine.py        # vendored TronGame (from competition/tron/tron.py)
├── generate_round.py         # ships; flag stays off (legacy round-gen until later phase)
├── convert_to_onnx.py        # convert_model entrypoint; ships, wired later
├── fixtures/ + tests/        # determinism + scoring-parity vs legacy
└── .github/workflows/release.yml   # build + push (both images) + cosign sign
```

**Player port:** the legacy `launch_tron_rl.py` HTTP server (`/health`, `/game`, `/move`)
becomes a gym_v1 `Player`: `/health`→`is_ready`, `/game`→`reset`, `/move`→`act`. The 5-channel
`encode_state` stays on the player side unchanged. The legacy Layer-2 `sandbox_screener.py`
behavioral checks move into `is_ready()` (a model that fails validation never reports ready →
the referee forfeits it → maps to the legacy screening-failure result).

**Referee port:** `competition/tron/tron.py`'s `TronGame` engine + `run_duel_game`'s
health/init/move loop + the death-cause cascade become the SDK `Referee.play_game`, emitting
`raw_scores` + `player_stats` per the §4 contract.

**Layer-1 screener** (`TronScreener`, torchscript format/weights) runs **orchestrator-side** and
is unchanged; both paths consume the same `job.screening_status` (structural parity — Phase 1
§3.5). It stays in `EVAL_REGISTRY` until screening is externalized in a later phase.

---

## 6. Registry — `apex-competitions-registry`

Lives in [`macrocosm-os/apex-competitions-registry`](https://github.com/macrocosm-os/apex-competitions-registry) (private) — the GitOps source of truth (holds `competitions/tron/{1.0.0,1.1.0}.yaml`, battleship, and the `active/<env>.yaml` pointers).

- `competitions/tron/1.0.0.yaml` + `input.schema.json` (with both image digests filled from the
  release CI run).
- **Phase 3:** `active/stage.yaml` → `tron: "1.0.0"`.
- **Phase 4:** `active/prod.yaml` → `tron: "1.0.0"` (stop and confirm before opening the PR).

The syncer mirrors + cosign-verifies **both** images (player + referee) and upserts
`competition_spec_versions` (referee columns already exist from Phase 0) + the env active pointer.

---

## 7. Phase 3 shadow validation (revertable commit — never merges to main)

Same discipline as Phase 1: the shadow subsystem lands as **one `git revert`-able commit** on
this branch, runs on PR-staging, and is reverted before merge. Because duels differ from solo,
the comparison is **per-game** (not per-submission):

- After the canonical legacy `TronEvaluation` delivers, run `DuelRunnerGymV1` in shadow (never
  delivers; all failures swallowed + logged).
- Record per game: winner, `terminal_reason`, per-player score; and per duel: `match_outcome` +
  per-player mean score — into `shadow_duel_comparisons`.
- **Gate:** Tron is deterministic given seed + models, so per-game winner/score parity for
  ≥99% of games, and identical `match_outcome` for ≥99% of duels, over one full stage bracket.

*(This commit is authored after the platform code is validated; see the plan for its file list.)*

---

## 8. Rollout (operator steps — STOP before any push/flag flip and confirm)

**Phase 3 (stage + shadow):**
1. Publish `apex-competition-tron` (build → push both images by digest → cosign-sign). Capture
   digests. Set package visibility: **referee public, player private** (§5). Grant the
   spec-syncer a `read:packages` token (external-secrets) so it can pull the **private player**
   image to mirror/verify; the public referee pulls anonymously.
2. Registry PR: `competitions/tron/1.0.0.yaml` + `active/stage.yaml`. Syncer mirrors + verifies +
   activates in the stage DB (confirm the `stage | tron` active row + `signature_verified_at`).
3. Apply the shadow commit; PR-staging deploy; run one full stage bracket; analyze
   `shadow_duel_comparisons` against the §7 gate.
4. Revert the shadow commit. Merge only the permanent deliverables.

**Phase 4 (prod cutover — Tron only):**
5. Activate `tron` in `active/prod.yaml` (**push — confirm first**); syncer mirrors/verifies/
   activates in prod.
6. Set `SPEC_DRIVEN_EVAL=tron` on the prod worker (per-competition; **do not** rely on the global
   `SPEC_DRIVEN_ENABLED`). Migrate any existing global cutover (battleship) to `SPEC_DRIVEN_EVAL`
   too, so future activations don't auto-cut-over.
7. **Round-boundary rule:** flip only between Tron rounds (rounds are 1 day) so no bracket is
   evaluated on mixed paths.
8. Verify one full prod bracket (§9). Then apply the legacy-deletion commit.

---

## 9. Verification checklist (cutover bracket)

- [ ] `competition_active_versions` has `prod | tron | <spec_version_id>`; both images mirrored +
      `signature_verified_at` set; syncer `failures=0`.
- [ ] Worker logs `Spec-driven eval: tron v1.0.0 via DuelRunnerGymV1` for tron jobs.
- [ ] `sandbox-peer-egress` NetworkPolicy exists; referee pods reach player pods; no player pod
      gains internet egress.
- [ ] Per-duel `eval_metadata` carries `match_outcome` / `opponent_submission_id`; bracket
      advances and recovers normally; a winner is elected.
- [ ] ONNX conversion of the bracket winner still fires (legacy `_process_onnx_conversion_job`).
- [ ] `trace.jsonl` HISTORY artifacts land in S3.
- [ ] No `SandboxStartupError` spike (player startup budget = 300s, matching the legacy Tron
      band-aid for stage's small nodes).

---

## 10. Legacy deletion (Phase 4) — constrained; mostly deferred

After cutover the worker never calls `TronEvaluation`, but **the legacy Tron eval code cannot
be deleted at Phase 4** the way battleship's `runner.py` was. `RegistryEntry.runner_cls` (and
`input_schema`) are **required** fields (`base.py`); `generator_cls`/`screener_cls`/
`normalizer_cls` are optional. So while the `tron` `EVAL_REGISTRY` entry must stay — the
**orchestrator still screens** Tron (`screener_cls=TronScreener`) and **round-gen is still
legacy** (`generator_cls`) — the entry keeps `runner_cls=TronEvaluation` alive, and
`runner.py` imports the game engine (`competition.tron.tron`). Concretely:

- **Cannot delete yet:** `eval/tron/runner.py`, `competition/tron/tron.py`, `launch_tron_rl.py`
  — all reachable from the required `runner_cls` on the retained registry entry.
- **Deletable path (choose at Phase 4):** either (a) leave the legacy eval code in-tree,
  unused, until the screening + round-gen externalization phase removes the whole `tron` entry
  (cleanest; mirrors battleship keeping `models/generator/normalizer`), **or** (b) repoint
  `runner_cls` at a tiny stub `BaseEvaluation` subclass that raises if invoked, then delete
  `TronEvaluation` + engine + launcher while keeping `screener/generator/normalizer`. Option
  (a) is recommended — no stub, no risk, and the deletion lands naturally when the entry goes.
- `convert_to_onnx.py` stays until `convert_model` is wired.

Because of this, **there is no ready-to-apply Tron legacy-deletion patch in Phase 4** (unlike
battleship's) — the real deletion is the screening/round-gen externalization phase, which can
drop the entry and everything it pins in one go.

---

## 11. Rollback

| Step | Rollback |
|---|---|
| Flag | remove `tron` from `SPEC_DRIVEN_EVAL` on the prod worker → next tron job runs legacy `TronEvaluation`. Immediate; canonical path untouched. |
| Spec activation | `git revert` `active/prod.yaml` → no active prod spec → legacy even with the flag on. |
| Legacy deletion | `git revert` the deletion commit (restores `runner.py` + engine). Keep the flag off / spec deactivated while reverting. |

The legacy path stays fully intact until the deletion commit, so cutover is reversible with a
single flag change for the whole verification window.

---

## 12. Open items / risks

- **Referee-command convention** (`/app/referee.py`) is a stopgap for the schema having no
  referee command field; document for future duels or add a spec field via an SDK bump.
- **Player startup budget** is hardcoded to 300s in `DuelRunnerGymV1` (matches the legacy Tron
  band-aid). Revisit when stage sandbox nodes are upgraded.
- **Shadow sandbox cost:** shadow doubles sandbox usage per duel (up to ~6 concurrent slots per
  duel); acceptable at stage volume, note during the shadow bracket.
- **`base_seed` extraction** relies on the `tasks[0].input.seed` convention; a competition whose
  input omits it gets `seed=0` (deterministic but not per-round-varied). Fine for Tron.
</content>
</invoke>
