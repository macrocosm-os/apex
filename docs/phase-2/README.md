# Competition Externalization — Phase 2: Battleship prod cutover

**Status:** Ready for rollout (apex-mvp code changes landed; activation + flip are operator steps)
**Owner:** Platform
**Related:** [Phase 1 test plan](../competition-externalization-phase-1-test-plan.md) · [Phase 1 design](../superpowers/specs/2026-07-09-competition-externalization-phase-1-design.md)

---

## 1. What Phase 2 is

Cut the **battleship** competition over from the legacy in-process `BattleshipEvaluation`
(`EVAL_REGISTRY`) to the spec-driven `SoloRunner` **in production**, then delete the legacy
in-process evaluator. Phase 1 already proved score parity in shadow mode on PR-staging
(17/18 submissions bit-identical; the one miss was a legacy-side sandbox flake). Per the
approved plan we go **straight to cutover** in prod rather than re-running shadow there.

Cutover is gated by the existing global worker flag **`SPEC_DRIVEN_ENABLED`** plus the presence
of an **active battleship spec in the prod registry**. No new per-competition gate was added
(see §6 for the fragility this implies).

### Scope of the two commits

Phase 2 lands as **two commits**:

1. **Cutover commit** (merge before flipping the flag) — everything in this PR:
   - `SoloRunner` now uploads per-game **history** files (closes the HISTORY-file gap, §4).
   - `calculate_score_to_beat` is **spec-aware** (identity normalization + no `KeyError` when a
     pkg has no `EVAL_REGISTRY` entry) — §5.
   - `round_manager` defer-winner check tolerates a missing registry entry — §5.
   - These are all **backward compatible** while battleship is still in `EVAL_REGISTRY`, so
     merging them changes nothing until the flag is flipped.
2. **Deletion commit** (apply only after one clean prod round) — `battleship-legacy-deletion.patch`
   in this directory. Removes the `battleship` `EVAL_REGISTRY` entry and the battleship-only
   `runner.py` / `screener.py`. **Kept intentionally** (see §7): `battleship/models.py`,
   `battleship/generator.py`, `battleship/normalizer.py`, and all of
   `shared/competition/.../battleship/` — `rl_battleship` still imports them and is not ported
   until Phase 4.

---

## 2. Companion change in the competition repo (REQUIRED to close the HISTORY gap)

`apex-competition-battleship`'s `evaluate.py` must write its per-game history artifacts (the
spec-driven analogue of the legacy `game_<id>.json` uploads) into a **`history/` subdirectory
of `/data`**:

```
/data/history/game_<id>.json      # one file per game
```

The platform (`SoloRunner._collect_history_files`) copies whatever lands in `<mount>/history/`
out to the worker's shared-volume root (so it survives the sandbox-mount teardown) and delivers
it to the orchestrator as `FileType.HISTORY`, exactly like the legacy runner did. If the image
writes nothing there, no history is uploaded — same as today, no error. **Ship this image change
(new digest, re-signed, re-activated in the registry) before or with the cutover** or battleship
loses its `game_*.json` history uploads at cutover.

---

## 3. Cutover sequence (operator steps — STOP before any repo push and confirm with the owner)

> **Round-boundary rule (critical for the global-flag approach).** The worker picks legacy vs
> spec-driven **at eval time**, by `SPEC_DRIVEN_ENABLED` + active spec — *not* by when the round
> started. Flipping the flag while a battleship round is mid-evaluation would score some of that
> round's submissions on the legacy ~0–1009 scale and some on the spec 0–1 scale (a mixed-scale
> round). **Only flip the flag while battleship has no in-flight EVALUATION round** — i.e.
> between rounds. Battleship rounds are 1 day, so pick the gap.

1. **Publish the image** (competition repo): build → push by digest → cosign-sign the
   `evaluate.py` (+`history/`) image. Capture the digest.
2. **Activate in prod registry** (`apex-competitions-registry`): PR adding
   `active/prod.yaml → battleship: "<version>"` (spec version pinned to the new digest).
   *This is a push — stop and confirm before opening it.*
3. **Sync**: the prod spec-syncer mirrors + cosign-verifies + upserts
   `competition_spec_versions` / `competition_active_versions (env=prod)`. Trigger a one-shot
   `kubectl -n apex-prod create job --from=cronjob/spec-syncer …` for low latency; confirm
   `competition_active_versions` has the `prod | battleship` row.
4. **Confirm the live competition row** (`competition` table, prod): `baseline_raw_score` should
   be `0.0` (scale-agnostic; see §5). If some non-zero legacy-scale baseline was ever set,
   reset it to `0.0`.
5. **Flip the flag** (between rounds, per the rule above): set `SPEC_DRIVEN_ENABLED=true` on the
   prod `worker` deployment (`apex-config-worker` ConfigMap / `kubectl set env`). Round-gen stays
   legacy (`SPEC_DRIVEN_ROUND_GEN` does **not** include battleship — the entrypoint ships, the
   flag stays off).
6. **Verify one full round** (§8).
7. **Delete legacy** (after one clean round): apply `battleship-legacy-deletion.patch` as its own
   commit/PR (§7).

---

## 4. HISTORY-file gap — how it was closed

Legacy `BattleshipEvaluation` wrote `game_<id>.json` to the worker's shared root and returned
them as `FileType.HISTORY`. `SoloRunner` previously returned only `FileType.LOG`. The fix
(`src/worker/src/worker/spec_runners.py`):

- The image writes history to `/data/history/` (§2).
- `SoloRunner._collect_history_files` copies each file out of the soon-wiped sandbox mount
  (`clean_up(delete_files=True)` clears `mount_dir`) to `shared_volume_path_on_worker` under a
  submission-scoped name (`sub<id>_<name>`), and attaches them as `FileType.HISTORY`.
- Best-effort: a missing `history/` dir or a per-file copy error yields fewer/no history files
  rather than failing the eval — history is observability, not the score.

Covered by `test_collect_history_files_*` in `src/worker/tests/test_spec_dispatch.py`.

---

## 5. Raw-score re-baselining — audit findings

At cutover, battleship's stored **`eval_raw_score` changes scale**: legacy reported the raw
game score (`base_win_points + speed_bonus`, ~0–1009) and normalized `eval_score = raw/1009`;
the spec-driven `SoloRunner` reports the already-normalized value as **both** `eval_raw_score`
and `eval_score` (0–1, identity — "option A", the platform does not re-normalize).

| Consumer | Uses | Effect at cutover | Action |
|---|---|---|---|
| **Ranking / weights** | `eval_score` / `top_score` (0–1, unchanged) | none | none |
| **Winner election** (`determine_solo_winner`) | raw-vs-raw (`eval_raw_score` vs `score_to_beat_raw`) | safe **within a round** (all submissions same scale, given the round-boundary flip + the version-0 auto-submission re-seeding the round's top). See the round-boundary rule in §3. | flip between rounds |
| **`baseline_raw_score`** (gating fallback + display) | raw | `0.0` is scale-agnostic (`max(0, raw)` / `min(0, raw)` behave the same) | confirm live row is `0.0` |
| **`calculate_score_to_beat`** (display: `/miners`, competition-details) | raw → normalizer | **while battleship is still in `EVAL_REGISTRY`** (i.e. before the deletion commit), it keeps applying `BattleshipNormalizer` (÷1009) to the now-0–1 raw, so the displayed *score-to-beat* is ~1000× too small for the verification round. **Self-heals** once the deletion commit lands (registry entry gone → identity). | expected transient; land the deletion promptly, or accept one round of wrong score-to-beat display |
| **Historical raw-score panels** | stored `eval_raw_score` across rounds | pre-cutover rows are ~0–1009, post-cutover rows are 0–1 (mixed scales in historical views) | cosmetic; no fix |

**Net:** nothing rank- or weight-visible changes (that was the whole point of the image emitting
the normalized number as `raw_score`). The only real transient is the score-to-beat **display**
value during the verification round, which the deletion commit removes.

The cutover commit already makes `calculate_score_to_beat` correct for the *post-registry* world
(identity normalization when a pkg has no `EVAL_REGISTRY` entry) and adds a `lower_is_better`
override param for future lower-is-better spec competitions (iota) that don't have a registry
normalizer to consult.

---

## 6. Fragility of the global `SPEC_DRIVEN_ENABLED` gate

We reuse the existing **global** flag rather than a per-competition allowlist. Consequences:

- With `SPEC_DRIVEN_ENABLED=true` in prod, **every** competition that has an active prod spec
  cuts over. In Phase 2 only **battleship** is activated in `active/prod.yaml`, so it is the only
  one affected — everything else has no active prod spec and falls back to `EVAL_REGISTRY`.
- **Do not** activate any other competition's spec in `active/prod.yaml` until that competition is
  itself ready to cut over. Activation == cutover once the flag is on.
- **Recommendation for Phase 3/4:** reintroduce a per-competition eval allowlist (mirroring the
  existing `SPEC_DRIVEN_ROUND_GEN` set) before more than one competition is spec-driven in prod,
  so competitions can be activated (and shadow-validated) without immediately cutting over.

---

## 7. Legacy deletion (`battleship-legacy-deletion.patch`)

Apply **after** one clean prod round on the spec path:

```bash
git apply docs/phase-2/battleship-legacy-deletion.patch
# review, then commit as its own PR
```

It removes:
- the `"battleship"` entry from `EVAL_REGISTRY` (+ the `BattleshipEvaluation` / `BattleshipScreener`
  imports),
- `shared/backend/src/backend/eval/battleship/runner.py`,
- `shared/backend/src/backend/eval/battleship/screener.py`.

It **deliberately keeps** (contrary to the parent design's "delete the whole dir" — that is not
safe yet):
- `shared/backend/src/backend/eval/battleship/models.py`,
- `shared/backend/src/backend/eval/battleship/generator.py`,
- `shared/backend/src/backend/eval/battleship/normalizer.py`,
- all of `shared/competition/src/competition/battleship/`.

Because **`rl_battleship`** (`shared/backend/.../eval/rl_battleship/runner.py`, not ported until
Phase 4) reuses `BattleshipEvalInputDataSchema`, `BattleshipInputDataGenerator`,
`BattleshipNormalizer`, and `competition.battleship.battleship` (`Name`, `run_game`, `GameResult`,
`RemotePlayer`); and `scheduler/db/mock.py` imports `battleship.models`. Deleting those modules
now breaks rl_battleship at import time. They come out when rl_battleship is ported.

Verified: applying the patch leaves `registry`, `rl_battleship.runner`, and `scheduler.db.mock`
importing cleanly with no `battleship` registry entry; the patch reverts cleanly.

---

## 8. Verification checklist (cutover round)

- [ ] `competition_active_versions` has `prod | battleship | <spec_version_id>`; syncer run
      `failures=0`; image mirrored + `signature_verified_at` set.
- [ ] Worker logs `Spec-driven eval: battleship v<version> via SoloRunner` for battleship jobs.
- [ ] `submission.eval_raw_score` for new battleship submissions is in `[0, 1]`;
      `eval_raw_score == eval_score`.
- [ ] Ranking (`top_score` / `eval_score` order) matches expectations; no weight discontinuity.
- [ ] `FileType.HISTORY` `game_*.json` files land in S3 for battleship submissions (confirms §2
      image change + §4 platform change).
- [ ] No `SandboxStartupError` spike / no runner exceptions.
- [ ] Round completes and elects a winner normally (the version-0 auto-submission seeds the
      round's top on the 0–1 scale).

---

## 9. Rollback

| Step | Rollback |
|---|---|
| Flag flip | `SPEC_DRIVEN_ENABLED=false` (or unset) on the prod worker → next battleship job runs legacy `BattleshipEvaluation`. Immediate, canonical path untouched. |
| Spec activation | `git revert` `active/prod.yaml` in the registry → no active prod spec → worker falls back to legacy even with the flag on. |
| Legacy deletion | `git revert` the deletion commit (restores registry entry + `runner.py`/`screener.py`). Keep the flag off / spec deactivated while reverting. |

The legacy path stays fully intact until the deletion commit lands, so cutover is reversible with
a single flag flip for the entire verification window.
