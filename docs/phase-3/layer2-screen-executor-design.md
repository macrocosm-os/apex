# Design: Layer-2 `screen` executor (behavioural screening in a sandbox)

**Status:** Proposal
**Owner:** Platform
**Context:** Follow-up to the screening redesign in
[`competition-externalization-design.md` §Screening](../competition-externalization-design.md).
Layer-1 (generic, spec-configured `GenericScreener`) is implemented. This doc designs the
**Layer-2** executor for the optional `entrypoints.screen` entrypoint (SDK v0.3.0).

## Problem

Layer-1 screening is generic, in-process, and fast (AST guard for code / weights validator for
models — `_passes_competition_screener`). But a few competitions need **behavioural / bespoke**
checks that run competition-owned code:

- **Tron** — `sandbox_screener.py` (anti-cosmetic-RL: parameter-count, output-variation,
  gradient-flow, ablation). Today it's **baked into the player image** and run at `is_ready`, so
  the player image must be private and the check ships with every eval.
- **aurelius_steering** — safetensors validation + **concept match against the round's active
  concept** (so the verdict is *per-round*, not just per-submission).

The spec now declares Layer-2 via `entrypoints.screen` (its own digest-pinned image, so it can
be private while the player image is public). What's missing is the **platform executor** that
runs that image and turns exit code into a `screening_status`.

## The model — a sandboxed screen job

Layer-2 runs as a **worker sandbox job**, mirroring how eval/round-gen/onnx jobs already flow —
NOT in the orchestrator process (screening must not run competition code in-process, and a
sandbox is the real trust boundary).

```
submission arrives
   └─ Layer-1 (in-process GenericScreener)  ── fail ─▶ reject submission (as today)
        └─ pass ─▶ orchestrator enqueues a SCREEN job (only if spec has entrypoints.screen)
                     └─ worker runs the screen sandbox ─▶ exit 0/≠0 ─▶ POST verdict
                          └─ orchestrator sets submission.screening_status = passed|failed
                               └─ failed ─▶ submission rejected (never enqueued for eval)
                                  passed ─▶ eligible for eval; verdict reused via
                                            JobResponse.screening_status (existing path)
```

### Screen sandbox contract

A new `JobType.SCREEN`. The worker runs `entrypoints.screen.image` (mirrored, digest-pinned)
with:

- submission artifact mounted at `submission.target_path` (file mount, same as the player).
- round input provided the same way as `evaluate` (env `CONFIG_JSON` / `/data/input.json`) — so
  per-round checks like aurelius's concept-match work.
- `command = entrypoints.screen.command`; `run_timeout_in_seconds = entrypoints.screen.timeout_s`.
- **`network_disabled: true`, `allow_internet: false`** — a screener needs no network (unlike a
  player/referee), so it's the most locked-down sandbox.
- **Verdict = exit code**: `0` = pass, non-zero = fail. Optional `/data/screen_result.json`
  `{"passed": bool, "reason": str}` for a structured reason (surfaced to the miner); absent →
  fall back to exit code.

A new generic `ScreenRunner` in the worker (sibling of `SoloRunner`) builds the sandbox from the
spec, runs it, maps exit-code/`screen_result.json` → verdict. No competition code in-process.

### Verdict caching & re-screening

- `submission.screening_status` (`passed`/`failed`/`None`) already exists and is threaded to the
  eval job (the reuse path is unchanged).
- **Per-round validity.** Most Layer-2 checks are submission-intrinsic (Tron) → cache once. But
  aurelius's concept-match depends on the **round's active concept**, so its verdict must be
  **invalidated / re-run when the round (concept) changes**. The screen job keys its cached
  verdict on `(submission_id, round_id)` (or the concept hash from the round input); a submission
  carried into a new round re-screens. Layer-1 has no such dependency.

## Failure semantics

- Screen image crash / timeout / no verdict → **fail closed** (reject the submission), attributed
  to the screener image, logged. (A screener that can't run must not admit an unscreened model.)
- `entrypoints.screen` absent → Layer-2 is a no-op; Layer-1 alone gates (most competitions).

## Platform pieces

1. **SDK** — `entrypoints.screen` (image + command + timeout_s). ✅ done (v0.3.0).
2. **spec-syncer** — mirror + cosign-verify the screen image (like the referee image) and store
   its mirrored ref. New DB columns `screen_image_ref/digest/mirrored_screen_image_ref`
   (mirrors the referee columns), or reuse a generic "aux image" slot.
3. **worker** — `JobType.SCREEN` + `ScreenRunner` (build screen sandbox from spec, run, read
   exit code / `screen_result.json`).
4. **orchestrator** — after Layer-1 passes, enqueue a SCREEN job when the spec has
   `entrypoints.screen`; on the verdict, `set_screening_status` and reject on fail. Key the
   cache on `(submission_id, round_id)` so concept-dependent checks re-run per round.
5. **competitions** — Tron's `sandbox_screener.py` moves into a `tron-screener` image referenced
   by `entrypoints.screen`; removed from the player image → **Tron player image goes public**,
   private-package scope shrinks to the screener image. aurelius ships a screener image for its
   concept-match.

## Migration

Additive and per-competition, behind the presence of `entrypoints.screen`:

1. Land the syncer + `ScreenRunner` + orchestrator enqueue (no-op while no spec declares `screen`).
2. Per competition that needs Layer-2: publish its screener image, add `entrypoints.screen` to
   its spec, shadow-compare Layer-2 verdicts (screen-job verdict vs the legacy baked/`screener_cls`
   verdict) before cutover — same verdict-parity gate as eval shadow.
3. Tron: move `sandbox_screener.py` out of the player image; make the player image public.

## Open questions

- **Timing.** Enqueue the SCREEN job at *submission* (fail fast, but a submission may predate a
  round) or at *round-eval build* (aligns with the active concept)? Recommendation: at submission
  for submission-intrinsic checks; concept-dependent checks (aurelius) re-screen at round build.
  A `screen.per_round: bool` spec hint could make this explicit.
- **One screen job per submission vs. batched** for duel pairs — reuse the existing per-submission
  screening flow; duels screen each submission independently (as today).
- **DB**: dedicated `screen_image_*` columns vs. a generic aux-image table if more aux images
  appear later (convert_model is command-only today, but could grow an image).
