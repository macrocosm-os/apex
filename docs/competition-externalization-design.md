# Competition Externalization Design

Status: Proposal — core runtime implemented; being validated on stage (Jul 2026)
Owners: Platform

> **Reconciled with `main` (Jul 2026).** Since the first draft the repo added a seventh competition (`aurelius_steering`, the first **GPU** competition, with its own GPU round-generation path), a two-layer **screening** model (cached pre-eval verdicts + an in-sandbox screening layer), **round-generation `sandbox_data`** (audit artifacts persisted off the miner-visible input), and **incremental per-submission result delivery** in the worker. These are folded into the contract and phases below.

> **Implementation status (Jul 2026).** The generic runtime is built and exercised end-to-end on a PR-clone of stage: the spec-syncer mirrors signed images by digest and populates `competition_active_versions`; `SoloRunner` (battleship 2.0.0, `custom` protocol) and `DuelRunnerGymV1` (tron 1.0.0, `gym_v1`) both ran green on stage with the player and referee in **separate sandboxes**. The **two-layer screening** is implemented (generic Layer-1 in-process + Layer-2 `SCREEN` job with a fail-closed verdict file) with unit + integration tests; **Tron is the first competition moving its behavioural screener into its own `screen` image** (companion PR open) so its player image can go public. The mirror registry is **`ghcr.io/macrocosm-os/apex-images-mirror`**. The phase plan below is the original rollout; where it says "will", most of Phases 0–4 is now done or in stage validation.

## TL;DR

Today, a competition is partly a thing that runs inside a sandbox image (training code, model server, dockerfile) and partly a thing that runs **inside our `worker`/`scheduler`/`orchestrator` processes** (`BaseEvaluation` subclasses, generators, normalizers, screeners, the `EVAL_REGISTRY` dict). Adding a competition is a code change in `apex-mvp`. Conflicting dependencies between competitions sit in one Python interpreter. Duels are orchestrated by Python imported into our worker (`from competition.tron.tron import run_duel_game`).

This doc proposes splitting competitions out of `apex-mvp` entirely and turning the platform into a generic runtime driven by a declarative, versioned **competition spec** (`apex.competition.v1`). Specs live in a dedicated GitOps registry repo. A k8s `CronJob` syncs them into our DB and mirrors signed images into our registry. The worker dispatches by `kind`/`protocol` declared in the spec, not by importing competition code. A duel is a standardized gym-style HTTP protocol between platform-spawned **player sandboxes** and a competition-owned **referee image** that holds the game logic.

After this change:

- `apex-mvp` contains zero competition source.
- Adding a new competition is a PR to the registry repo, not to `apex-mvp`.
- External competition designers can be onboarded with the same flow we use internally.
- The platform's trust gate is signed image digests + admin-merged active pointers, both reviewable in git.

The rollout is phased so every phase delivers a working system that we can stop at indefinitely. Production behavior does not change until a competition's cutover phase is reached; legacy and spec-driven paths run side by side and shadow-compare scores before cutover.

## Problem

Competitions today are three things glued together by [`shared/backend/src/backend/eval/registry.py`](shared/backend/src/backend/eval/registry.py):

1. **In‑process host code.** A `BaseEvaluation` subclass per competition orchestrates the eval, plus `BaseGenerator` / `BaseNormalizer` / `BaseScreener` subclasses for round generation, scoring normalization, and submission gating. All of this is loaded into our worker, scheduler, and orchestrator Python interpreters. Concretely, the worker dispatches by package name into the registry:

```462:481:src/worker/src/worker/worker.py
            eval = EVAL_REGISTRY.get(job.competition_pkg)
            if not eval:
                raise ValueError(
                    f"Competition package {job.competition_pkg} not found in registry {EVAL_REGISTRY.keys()}"
                )

            # create the evaluation runner instance
            runner_cls: type[BaseEvaluation] = eval.runner_cls
            runner_input: BaseModel = eval.input_schema
            selected_sandbox = self._get_selected_sandbox_class()
            runner: BaseEvaluation = runner_cls(
                sandbox_cls=selected_sandbox,
                competition_id=job.competition_id,
                round_number=job.round_number,
                competition_pkg=job.competition_pkg,
                submission_ids=job.submission_id,
                hotkeys=job.hotkey,
                codes=job.raw_code,
                shared_volume_path_on_worker=SHARED_VOLUME_PATH_ON_WORKER,
            )
```

As of `main` there are **seven** registered competitions (`matrix_compression`, `battleship`, `rl_battleship`, `energy_arbitrage`, `iota_simulator`, `tron`, `aurelius_steering`). `aurelius_steering` is the first **GPU** competition and adds two wrinkles the externalized design must carry: a GPU round-generation path (baseline scoring on a GPU fleet, opted into via generator args) and `RegistryEntry.defer_winner_to_round_completion=True` (the solo winner is elected at round completion, not incrementally).

2. **In‑sandbox code.** `shared/competition/src/competition/<pkg>/` ships the launcher scripts, training code, and `dockerfiles/Dockerfile` that the image is built from by [`build-competition-images.yml`](.github/workflows/) and pushed to `ghcr.io/macrocosm-os/apex-mvp:sb-<env>-<pkg>-<commit>`.

3. **Cross-boundary Python imports.** Some host-side runners load competition code into our process:

```25:25:shared/backend/src/backend/eval/tron/runner.py
from competition.tron.tron import GameConfig, TronGameResult, run_duel_game
```

`run_duel_game` runs **in the worker**, drives the match over HTTP against two player sandboxes, and returns the winner. The platform process owns the game rules.

Consequences:

- **Slow onboarding.** Adding a competition requires a PR to `apex-mvp` (a private repo) touching `shared/competition/`, `shared/backend/src/backend/eval/`, `EVAL_REGISTRY`, dockerfiles, and CI.
- **Dependency conflicts.** Every competition's host-side dependencies live in the same `uv` workspace. A competition that wants a newer `torch` or an exotic library either gets blocked or forces an upgrade across the platform.
- **No path for external designers.** Third parties cannot contribute to `apex-mvp` (private), cannot ship Python that runs in our process (trust boundary), and have no contract to target.
- **Coupled redeploys.** Updating any competition redeploys the worker.

## Goals

- Competitions live in their own repos. `apex-mvp` contains no competition source after migration.
- Adding/updating a competition is a GitOps PR to a registry repo. Activation is admin-merge-gated.
- The platform never imports competition code into its processes.
- External designers can ship competitions via the same path we use internally, with trust enforced by signed images and merged active pointers.
- Every migration phase ships a working production system. Rollback is `git revert` on the registry repo or a feature flag in the worker.
- Production scoring is shadow-verified before any cutover.

## Non-goals

- Replacing the orchestrator → worker job model. The pull-based Redis queue, `JobResponse`/`JobResults` shapes, and worker lifecycle stay as-is.
- Replacing the sandbox runtime. `KubernetesSandbox` keeps doing what it does — the spec just tells it what to run.
- Supporting arbitrary submission artifact types. Per the May 2026 production submissions sample (78,338 rows), 99.99% are `.py` (88.0%) or `.pt` (11.8%). The contract supports `code`, `torchscript`, `onnx`. Outliers (`.zip`, `.rs`) are out of scope.
- Multi-region or multi-cluster spec distribution. The syncer targets one DB per env.
- Defining emission or incentive allocation in the competition spec. `incentive_weight` stays on the platform `competitions` row (today seeded at competition creation; eventually driven by alpha voting — see [Future work](#future-work)).

## Repo topology

Four repos, with clear ownership and visibility:

| Repo                         | Visibility | Owner                                                   | Contains                                                                                                                                 |
| ---------------------------- | ---------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `apex-mvp`                   | private    | Platform                                                | Orchestrator, scheduler, worker, spec syncer, generic runner templates. No competition source.                                           |
| `apex-competition-sdk`       | public     | Platform                                                | `apex.competition.v1` JSON Schema, base images (gym_v1 player base, referee base), local dev harness CLI, designer docs.                 |
| `apex-competition-<name>`    | public     | Competition designers (with Platform CODEOWNER on spec) | One repo per competition (or grouped). Player image, referee image, generator/screener entrypoints, spec YAML, reference fixtures.       |
| `apex-competitions-registry` | private    | Platform                                                | GitOps registry. Holds `competitions/<id>/<version>.yaml` and `active/{stage,prod}.yaml`. Merged PRs are the platform's source of truth. |

Why the registry stays private even though competitions are public: activation is a Macrocosmos-controlled gate. Designers self-serve their own competition folder; activation is platform-only.

## The spec — `apex.competition.v1`

Declarative, versioned, JSON Schema-validated. Stored as YAML in the registry repo.

```yaml
schema: apex.competition.v1
id: tron
version: 1.4.0
display_name: "Tron Duel"
kind: duel                     # solo | duel
process_type: cpu              # cpu | gpu
resources:
  cpu_limit: 2
  mem_limit: 1.5Gi
  gpu_count: 0
image:
  ref: ghcr.io/some-org/tron-player
  digest: sha256:abc123...     # pin by digest; tags are forbidden
submission:
  artifact_type: torchscript   # code | torchscript | onnx
  max_size_mb: 100
  target_path: /app/model.pt
screening:                     # Layer-1: the GENERIC platform screener, configured (no competition code).
  # Keyed by submission.artifact_type — the platform runs the matching validator with these knobs:
  #   code   -> shared ASTGuard:  { max_size_mb, extra_forbidden_modules: [...], extra_forbidden_calls: [...] }
  #   model  -> weights validator:{ max_size_mb, min_weight_bytes, max_code_weight_ratio }
  max_size_mb: 100
  min_weight_bytes: 10240
  max_code_weight_ratio: 10
input_schema:
  $ref: ./input.schema.json    # JSON Schema, validated platform-side before launch
defaults:
  baseline_score: 0.5
  baseline_raw_score: 0.5
  round_length_in_days: 1
  submission_reveal_days: 1
  lower_is_better: false
  defer_winner_to_round_completion: false   # true → winner elected at round completion, not incrementally (e.g. aurelius_steering)
entrypoints:
  evaluate:                    # HTTP server in the PLAYER sandbox (the miner submission).
    command: ["python", "/app/launch_tron_rl.py", "--port", "8000"]
    # For BOTH solo and duel the submission runs as an isolated player server the
    # referee/evaluator reaches over the per-job network — it never shares a sandbox with
    # the scorer. (Legacy 1-sandbox solo, where evaluate.command WAS the scorer, is
    # deprecated; see "Solo evaluation isolation".)
    network_disabled: false    # players must be reachable by the referee/evaluator
    allow_internet: false
    timeout_s: 600
    http_api:
      port: 8000
      readiness_path: /health
      protocol: gym_v1
  generate_round:              # optional
    command: ["python", "/app/generate_round.py"]
    output_file: /data/generated_tasks.json   # { tasks: [...], sandbox_data: {...} }
    timeout_s: 300
    # process_type: gpu        # optional override — GPU baseline round-gen (e.g. aurelius_steering)
  convert_model:               # optional (replaces ONNX conversion job)
    command: ["python", "/app/convert_to_onnx.py"]
    input_filename: model.pt
    output_filename: model.onnx
    timeout_s: 300
  screen:                      # optional — Layer-2 BEHAVIOURAL screener (bespoke checks only).
    # Runs in its OWN image, so the secret checks can stay private while the player image is
    # public. Exit 0 = pass, non-zero = fail. Only Tron (anti-cosmetic-RL) and aurelius
    # (concept match) need this; most competitions omit it and rely on Layer-1 alone.
    image:
      ref: ghcr.io/some-org/tron-screener
      digest: sha256:aaa111...
    command: ["python", "/app/screen.py"]
    timeout_s: 120
referee:                       # REQUIRED for both solo and duel — the competition-owned scorer
  protocol: gym_v1             # gym_v1 | custom (custom = referee speaks the player's own HTTP)
  image:                       # game engine + scorer; runs by convention entrypoint /app/referee.py
    ref: ghcr.io/some-org/tron-referee
    digest: sha256:def456...
  timeout_s: 900
duel:                          # kind: duel ONLY — match structure (omit for solo, which is 1 player / 1 game)
  players_per_match: 2
  num_games_default: 5
  swap_sides: true
signature:
  cosign_identity: https://github.com/some-org/tron-competition/.github/workflows/release.yml
  cosign_issuer: https://token.actions.githubusercontent.com
```

### Submission artifact types

| Artifact type | What the platform writes               | Where                                                | Used for                                 |
| ------------- | -------------------------------------- | ---------------------------------------------------- | ---------------------------------------- |
| `code`        | Raw bytes of the submitted source file | `submission.target_path` (e.g. `/app/submission.py`) | Today's `.py` submissions (88% of prod). |
| `torchscript` | Raw `.pt` bytes                        | `submission.target_path` (e.g. `/app/model.pt`)      | Today's `.pt` submissions (12% of prod). |
| `onnx`        | Raw `.onnx` bytes                      | `submission.target_path`                             | Post-conversion artifacts.               |

The image's entrypoint is responsible for loading the artifact from `target_path`. The platform does not interpret it. The artifact is written **only to the player sandbox** — the scorer (referee/evaluator) runs in a separate sandbox and never has the submission on its filesystem, so a malicious artifact cannot reach the scoring logic.

### Resource ceilings

The platform enforces a per-env `[floor, ceiling]` band on `resources`. Ceilings cap what any spec can request; the floor exists to catch misconfigured (near-zero) requests, not to force competitions to over-provision:

| Env     | CPU floor | CPU max | Mem floor | Mem max | GPU pools                          |
| ------- | --------- | ------- | --------- | ------- | ---------------------------------- |
| `stage` | 100m      | 2       | 256Mi     | 2 Gi    | none                               |
| `prod`  | 100m      | 4       | 256Mi     | 4 Gi    | GPU opt-in pool, gated by Platform |

Specs outside `[floor, ceiling]` fail validation at sync time — before mirroring or activation. There is no override for the ceiling; the floor is a fixed platform constant, not designer-configurable.

### Spec `defaults` vs platform metadata

The spec's `defaults` block holds **eval and scheduling parameters** the worker/scheduler need to run a competition: baselines, round length, reveal window, score direction (`lower_is_better`). These are competition-definition concerns and travel with the spec version.

**Not in the spec:** economic / subnet parameters such as `incentive_weight` and `burn_factor`. Today they live on the `competitions` table and are set when Platform creates or activates a competition (historically copied from `EVAL_REGISTRY` `default_incentive_weight` at onboarding). External designers do not declare emissions share in their registry PR.

Eventually `incentive_weight` should be **computed dynamically from alpha voting**, not pinned in git or in a designer-owned YAML file. The spec syncer and worker ignore it; the orchestrator/validator weight path ([`src/orchestrator/src/orchestrator/validator/weights.py`](src/orchestrator/src/orchestrator/validator/weights.py)) remains the source of truth, with a future alpha-voting layer updating `competitions.incentive_weight` (or a sibling table) without a spec version bump.

## Lifecycle of a new competition — start to finish

This is the whole flow for standing up a competition, and it is the **same flow internally and for an external designer**. The only step an outside party cannot do is activation (step 6), which is Platform-gated forever. No step touches `apex-mvp`.

### Who owns what

| Actor | Owns | Cannot do |
| ----- | ---- | --------- |
| **Competition designer** (internal or external) | The competition repo: player image, referee image, optional `generate_round` / `convert_model` / `screen` images, `spec.yaml`, `input.schema.json`, fixtures. Opens the registry **spec-add** PR (`competitions/<id>/<version>.yaml`). CODEOWNER of `competitions/<id>/**` (from Phase 6). | Merge to `active/**` (activation), touch `apex-mvp`, or run any Python in a platform process. |
| **Platform** | Reviews spec-add PRs; owns `active/**` (activation), the spec-syncer, the image mirror, and the generic runtime. | — |

### Authoring (designer)

1. **Fork the SDK example.** `apex-competition-sdk` ships an example competition + base images (gym_v1 player/referee bases) and the `apex.competition.v1` JSON Schema. A competition is, at minimum:
   - a **player image** — loads the submission from `submission.target_path` and serves the HTTP API (`/health`, `/reset`, `/act` for `gym_v1`, or the competition's own API for `custom`);
   - a **referee image** — the game engine + scorer, run by convention entrypoint `/app/referee.py`; reads `MATCH_ID`/`SEED`/`CONFIG_JSON`/`PLAYER_URLS`/`NUM_PLAYERS` and writes `/data/result.json`. Required for **both** solo and duel (a solo eval is a 1-player match — see [Solo evaluation isolation](#solo-evaluation-isolation--a-solo-eval-is-a-1-player-duel));
   - a `spec.yaml` + `input.schema.json`, and optionally `generate_round` / `convert_model` / `screen` entrypoints.
2. **Choose screening** (see [Screening](#screening)): set the spec's `screening` block for Layer-1 (generic, no code), and/or ship a private `screen` image + `entrypoints.screen` for bespoke Layer-2 behavioural checks.
3. **Test locally.** `apex-dev preflight` validates the spec + screener against a local fixture; `apex-dev run --spec ./spec.yaml --input fixtures/input.json` runs the eval exactly as the platform would — spinning up the player and referee sandboxes and reading `result.json`. Iterate until green **before** pushing; a spec that validates but can't actually run never gets past this.
4. **Release the images.** Tag a version → the repo's release CI builds each image, pushes it **by digest**, and **cosign-signs** it (keyless, GH OIDC identity = `signature.cosign_identity`). The build emits the player/referee/(screen) digests.
5. **Open the registry spec-add PR.** Add `competitions/<id>/<version>.yaml` (the spec with the digests from step 4 pinned) [+ `input.schema.json`] to `apex-competitions-registry`. Immutable once merged: any change is a **new** `<version>.yaml`. Platform reviews the spec (schema, resource ceilings, referee timeout, signing identity).

### Activation + go-live (Platform + syncer)

6. **Activate.** A Platform admin merges a one-line change to `active/<env>.yaml` (`<id>: "<version>"`), **stage first, then prod**. This is the gate; rollback is a `git revert` of this line.
7. **Sync.** The spec-syncer `CronJob` (per env; also a one-shot on merge) fast-forwards the registry, then for each new `(id, version)`: validates against `apex.competition.v1` + env resource ceilings, cosign-verifies and **mirrors the player/referee/(screen) images by digest** into `ghcr.io/macrocosm-os/apex-images-mirror`, inserts the immutable `competition_spec_versions` row, and points `competition_active_versions` at the version named in `active/<env>.yaml`. A failed sync never blocks running evals — the platform keeps serving the last-activated version.

### Live (platform runtime — no competition code in-process)

8. **A submission arrives.** Layer-1 screening runs in-process at submit (reject on fail). If the spec declares `entrypoints.screen`, the submission is enqueued as a pre-eval `SCREEN` job and stays out of pairing/batching until it passes (see [Screening](#screening)).
9. **Evaluation.** The worker resolves the active spec by `competition_pkg`, dispatches by `kind`/`referee.protocol` to the generic `SoloRunner` / `DuelRunner` template (it imports **no** competition code), pulls the mirrored images **by digest**, and runs the player + referee in **separate sandboxes** on a per-job network. The referee writes `result.json`; the platform reads the scores and delivers `EvaluationResults`.
10. **Update or roll back.** A new competition version repeats steps 4–7 (new tag → new `<version>.yaml` → activation flip). Rollback is a `git revert` of the `active/<env>.yaml` change; the previous version stays mirrored and immediately usable.

**End state for the designer:** shipping and updating a competition is entirely repo-work + one registry PR — no access to, or change in, `apex-mvp`. The platform's trust gate is signed image digests plus Platform-merged active pointers, both reviewable in git. This exact flow was exercised end-to-end on a stage clone for **tron** (duel) and **battleship** (solo) — see the implementation-status note above.

## Runtime — platform side

### 1. DB schema

Two new tables:

- `competition_spec_versions` — immutable.
  - `id` (pk), `competition_id`, `version` (semver), `spec_json` (jsonb), `input_schema_json` (jsonb),
    `image_ref`, `image_digest`, `mirrored_image_ref` (the PLAYER image), `referee_image_ref`, `referee_image_digest`, `mirrored_referee_image_ref` (the REFEREE/scorer image — populated for both solo and duel), `screen_image_ref` (nullable), `screen_image_digest` (nullable), `mirrored_screen_image_ref` (nullable — the optional Layer-2 `entrypoints.screen` image),
    `signature_verified_at`, `created_at`, `git_sha` (registry repo commit that introduced this version)
  - `UNIQUE (competition_id, version)`
- `competition_active_versions` — mutable, one row per env per competition.
  - `env`, `competition_id`, `spec_version_id` (fk), `activated_at`, `activated_by_git_sha`
  - `PRIMARY KEY (env, competition_id)`

Existing `Job` rows gain `spec_version_id` (nullable until phase 2, NOT NULL after migration completes). This is how we attribute "which spec ran this job" for replay/audit.

### 2. Spec syncer — k8s `CronJob`

A new workspace member, `src/spec-syncer/`. Single Python script, no FastAPI, no background loop. Runs as a `CronJob` every 1 minute on stage and every 5 minutes on prod. Triggered manually on registry merge via a `kubectl create job --from=cronjob/spec-syncer` from the registry repo's GH Action for low-latency promotions.

Per run:

1. Clone (or `git pull`) the registry repo at `main` into `/tmp/registry`. Auth via a deploy key in `external-secrets` (read-only).
2. For each `competitions/<id>/<version>.yaml`:
   - Validate against `apex.competition.v1` JSON Schema.
   - Validate `resources` against env ceilings.
   - Resolve `image.digest` (the player) and `referee.image.digest` (required for **both** solo and duel), plus `entrypoints.screen.image.digest` if the spec declares a Layer-2 screener.
   - Verify each image's cosign signature against `signature.cosign_identity`/`cosign_issuer` (identity matched as a regexp, so a `@refs/tags/*` glob in the spec matches the actual per-tag signing identity).
   - If not yet present as a row in `competition_spec_versions`: mirror the image(s) into our registry (copy by digest), then insert the row. Idempotent. A signature failure in warn mode **skips** that spec (not mirrored, not activated) rather than failing the run.
3. Read `active/{env}.yaml`. For each `(competition_id → version)` pair, ensure the matching row in `competition_active_versions` matches; update atomically if not.
4. Emit prometheus metrics: `spec_syncer_runs_total`, `spec_syncer_validation_failures_total{competition,reason}`, `spec_syncer_signature_failures_total`, `spec_syncer_active_drift_seconds` (registry commit → DB activation latency).
5. On any failure: alert (Sentry + Slack via existing channels). The sync run exits non-zero but production keeps running on whatever was last activated — failed sync never blocks evals.

The syncer is the only writer to `competition_spec_versions` and `competition_active_versions`. The orchestrator and worker are readers.

### 3. Runner templates in the worker

`Worker.execute_job` ([src/worker/src/worker/worker.py](src/worker/src/worker/worker.py), currently ~line 451) is replaced by a dispatcher that loads the spec by `competition_pkg` (or `spec_version_id` once jobs carry it) and selects a template by `kind`/`duel.protocol`:

- **`SoloRunner`** — `kind: solo`. A solo evaluation is a **1-player duel** (see [Solo evaluation isolation](#solo-evaluation-isolation--a-solo-eval-is-a-1-player-duel)): it spawns **two sandboxes** — a **player** sandbox running the miner submission behind its HTTP API (`entrypoints.evaluate` + `http_api`), and the competition-owned **referee** sandbox (`referee.image`, the same referee concept duels use) holding the game engine + scorer. The referee drives the single player over the per-job network and writes `/data/result.json`; the platform reads `raw_scores[0]` and returns one `EvaluationResults` (normalization ownership is still open — see [Open questions](#open-questions)). The miner submission and the scorer never share a sandbox — this closes the class of attacks where a `code` submission reads, edits, or patches the scoring logic or fabricates `result.json`. Replaces today's `matrix_compression` / `battleship` / `energy_arbitrage` / `iota_simulator` / `rl_battleship` / `aurelius_steering` runners (the last is GPU — see [GPU competitions](#gpu-competitions)). It reuses the duel machinery wholesale (`Referee` base, per-job network, `sandbox-peer-egress` NetworkPolicy, env injection, digest-pinned mirrored images) — the only differences from `DuelRunnerGymV1` are one player and reading the solo result.
- **`DuelRunnerGymV1`** — `kind: duel`, `protocol: gym_v1`. Spins up N player sandboxes (one per submission, each running `entrypoints.evaluate` exposing the gym_v1 HTTP API) plus 1 referee sandbox from `duel.referee_image` on a shared per-job network. Drives the loop: for `num_games_default`, with `swap_sides`, launch the referee with `PLAYER_URLS` env, await `/data/result.json`, aggregate.
- **`DuelRunnerCustom`** — `kind: duel`, `protocol: custom`. Identical to `DuelRunnerGymV1` minus the `gym_v1` schema enforcement on player endpoints. The referee speaks whatever protocol it wants to the players. Platform contract drops to: provision the network, inject `PLAYER_URLS`, wait for `result.json`.

None of these templates import competition code. Everything they need comes from the spec.

**Incremental result delivery.** The worker already streams per-submission results back to the orchestrator as they finalize, rather than only at end-of-run (`BaseEvaluation.set_result_callback` / `_emit_result`, drained by `Worker._deliver_result`, with an end-of-run sweep for anything that didn't land). The generic templates preserve this: as each submission's (or duel pair's) `result.json` is read, the template emits it through the same callback. This matters for long GPU rounds where we don't want to hold all results until the last sandbox exits. `defer_winner_to_round_completion` is orthogonal — it governs *winner election timing* in the scheduler, not result delivery.

### GPU competitions

`aurelius_steering` is the first GPU competition and the design must not regress it:

- **`process_type: gpu`** in the spec routes eval sandboxes to the GPU nodepool. The per-env resource ceilings already gate GPU behind a Platform-controlled opt-in pool.
- **GPU round generation.** Its `generate_round` runs baseline scoring on a GPU fleet (same machine slug / GPU arch / pinned image as the eval fleet, so the baseline is comparable). In spec terms this is `entrypoints.generate_round` with a `process_type: gpu` override. The competition, not the platform, owns fleet setup inside the image.
- **`defer_winner_to_round_completion: true`** — carried in `defaults` (above); the scheduler reads it to decide when to elect the round winner.

### Screening

Screening is a **two-layer** model, and the externalized design keeps almost all of it OFF the competition author. A survey of all seven `screener_cls` implementations (Jul 2026) is what drives the shape: Layer-1 screening collapses into two shared families plus one outlier. **Code** submissions (battleship, matrix_compression, iota_simulator, energy_arbitrage) all run the same shared `ASTGuard` with two knobs — a max-size cap + optional extra forbidden modules/calls (matrix_compression and iota_simulator are byte-identical; energy_arbitrage is the same skeleton with a stricter list). **Model** submissions (rl_battleship, tron) are literally one class — `TronScreener(RLBattleshipScreener): pass` — validating TorchScript format, size, and code:weight ratio. `aurelius_steering` (safetensors + concept match) is the outlier. So:

1. **Layer 1 — one generic, platform-run screener, configured from the spec.** The platform ships a single screener keyed on `submission.artifact_type` and configured by the spec's `screening` block: `code → ASTGuard + {max_size_mb, extra_forbidden_modules, extra_forbidden_calls}`; `model → weights validator + {max_size_mb, min_weight_bytes, max_code_weight_ratio}`. **No per-competition `screener_cls`** — the ~7 today collapse into config.
2. **Layer 2 — optional, competition-owned `screen` entrypoint (behavioural/bespoke).** The only genuinely per-competition screening is Tron's **behavioural** `sandbox_screener.py` (anti-cosmetic-RL); aurelius's concept-match also fits here. When a spec declares `entrypoints.screen`, the platform runs it as a dedicated pre-eval **`SCREEN` job** (its own `JobType`): a separate, **network-disabled** sandbox from the competition's `screen` image with the submission mounted at `target_path` and the round input provided as for `evaluate`. The screener writes `/data/screen_result.json` = `{"passed": bool, "reason": str}` (the runtime is **fail-closed** — a crash, timeout, or missing file is a `failed` verdict; the entrypoint also exits 0/pass, non-zero/fail as a secondary signal). Most competitions omit it and rely on Layer-1 alone. Because the `screen` image is **separate from the player image**, a competition needing Layer-2 *secrecy* keeps only the `screen` image private — **the player image can be public** (restoring miner-local `apex-dev run`, and shrinking any private-package token to the `screen` image alone).
3. **Layer-2 lifecycle + verdict caching.** The two layers run at different times. **Layer-1** runs in-process at *submission time* — a fail rejects the submission before it is stored. **Layer-2**, when declared, is gated at *submission time* too: the submission is enqueued as a `SCREEN` job and stays `QUEUED`/`EVALUATING` (never `PENDING`) while it screens, so the round manager's `PENDING`-only pairing/batching **auto-excludes it** (no round-manager change needed). On a **pass** the platform makes it eval-eligible (solo → enqueue for eval; duel/GPU → set `PENDING` so the round manager pairs/batches it); on a **fail** it is rejected. The cached verdict (`"passed"`/`"failed"`/`None`) lives on `submit_metadata` and is threaded to the eval job as `JobResponse.screening_status`; the generic `SoloRunner`/`DuelRunner` templates read `runner.screening_statuses` to **skip or forfeit** an already-failed submission rather than re-screening. Feature-flagged by presence of `entrypoints.screen` — off for every competition until one opts in (Tron first).

Two consequences worth calling out:

- **Unblocks legacy-entry deletion.** The generic Layer-1 screener removes the ~7 `screener_cls` wirings — which, together with the generic runner (`runner_cls`) and generator, is what lets the whole `EVAL_REGISTRY` entry be deleted at [Phase 5](#phase-5--platform-cleanup), not just the runner.
- **The trust boundary is the sandbox, not the AST list.** Spec-configured forbidden-lists are visible wherever the spec is; that's acceptable — an AST list is a tripwire, and the sandbox (no network, non-root, seccomp) is the actual defense.

### 4. Image trust

- Pull only happens once, at sync time, into our registry (`ghcr.io/macrocosm-os/apex-images-mirror`). Runtime pulls hit our registry only.
- Cosign signature verified at sync time against the spec's declared identity.
- Digest in spec → digest in our mirror → digest in `pod.spec.containers[].image`. Three points of pin.
- The competition's source registry can disappear and prod keeps running — we have a copy.

## The `gym_v1` duel protocol

Documented in detail in `apex-competition-sdk`; sketched here for the design.

### Player sandbox endpoints

All bodies are JSON. Observation/action shapes are opaque to the platform — the referee defines them per competition. The platform only enforces transport.

| Method | Path      | Body                                       | Response          | Notes                                                                                           |
| ------ | --------- | ------------------------------------------ | ----------------- | ----------------------------------------------------------------------------------------------- |
| `GET`  | `/health` | —                                          | `{ ready: bool }` | Platform readiness probe.                                                                       |
| `POST` | `/reset`  | `{ match_id, player_index, seed, config }` | `204`             | Called once per match per player. `config` is the spec-defined opaque JSON.                     |
| `POST` | `/act`    | `{ observation, deadline_ms }`             | `{ action }`      | Called per turn by the referee. Must respect `deadline_ms` or the referee may forfeit the turn. |

### Referee image contract

Env vars injected by the platform:

- `MATCH_ID` — opaque string, unique per (job, game_index).
- `SEED` — int, the per-game seed.
- `CONFIG_JSON` — the opaque competition config from the spec/round-generator.
- `PLAYER_URLS` — comma-separated, in canonical order. The platform handles `swap_sides` by reordering this string before launching the referee.
- `NUM_PLAYERS` — int.

Output: `/data/result.json` written before the referee container exits:

```json
{
  "raw_scores": [1.0, 0.0],
  "winner": 0,
  "terminal_reason": "player_1_collision",
  "steps": 47,
  "metadata": { "...competition-specific..." }
}
```

Optional `/data/trace.jsonl` (one event per line) for replay/observability. Shipped to S3 by the platform if present.

Failure semantics:

- Player HTTP error or timeout → referee decides (forfeit, retry, draw). Platform doesn't intervene.
- Referee crash or no `result.json` → platform treats as a failed game, scores 0 for all participants, attributes failure to the referee (not to submissions).
- Referee `result.json` parse failure → same as crash.

### Why this works for today's competitions

Mapping the existing `TronEvaluation` (`shared/backend/src/backend/eval/tron/runner.py`) onto `gym_v1`:

- The `python launch_tron_rl.py --port 8001` HTTP server already does the player role; we add `/reset` and `/act` (or rename existing endpoints).
- `competition.tron.tron.run_duel_game` becomes the referee's `main`. The 280 lines of `runner.py` that handle sandbox spin-up, `swap_sides`, per-game retry, and score aggregation collapse into the generic `DuelRunnerGymV1` template.

## Solo evaluation isolation — a solo eval *is* a 1-player duel

**Why this changed.** The original solo template ran the miner submission and the scoring
logic in **one sandbox** (`entrypoints.evaluate.command` loaded the artifact *and* ran the
scorer — see the first battleship image, whose `evaluate.py` launched the submission as a
loopback subprocess and ran the game engine in the same container). That is a security hole:
a `code` submission executes arbitrary code in the same container as the scorer, so it can
read the scoring logic, tamper with intermediate state, or simply overwrite `/data/result.json`
with a perfect score. Duels never had this problem — their scorer (the **referee**) is already a
separate sandbox. **Solo now adopts the identical split — there is no separate "evaluator"
concept; a solo eval is just a referee scoring one player.**

**The model.** A solo evaluation spawns **two sandboxes**, identical to a duel with
`players_per_match: 1`:

- **player** — the miner submission behind its HTTP API. Launched from `image` with
  `entrypoints.evaluate.command`, exposing `http_api` (`/health`, `/reset`, `/act` for `gym_v1`).
  Long-lived server; readiness-gated; `network_disabled: false, allow_internet: false` so it is
  reachable by the referee but has no egress. The artifact is written here and nowhere else.
- **referee** — the competition-owned scorer (game engine + scoring), from `referee.image`
  (digest-pinned, cosign-signed, mirrored). The **same** `Referee` SDK base, `/app/referee.py`
  convention, env contract, and result contract as a duel — just one player:

  - `PLAYER_URLS` — one URL. `SEED`, `CONFIG_JSON` (round input / task config), `NUM_PLAYERS=1`, `MATCH_ID`.
  - Writes `/data/result.json` in the **same** shape duels use:
    `{ "raw_scores": [0.83], "winner": 0, "terminal_reason": "...", "steps": N, "metadata": {...} }`.
    `SoloRunner` reads `raw_scores[0]` as the submission's `eval_raw_score`.
  - Optional `/data/history/*` and `/data/trace.jsonl` → `FileType.HISTORY`.

**Protocol.** `referee.protocol` is `gym_v1` (default) or `custom`, same meaning as for duels.
Turn-based competitions use `gym_v1` — e.g. battleship maps naturally: `/reset` starts a game
(seed + ship config), `/act` returns the next shot given the hit/miss observation. `custom` is
the escape hatch only for competitions that can't express `reset`/`act` (e.g. a compress/decompress
batch API): the referee then speaks the player's own HTTP; the platform only provisions the
network, injects the env, and waits for `result.json`.

**Failure semantics** are the referee contract, unchanged:
- Player HTTP error/timeout → the referee decides (forfeit/retry/zero). Platform doesn't intervene.
- Referee crash / no `result.json` / parse failure → platform scores 0, attributed to the referee.
- A submission pre-screened `failed` forfeits without launching either sandbox (verdict reuse).

**Schema.** The scorer moves out of `duel` into a shared, **required** top-level `referee` block
(`protocol`, `image {ref,digest}`, `timeout_s`) used by both kinds; `duel` keeps only the
match params (`players_per_match`, `num_games_default`, `swap_sides`) and is present only for
`kind: duel`. No backward-compat / optional path: since nothing is live yet (experimental), the
legacy single-sandbox `SoloRunner` is **deleted**, not deprecated.

**Consequences (all in-flight, experimental):**
- **apex-mvp:** `SoloRunner` is reworked to "launch player + referee, drive as a 1-player duel,
  read `raw_scores[0]`" — collapses onto `DuelRunnerGymV1`'s helpers. The spec-syncer and
  `DuelRunnerGymV1` read the referee from `referee.image` (was `duel.referee_image`); the DB
  columns are unchanged (still `referee_image_*`).
- **tron:** its spec restructures (`duel.referee_image` → `referee.image`; `duel` slims to match
  params). **No image rebuild** — the referee image and its `/app/referee.py` are unchanged, so
  the digests stand; only `spec.yaml` + the registry version move fields.

## Phased rollout

Each phase delivers a working production system. Stop at any boundary and run forever. Rollback per phase is documented.

### Phase 0 — Foundations (no behavior change)

**Ships:**

- `apex.competition.v1` JSON Schema (in `apex-competition-sdk`).
- DB migrations for `competition_spec_versions`, `competition_active_versions`, `jobs.spec_version_id` (nullable).
- `src/spec-syncer/` k8s `CronJob`, deployed to stage and prod. With no specs in the registry, it's a no-op.
- `apex-competitions-registry` exists, empty except CODEOWNERS and a README.
- Generic `SoloRunner` and `DuelRunnerGymV1` templates in the worker, **gated behind a feature flag** (`SPEC_DRIVEN_ENABLED=false`). Worker still uses `EVAL_REGISTRY` exclusively.
- **Spec-driven `generate_round` for one pilot competition.** `_process_round_generation_job` in [`src/worker/src/worker/worker.py`](src/worker/src/worker/worker.py) gains a spec-driven branch: when the pilot competition's `SPEC_DRIVEN_ROUND_GEN` flag is on, the worker resolves the active spec, uses `spec.entrypoints.generate_round.command` + the mirrored image instead of the per-job `payload.generator_script_path` + `sb-<env>-<pkg>-<commit>` tag, and continues to read `/data/generated_tasks.json` as today. Other competitions go through the legacy code path unchanged.
  - Rationale: `_process_round_generation_job` is already image-driven and file-output-based, so it's the lowest-risk place to exercise the spec lookup, image-mirror resolution, and per-env active pointer end-to-end before we touch the eval path.
  - **Output contract.** `generated_tasks.json` is `{ tasks: [...], sandbox_data: {...} }`. `tasks` becomes the miner-visible round input; `sandbox_data` is an audit-only payload (e.g. baseline scoring metrics) that the scheduler persists on `round.sandbox_data` and never exposes to miners — large blobs (per-prompt completions, histories) are offloaded to S3 with only a `history_file_path` reference kept inline. The worker already threads this through `JobResults.round_generation_sandbox_data`; the spec-driven branch must preserve it, not just `tasks`.
- `apex-competition-sdk` v0.1 published with: JSON Schema, gym_v1 base images, a `apex-dev run --spec ./spec.yaml --input fixtures/input.json` CLI that runs an eval locally exactly like the platform would.
- Image mirror configured (`ghcr.io/macrocosm-os/apex-images-mirror`).
- Cosign verification working end-to-end on a throwaway test image.

**Success criteria:**

- A toy "hello-world" spec can be pushed to the registry, mirrored, signed, and resolved by the worker.
- The pilot competition's round generation runs end-to-end on stage via the spec-driven branch, producing a `generated_tasks.json` byte-for-byte identical (or semantically equivalent — e.g. modulo timestamps) to the legacy path. The eval for that round still runs through the legacy `EVAL_REGISTRY`-driven path.
- `SoloRunner` runs end-to-end on stage for the toy spec with the feature flag on for that competition only.
- Production unchanged.

**Rollback:** Per-competition flag flip turns the spec-driven round-gen branch off; worker falls back to legacy `_process_round_generation_job`. Migrations, cronjob, sdk repo all deletable as no-ops in prod.

### Phase 1 — First solo competition, shadow mode (no behavior change)

**Ships:**

- `apex-competition-battleship` public repo with: Dockerfile, evaluate entrypoint, generator, normalizer, screener, fixtures, spec YAML. Reference implementation for the public. (Supersedes the earlier plan to pilot with `matrix_compression`, which was dropped — see [Open questions](#open-questions), item 1.)
- Spec activated in **stage** registry only.
- Worker, when handling a `battleship` job in stage, runs **both** paths:
  - Legacy `BattleshipEvaluation` from `EVAL_REGISTRY` (canonical, results returned to orchestrator).
  - Spec-driven `SoloRunner` (shadow, results logged + emitted as `shadow_*` prometheus metrics).
- Comparison job: per-submission score diff. Alert on `|legacy - shadow| > epsilon`.

**Success criteria:** Over one full round in stage, shadow and legacy scores match within tolerance for ≥99% of submissions. Investigate any mismatch before proceeding.

**Rollback:** Deactivate the spec in stage registry (`git revert` on `active/stage.yaml`). Feature flag off for battleship. Worker reverts to legacy-only.

### Phase 2 — First solo competition, prod cutover

**Ships:**

- Spec activated in **prod** registry, still in shadow mode for one round.
- After the shadow round confirms parity, flip `battleship` to **spec-driven only** in prod via a one-line config flag.
- After one full round on the spec path with no incidents, delete `shared/backend/src/backend/eval/battleship/` and `shared/competition/src/competition/battleship/`, drop `battleship` from `EVAL_REGISTRY`.

**Success criteria:** One full prod round on the spec path with no score regressions vs the prior round's distribution. No operational incidents.

**Rollback:** Re-enable legacy path via feature flag; revert deletion PR. Spec stays activated but unused. The spec path can stay dormant in prod indefinitely.

### Phase 3 — First duel: Tron in shadow mode (no behavior change)

**Ships:**

- `apex-competition-tron` public repo with:
  - Player image: `launch_tron_rl.py` refactored to expose `gym_v1` endpoints (`/health`, `/reset`, `/act`).
  - Referee image: `competition.tron.tron.run_duel_game` extracted into a referee `main` that reads `PLAYER_URLS`/`SEED`/`CONFIG_JSON`, runs the match loop, writes `/data/result.json`.
- Spec activated in stage. Worker runs both paths for Tron jobs:
  - Legacy `TronEvaluation` (host-side game loop) — canonical.
  - `DuelRunnerGymV1` with the new player+referee images — shadow.
- Comparison logged per-game, not just per-submission, to catch protocol drift early.

**Success criteria:** Score parity within tolerance over one stage round of duels. No referee crashes. Replay traces from both paths produce equivalent winners on a sample of 50 games.

**Rollback:** Same as Phase 1 — deactivate spec, disable shadow path, legacy continues.

This is the riskiest phase. It validates `gym_v1`, the referee image contract, the per-job network, `swap_sides` handling in the generic template, and the image-mirror pipeline for two images per spec. If this works, the architecture is de-risked.

### Phase 4 — Tron cutover + remaining competitions

**Ships:**

- Tron prod cutover (same shadow-then-flip pattern as Phase 2).
- One-by-one port: `matrix_compression`, `rl_battleship`, `energy_arbitrage`, `iota_simulator`, `aurelius_steering`. Each:
  - Public competition repo created.
  - Stage shadow → stage cutover → prod shadow → prod cutover.
  - Legacy code deleted after one clean prod round.
- `matrix_compression`'s `generate_round` currently depends on historical IOTA run files that no longer exist and can't be shadow-validated as-is (this is why it was dropped as the Phase 0/1 pilot in favor of `battleship`). Its generator needs a fix — either a new self-contained data source or a frozen fixture checked into its competition repo — before this port can start; track as a prerequisite, not a blocker for the other four.
- `aurelius_steering` is the GPU case: it also exercises the GPU nodepool, the GPU round-generation path, and `defer_winner_to_round_completion`. Sequence it last in this phase so the CPU competitions de-risk the template first.
- **Each port folds Layer-1 screening into the spec's `screening` block** (the generic platform screener) and, only if the competition needs behavioural checks, ships a Layer-2 `screen` image — removing that competition's `screener_cls` wiring. Combined with the generic runner and spec-driven round-gen, this is what lets the **whole `EVAL_REGISTRY` entry** be deleted with the legacy code, not just `runner.py` (see [Screening](#screening)). Tron's behavioural `sandbox_screener.py` moves into its own `screen` image at this point, which also lets the Tron **player** image go public.
- Each competition is independent. A regression in `iota_simulator` does not affect `tron`.

**Success criteria:** All seven competitions running spec-driven in prod for one full round with no regressions.

**Rollback:** Per-competition feature flag, then `git revert` of its deletion PR.

### Phase 5 — Platform cleanup

**Ships:**

- Delete `EVAL_REGISTRY` and `shared/backend/src/backend/eval/<pkg>/` directories.
- Delete `shared/competition/` entirely.
- Delete `build-competition-images.yml` GH Action.
- Drop the legacy code path in `Worker.execute_job`; the dispatcher is the only path. Remove feature flag.
- `jobs.spec_version_id` becomes `NOT NULL`.

**Success criteria:** `apex-mvp` has no competition source. No competition-specific code outside the generic dispatcher and templates. CI green.

**Rollback:** Revert the deletion PRs. Costly but possible until Phase 6 opens the gate to external designers.

### Phase 6 — Open the gate to external designers

**Ships:**

- CODEOWNERS in `apex-competitions-registry` updated so competition designer GitHub users own their `competitions/<id>/**` folders; Platform owns `active/**`.
- Cosign signature verification moves from "warn" to "enforce" (Phase 0 already verifies; this phase rejects unverified specs entirely).
- Public onboarding docs in `apex-competition-sdk`: how to fork the example competition repo, build/sign/push, open a PR to the registry.
- A "competition certification" checklist enforced by the sync validator: resource caps respected, schema valid, image signed, referee timeout sensible, etc.

**Success criteria:** An external designer (or someone simulating one) lands a new competition end-to-end through PRs, with no platform code change.

**Rollback:** Tighten CODEOWNERS back to Platform-only on `competitions/**` until issues are sorted.

## Risks and mitigations

| Risk                                                                                                                                             | Mitigation                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Score regression at cutover.** Spec path computes a score subtly different from legacy.                                                        | Shadow mode in stage and prod before every cutover. Compare per-submission; alert on `                                                                                                                                                                                                                               | legacy - shadow | > epsilon`. Hold cutover until parity holds for one full round. |
| **Referee image is the new SPOF for duels.** A bad referee deployment can fail every match.                                                      | Pin by digest. Mirror into our registry. Treat referee image promotion the same as platform deploys (stage-first, prod with bake time). `referee_timeout_s` enforced; referee crash scored as 0 for all participants and *attributed to the referee*, not the submissions (so we don't punish miners for our infra). |
| **Compromised competition image.** External designer publishes a malicious image, or a benign one gets MITM'd.                                   | Pin by digest (no tag mutation). Cosign signature verification at sync. Image mirror means we have a copy if the source registry is taken down. Standard sandbox isolation (`network_disabled` defaults, seccomp profile, no host mounts) limits blast radius even if the image is hostile.                          |
| **Spec syncer falls behind.** Activation latency between merging a registry PR and prod picking it up.                                           | Prom metric `spec_syncer_active_drift_seconds`; alert on >5 min for prod, >2 min for stage. Registry merge triggers a one-shot `Job` via GH Action for immediate sync; the cronjob is the reconciliation safety net.                                                                                                 |
| **JSON Schema input validation rejects valid submissions during migration.** Pydantic ↔ JSON Schema translation drift.                           | The `input_schema` for each competition is emitted from the existing pydantic model during port, not hand-written. Shadow mode catches any schema drift before cutover.                                                                                                                                              |
| **External designer ships a spec that the platform validates but the runtime can't actually execute** (e.g. referee never writes `result.json`). | Local dev harness in `apex-competition-sdk` (`apex-dev run`) executes the spec exactly like the platform — designers can't ship without it passing locally. Stage activation is mandatory before prod.                                                                                                               |
| **Image mirroring storage cost.** Every spec version's image stays in our registry.                                                              | Retention policy: keep all `active` versions across envs + the last N inactive versions per competition. Older versions garbage-collected. Mirror image sizes (~hundreds of MB each) at ~50 competitions × ~10 versions retained = ~250 GB worst case. Cheap on GHCR.                                                |
| **GitOps repo loses its history / is force-pushed.**                                                                                             | Repo is protected (no force-push to `main`, signed commits required on `main`). Activation history is also in DB (`competition_active_versions.activated_by_git_sha`) — we can reconstruct what was live at time T even without git.                                                                                 |

## Open questions

1. ~~**Pilot competition for Phase 0 `generate_round`.**~~ **Resolved: `battleship`.** `matrix_compression`'s generator depends on historical IOTA run files that no longer exist, so it can't be validated end-to-end and is disqualified as a pilot. `battleship` (plain, non-duel, non-RL) has a fully self-contained generator — samples task count/turn limits from a Beta distribution, no external file or S3 dependency (see `BattleshipInputDataGenerator.generate_tasks`) — making it the cheapest round-gen path to validate. `rl_battleship` was considered and rejected: it reuses the same generator but adds RL-specific eval complexity we don't need to pull into Phase 0. `battleship` is now also the Phase 1 shadow target (see below), so this pilot's spec carries forward directly.
2. ~~**`screen` migration timing.**~~ **Resolved: migrate together with `evaluate`, per competition.** Staging `screen` separately from `evaluate` would let shadow comparisons drift silently — if the legacy and spec-driven screeners disagree, we'd misattribute a score mismatch to the eval path instead of the screener. Per the two-layer screening model (see [Screening](#screening)), shadow comparison for each competition's migration must assert **verdict parity** (`screening_status` matches) in addition to score parity before cutover.
3. **ONNX conversion.** Same shape as `generate_round` — `_process_onnx_conversion_job` is already image-driven. Open whether to also pull `convert_model` into Phase 0 alongside `generate_round`, or defer to a later phase. The bespoke `JobType.ONNX_CONVERSION` worker code goes away in Phase 5 regardless.
4. ~~**Per-spec resource defaults vs ceilings.**~~ **Resolved: allow a floor, no ceiling override.** Designers may request resources below a platform-enforced floor (256Mi mem / 100m cpu) — anything lower is almost always a misconfiguration, not a real need, but blocking it outright adds friction for genuinely tiny competitions. Ceilings (see [Resource ceilings](#resource-ceilings)) remain hard: no spec field can push a sandbox above the per-env cap. The spec syncer's static validation gate rejects specs outside `[floor, ceiling]` at sync time — before mirroring or activation.
5. **Migration of in-flight rounds.** A round started under the legacy path must finish under the legacy path. Cutover happens on round boundaries, not mid-round. Concretely: `jobs.spec_version_id` is `NULL` for jobs started under legacy; the worker picks the path by checking that column. Round generation chooses the path for the whole round.
6. **Normalization ownership.** Today `BaseNormalizer` subclasses run in our process, some with non-trivial context (e.g. IOTA's `max_epoch_time`). Two options: (A) the competition's eval sandbox emits `eval_score` directly in `result.json` and the platform treats it as opaque (only `lower_is_better` matters for ranking) — consistent with "platform never runs competition logic"; (B) the spec declares a parametric normalizer the platform applies. Recommend A, but confirm before locking the `result.json` contract. Note `aurelius_steering` also has a `scoring.py` module, so scoring/normalization is already non-uniform across competitions.

## Future work

- **Dynamic `incentive_weight` from alpha voting.** Replace static DB values (and today's `RegistryEntry.default_incentive_weight`) with a subnet-driven allocation updated on a schedule or per epoch. Competitions keep a stable `competition_id`; only the platform weight table changes. No spec or registry PR required when emissions mix shifts.
- **Multi-region spec distribution.** If we ever run more than one prod cluster, the syncer's "one DB per env" assumption needs revisiting. Likely solution: per-region DB, registry is the single source of truth, drift metric per region.
- **Spec hot-reload without job restart.** Today an activated spec change picks up on the worker's next job. For long-running jobs (rare, but possible for evaluations) we may want a midflight cancel + restart. Not needed for current job durations.
- **`code_archive` artifact type.** Adding `.zip` is mechanically simple if we ever need it — write to `target_path` and unzip. Deferred until a competition needs it.
- **Multi-agent (`kind: multi_agent`).** N>2 players without the duel structure. Probably needs a new protocol (`gym_v1_ma`); shape it when we have a real candidate.
- **Submission preflight in `apex-dev`.** Designers can today run `apex-dev run`; we should add `apex-dev preflight` that runs the screener against the designer's local submission fixture before they push, catching schema/screen issues locally.

## Glossary

- **Spec.** A YAML document describing a competition: image refs, schema, entrypoints, eval/scheduling defaults. Validated against `apex.competition.v1`. Does not include `incentive_weight` (platform / alpha voting).
- **Spec version.** An immutable `(competition_id, version)` pair stored in `competition_spec_versions`. Identified by digest of the spec YAML + image digests.
- **Active pointer.** The row in `competition_active_versions` saying which spec version is live in a given env for a given competition.
- **Registry repo.** `apex-competitions-registry`. The GitOps source of truth.
- **Mirror registry.** `ghcr.io/macrocosm-os/apex-images-mirror`. Our copy of every activated image, pinned by digest.
- **Player sandbox.** A k8s `Job`/`Pod` running the competition's player image with a single submission's artifact mounted. Exposes the gym_v1 HTTP API.
- **Referee sandbox.** A k8s `Job`/`Pod` running the competition's referee image. Holds the game logic. Reads `PLAYER_URLS`, writes `/data/result.json`. Owned by the competition.
- **`gym_v1`.** The standardized HTTP wire protocol between a referee and N player sandboxes: `GET /health`, `POST /reset`, `POST /act`.
- **`custom` protocol.** Escape hatch where the referee speaks any protocol to its players; platform only provisions the network and waits for `result.json`.
- **Shadow mode.** Both the legacy and spec-driven paths execute; legacy scores are canonical, spec-driven scores are logged for comparison. Used to validate parity before cutover.

## References

- Existing eval registry: [shared/backend/src/backend/eval/registry.py](shared/backend/src/backend/eval/registry.py)
- Existing eval base classes: [shared/backend/src/backend/eval/base.py](shared/backend/src/backend/eval/base.py)
- Existing Tron host-side runner: [shared/backend/src/backend/eval/tron/runner.py](shared/backend/src/backend/eval/tron/runner.py)
- Existing Tron in-sandbox launcher / game engine: [shared/competition/src/competition/tron/launch_tron_rl.py](shared/competition/src/competition/tron/launch_tron_rl.py), [shared/competition/src/competition/tron/tron.py](shared/competition/src/competition/tron/tron.py)
- Worker dispatch loop: [src/worker/src/worker/worker.py](src/worker/src/worker/worker.py)
- Cosign keyless: <https://docs.sigstore.dev/cosign/signing/overview/>
- JSON Schema 2020-12: <https://json-schema.org/draft/2020-12/release-notes>
- K8s `CronJob`: <https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/>
