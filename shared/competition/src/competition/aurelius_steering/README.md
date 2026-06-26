# Aurelius Steering Competition

A partnership between **Aurelius (SN37)** and **Macrocosmos (SN1)** to crowdsource the discovery of *concept directions* inside a language model — vectors that, when added to the model's activations, reliably steer its output toward a target concept.

## What this is

In 2024 Anthropic released [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude): by amplifying a single internal feature they made the model obsess over the Golden Gate Bridge, mentioning it in nearly every reply regardless of the prompt. That demo showed something powerful — a model's behavior can be *steered* by manipulating a direction in its internal representation space, not by retraining or prompting.

This competition turns that idea into an open, incentivized search. Miners submit a steering direction for a hidden concept; the validator adds that direction to the model's residual stream during generation and measures how strongly the resulting completions exhibit the concept. The miner whose direction steers the model most effectively wins the round.

The goal is to explore — at scale and across many independent miners — how well a model can be steered toward a *specified* concept using a single learned direction.

## How it works at a glance

1. Each round targets one **concept** (e.g. *positive sentiment*). Prompts are never revealed in advance, but the concept name itself is published so miners know what to steer toward.
2. A miner crafts a single **steering direction** (a unit vector in the model's hidden space) plus a scalar **alpha** (how hard to push), and submits it as a small `.safetensors` file.
3. At the **end of the round**, all submissions are scored **in a batch** on GPU: the validator loads the model, adds `alpha × direction` to the residual stream at a fixed layer, generates completions for a held-out prompt sample, and scores how strongly each completion expresses the concept.
4. Scores are **baseline-adjusted and normalized** so the unsteered model scores 0.0 and a perfect steer scores 1.0. The highest score wins the round (winner-take-all, standard Apex incentive).

## The model and the steering mechanism

- **Model:** `google/gemma-3-12b-it` at a pinned revision, loaded **4-bit NF4** with **bf16** compute. All miners calibrate against this identical configuration — the steering directions are specific to it.
- **Hidden size:** `3840`. A direction is a vector of this dimension.
- **Steer layer:** `32` (fixed for the whole competition). The validator registers a forward hook that, at **every token position** during generation, does:

  ```
  hidden_states[layer 32] += alpha * direction
  ```

- **Decoding:** greedy (`do_sample=False`, `num_beams=1`), fixed per-round seed, short completions. Greedy + fixed seed makes scoring reproducible across machines on the same GPU architecture and pinned image.

Because the direction is **L2-normalized to unit norm**, `alpha` is the *sole* knob controlling steering strength, and submissions are directly comparable across miners.

## Concepts

The scorer is concept-agnostic; each round names one active concept. The four supported concepts:

| Concept | What a steered completion looks like |
|---|---|
| `birthday_cake` | Mentions birthday cake, candles, parties, "happy birthday" |
| `medical_disclaimer` | Adds disclaimers like "consult a healthcare professional", "not medical advice" |
| `positive_sentiment` | Skews strongly positive in tone (AFINN-style net valence) |
| `hedging` | Uses hedging language — "perhaps", "it might", "arguably", "it depends" |

Each concept is scored by a deterministic, version-pinned lexicon/detector, so results are reproducible.

**Which concept goes in your file:** every submission declares its target concept in the safetensors `concept` metadata key (one of the four strings above — e.g. `"birthday_cake"`). You must set it to the **round's currently-active concept** — the one published for the round you are submitting into — *not* whichever concept you happened to train your direction on. The active concept is announced per round and changes manually over time (see [Round structure](#round-structure-and-defaults)); a direction built for last round's concept must be re-declared (and ideally re-derived) for the new one. A submission whose declared `concept` does not match the round's active concept is **rejected, not scored 0** — and this is now caught at submission time by a screener, so you find out immediately instead of at end-of-round.

## Submission format

A single `.safetensors` file (~15 KB) containing exactly one tensor plus three metadata keys:

| Field | Requirement |
|---|---|
| tensor `direction` | shape `(3840,)`, dtype `float32`, finite, **L2 unit-norm** (tolerance `1e-3`) |
| metadata `alpha` | float in `[-32000, 32000]` — the steering strength; `0` is a valid unsteered baseline |
| metadata `layer` | must equal `32` |
| metadata `concept` | one of the four concept strings — must equal the **round's active concept** (the published concept for this round), not the concept you trained on |

Every failed check is a *typed rejection* (bad shape/dtype, non-unit-norm, non-finite, wrong layer, concept mismatch, alpha out of bounds), not a silent zero. The reference generator in [scripts/example_submission_generator.py](scripts/example_submission_generator.py) documents each rule inline.

> **Tuning `alpha`:** for gemma-3-12b at layer 32 the residual L2 magnitude is ~57k. Empirically, `alpha` below ~16k tends to be inert, the useful band is roughly **12k–16k**, and above ~20k the model degenerates into repetition. Start around `alpha=12000` and sweep from there.

## Evaluation

Evaluation runs against a **pinned scorer image** built from Aurelius' open-source [`concept-competition`](https://github.com/Aurelius-Protocol/concept-competition/tree/main) repository. The image hosts an API server that loads the model on a GPU and exposes a `/score` endpoint; the Apex validator provisions a GPU fleet, posts each submission to the scorer, and collects the resulting score and completions.

The scorer image is **digest-pinned** in the backend (`aurelius-steering-scorer`, see `DEFAULT_SCORER_IMAGE`), and the eval fleet runs on the same GPU architecture and image that the baseline was scored on, so all scores in a round are comparable.

### Batched, end-of-round evaluation

Unlike per-submission competitions, **evals are done in a batch at the end of the round**. The runner receives the full round's submissions in one job, provisions one GPU per mini-batch (`evals_per_gpu` submissions each), and scores them. Every submission produces exactly one result, error or not.

### Baseline and normalization

1. **Baseline:** at round generation, the validator scores an **unsteered** submission (a unit vector with `alpha=0`, so the model output is unaffected) on a freshly drawn round seed. This `baseline_score` is the concept's natural hit rate with no steering.
2. **Raw score:** each submission is posted to the scorer, which returns the concept score (a hit-fraction in `[0, 1]`). The eval pipeline takes the returned number as the submission's `raw_score` (stored as `eval_raw_score`) — *raw* meaning **before normalization**. If the round's optional [minimal-intervention reward](#minimal-intervention-reward-optional-off-by-default) is enabled, the scorer has already applied it before returning the number; from the validator's side this is just the score that came back.
3. **Headroom normalization:** the `eval_score` we store in the DB removes the baseline and rescales by the available headroom, so every concept maps the unsteered model to 0.0 and a perfect steer to 1.0:

   ```
   eval_score = clip((raw_score - baseline_score) / (1 - baseline_score), 0, 1)
   ```

   This makes scores comparable across concepts: a hard concept (high baseline) and an easy one (low baseline) are judged purely on how much of the *remaining* headroom the miner captures.

### Minimal-intervention reward (optional, off by default)

As of the **v0.1.4** scorer, a round may optionally reward steering that achieves the concept with a **gentler push** — a smaller total intervention on the residual stream — over brute-forcing the detector with an enormous one. (This replaces the experimental Hoyer-sparsity penalty from v0.1.3, which is gone in v0.1.4.)

This all happens **inside the scorer**, before the eval pipeline sees a number. When enabled, the scorer multiplies its pre-reward concept score by an `efficiency` factor in `(0, 1]`, derived from the submission's total absolute steering `push`, to produce the `score` it returns:

```
push       = |alpha| * sum(|direction|)        # total absolute steering applied
efficiency = exp(-push / push_scale)            # in (0, 1]; larger push_scale = gentler discount
```

A bigger `push_scale` tolerates more steering before discounting; as `push_scale → ∞` the factor → 1 (no discount). The day-score gates it (a submission that generates nothing on-concept stays at 0), and `efficiency ∈ (0, 1]` with `raw_score ∈ [0, 1]`, so the product stays in `[0, 1]`.

**It is off by default** and is enabled per round, by precedence **request > per-concept config > off**:

- **From the competition (the apex path):** set `push_scale` in the competition's `input_data_generator_args`. Round generation stamps it onto the round input ([`AureliusSteeringInputDataSchema.push_scale`](../../../../backend/src/backend/eval/aurelius_steering/models.py)) and the eval runner sends it on **every** `/score` call in the round — so the whole round (and its baseline) is scored under one setting. Omit it (or `null`) to leave the reward off. A recommended starting value is **~555000**.
- **In the scorer config:** a non-null `push_scale` for a concept in the scorer's `config/competition.yaml` (used when the request omits it).

`push` is **always computed and reported**, even when the reward is off, so it can be calibrated against the real distribution before being switched on. The `/score` response (and each submission's history file) carries:

| Scorer response field | Meaning |
|---|---|
| `score` | the concept score the scorer returns, **after** the reward (`= raw_score × efficiency`) |
| `raw_score` | the scorer's score **before** the reward |
| `push` | total absolute steering, `|alpha| × Σ|direction|` |
| `push_scale` | the scale in effect for this request (`null` = off) |
| `efficiency` | the multiplier applied: `exp(-push / push_scale)` (`1.0` when off) |

> **⚠️ `raw_score` means two different things across the two layers.** The scorer's `raw_score` is its *pre*-reward score. The eval pipeline's `raw_score` (what we store as `eval_raw_score`) is the scorer's *post*-reward **`score`** field. The full chain:
>
> ```
> score      (scorer)  =  raw_score (scorer) × efficiency
> raw_score  (eval / eval_raw_score)  =  score (scorer)
> eval_score (DB)      =  clip((raw_score − baseline_score) / (1 − baseline_score), 0, 1)
> ```
>
> With the reward off (the default), `efficiency = 1.0`, so the scorer's `raw_score` and `score` are equal and the distinction collapses.

> **Note on the baseline.** The unsteered baseline submission has `alpha = 0`, so `push = 0` and `efficiency = exp(0) = 1` for any `push_scale` — the baseline is never discounted. So even with the reward on, a (discounted) miner score is normalized against an undiscounted `baseline_score`. This is intended — the baseline is the no-steer reference point — but worth being aware of before enabling it.

## Round structure and defaults

| Setting | Default |
|---|---|
| Round length | **1 day** (`default_round_length_in_days = 1`) |
| Submission code visible after | **1 day** (`default_submission_reveal_days = 1`) — submissions stay hidden until then, then are published |
| Prompt sample size per round | `150` — configured per round |

- **No logs:** this competition produces **no execution logs**. There is nothing to inspect mid-round.
- **History file:** at the **end of the round**, each scored submission has a **history file** — the full scorer response, including the **prompt and the resulting inference (completion) for each prompt**. This is the artifact miners use to see exactly how their direction steered the model.
- **Concept cycling:** the active concept is **changed manually after X rounds**, by a joint decision between **Macrocosmos and Aurelius**. It does not rotate automatically on a fixed schedule.

## Tips for miners

### Building a steering direction

- A direction is a single unit vector in the model's 3840-dim hidden space at layer 32. A common, effective approach is **diff-of-means**: collect layer-32 activations on text that *does* express the concept and text that does not, take the difference of the two means, and L2-normalize it. That difference vector points from "not the concept" toward "the concept".
- Other approaches: probing classifiers (use the learned weight vector as the direction), sparse-autoencoder features, or contrastive activation pairs. Anything that yields a meaningful direction at layer 32 is fair game.
- **Always work against the canonical config** (gemma-3-12b-it, 4-bit NF4, bf16, layer 32). Directions derived from a differently-quantized or different-layer setup will not transfer cleanly.
- **Then tune `alpha`** against real steered completions. The direction sets *what* to push toward; `alpha` sets *how hard*. Sweep `alpha` (≈12k–16k) and watch for the sweet spot before the model degenerates.
- Normalize before submitting: `direction = direction / ||direction||`.

### Regenerating the baseline and a random steer

This folder ships a generator and two reference files you can reproduce:

- [scripts/example_submission_generator.py](scripts/example_submission_generator.py) — writes a valid `.safetensors` submission, with each field annotated by the rule it must satisfy.
- [scripts/baseline-zero-steer-positive_sentiment.safetensors](scripts/baseline-zero-steer-positive_sentiment.safetensors) — an unsteered (`alpha=0`) baseline submission. The filename's `positive_sentiment` suffix is the concept stamped in its metadata; for a different round, regenerate with that round's `--concept`.
- [scripts/sample-random-steer-positive_sentiment.safetensors](scripts/sample-random-steer-positive_sentiment.safetensors) — a random-direction steered submission for the `positive_sentiment` concept (a sanity-check / lower bound, not a real strategy).

Regenerate them from the `scripts/` directory. Pass `--concept` to stamp the round's active concept into the file's metadata — it defaults to `positive_sentiment`, so set it explicitly for any other round or your submission will be rejected with a concept mismatch. The commands below reproduce the committed `positive_sentiment` reference files; swap in the active concept (and filename) for your round:

```sh
# Unsteered baseline (alpha = 0)
python scripts/example_submission_generator.py --concept positive_sentiment --alpha 0 --out baseline-zero-steer-positive_sentiment.safetensors

# Random steered direction (deterministic via --seed), alpha = 12000
python scripts/example_submission_generator.py --concept positive_sentiment --alpha 12000 --seed 0 --out sample-random-steer-positive_sentiment.safetensors
```

The `--concept` value must be one of the four supported concepts and must equal the round's active concept. The random/baseline files are starting points and references — a competitive submission replaces the random `direction` with one you've actually derived for the active concept.

## How to submit

1. Build your `direction` for the round's active concept and tune `alpha`.
2. Write the `.safetensors` submission (use the generator as a template). Confirm it has the single `direction` tensor (`(3840,)`, `float32`, unit-norm) and the `alpha` / `layer=32` / `concept` metadata.
3. Submit it through the standard Apex miner flow for the `aurelius_steering` competition before the round closes.
4. All submissions are scored together when the round ends; your submission's code is revealed after the reveal window, and the history file lets you review the per-prompt completions your direction produced.

## References

- Anthropic — [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude)
- Aurelius — [concept-competition (scorer source)](https://github.com/Aurelius-Protocol/concept-competition/tree/main)
