#!/usr/bin/env python3
"""Generate an example miner submission file (.safetensors).

This is a GENERATOR script: running it WRITES a submission file you can then feed to
the the Apex competition for Aurelius Steering. The fields of that
submission are broken out as the variables below, each annotated with the rule it must
satisfy.

A submission is a safetensors file with exactly one tensor `direction` plus three
metadata keys (`alpha`, `layer`, `concept`). The scorer validates all of this in
`concept_scorer/submission.py`; the variables below map one-to-one onto those checks.

Run (each invocation writes one .safetensors submission file):
    python scripts/example_submission_generator.py                 # steered example (alpha=12000)
    python scripts/example_submission_generator.py --alpha 0       # unsteered baseline
    python scripts/example_submission_generator.py --concept hedging --out /tmp/sub.safetensors

Then score the file it produced:
    concept-scorer validate --submission example_submission_generator.safetensors --concept positive_sentiment
"""

from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# 1. The tensor: `direction`
#    - shape MUST be (hidden_size,) == (3840,) for gemma-3-12b   -> else BAD_SHAPE
#    - dtype MUST be float32 (F32)                               -> else BAD_DTYPE
#    - MUST be finite                                            -> else NON_FINITE
#    - MUST be L2 unit-norm (||direction|| == 1, tol 1e-3)       -> else NOT_UNIT_NORM
#    A real miner derives this (e.g. diff-of-means at the steer layer); here we just
#    make a deterministic unit vector so the file validates.
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 3840  # gemma-3-12b hidden_size; must equal config model.hidden_size

# ---------------------------------------------------------------------------
# 2. Metadata: alpha / layer / concept (stored as strings in __metadata__)
# ---------------------------------------------------------------------------
# Steering strength. hidden_states += alpha * direction at the steer layer, every token.
# Valid range is [-32000, 32000] (gemma-3-12b layer-32 residual L2 ~57k; <16k is inert,
# >32k degenerates). alpha=0 is a valid UNSTEERED baseline.  -> else ALPHA_OUT_OF_BOUNDS
ALPHA = 12000.0

# The steer layer. MUST equal config model.steer_layer (32).   -> else BAD_LAYER
LAYER = 32

# Which competition concept this vector targets. MUST equal the active_concept the
# scorer is run with: birthday_cake | medical_disclaimer | positive_sentiment | hedging.
#                                                              -> else CONCEPT_MISMATCH
CONCEPT = "positive_sentiment"

# Reproducible RNG seed for the demo direction (NOT part of the submission format).
DEMO_SEED = 0


def build_unit_direction(hidden_size: int, seed: int):
    """A deterministic, L2-normalized float32 `(hidden_size,)` vector."""
    import torch

    g = torch.Generator().manual_seed(seed)
    v = torch.randn(hidden_size, generator=g, dtype=torch.float32)
    v = v / v.norm(p=2)  # unit-norm so ||direction|| == 1
    return v.contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default=CONCEPT)
    ap.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="0 for an unsteered baseline; else a steered value in [-32000, 32000]",
    )
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--seed", type=int, default=DEMO_SEED)
    ap.add_argument("--out", default="example_submission_generator.safetensors")
    args = ap.parse_args()

    from safetensors.torch import save_file

    direction = build_unit_direction(HIDDEN_SIZE, args.seed)
    assert direction.shape == (HIDDEN_SIZE,)

    tensors = {"direction": direction}
    metadata = {
        "alpha": str(args.alpha),
        "layer": str(args.layer),
        "concept": args.concept,
    }

    save_file(tensors, args.out, metadata=metadata)
    print(f"generated submission file -> {args.out}")
    print(f"  direction: shape=({HIDDEN_SIZE},) dtype=float32 unit-norm")
    print(f"  metadata:  {metadata}")


if __name__ == "__main__":
    main()
