# Validators

> Run the scoring infrastructure that evaluates solver submissions and distributes rewards.

Validators are the backbone of Apex's evaluation: they run every submission in an isolated sandbox against each competition's scoring function, keep leaderboards current, and distribute emissions on-chain. Scoring is fair and reproducible — every submission is evaluated on the same terms.

For an overview of what Apex is and how competitions work, see the main [README](./README.md).

## Quick links

[Validator setup](https://docs.macrocosmos.ai/subnets/new-subnet-1-apex/validating) · [Auto-updater script](./scripts/README.md) · [Incentive mechanism](https://docs.macrocosmos.ai/subnets/new-subnet-1-apex/incentive-mechanism)

## What validators do

- **Evaluate** — run each submission in an isolated sandbox against the competition's objective function `f(x) → ℝ`.
- **Rank** — maintain a continuously-updating leaderboard as new submissions arrive and as solvers iterate.
- **Reward** — distribute emissions winner-takes-all to the solver holding the top-ranked submission, via the [incentive mechanism](https://docs.macrocosmos.ai/subnets/new-subnet-1-apex/incentive-mechanism).

## Getting started

Run the validator to continuously score submissions and distribute rewards. Follow the [validator setup guide](https://docs.macrocosmos.ai/subnets/new-subnet-1-apex/validating) for installation, configuration, and operational requirements.

To keep your validator current with releases, use the [auto-updater script](./scripts/README.md).

The validator implementation lives in [`src/validator`](./src/validator).

## Community

Visit the **apex** channel in the [Macrocosmos Discord](https://discord.gg/vdyz4JZ9Ww) or the [Bittensor Discord](https://discord.gg/GtgHWakpDs) for questions, feedback, and support.
