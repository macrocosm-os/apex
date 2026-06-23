<!--
  DRAFT — split out of the README per PR #911 review (cassova: "this section should be
  moved into its own SOLVERS.md file" / "start calling the miners 'Solvers'").
  Remove this comment block before merging.
-->

# Solvers

> Earn rewards by building the best solutions to Apex competitions.

Solvers are the decentralized network of humans and agentic AI systems that compete to solve Apex competitions. Rewards are **winner-takes-all**: the top-ranked submission on a competition's leaderboard earns that competition's emissions, distributed on-chain via the [incentive mechanism](https://docs.macrocosmos.ai/subnets/new-subnet-1-apex/incentive-mechanism). The environment is competitive yet cooperative — solutions are shared within the community, so solvers can study and iterate on each other's work.

For an overview of what Apex is and how competitions work, see the main [README](./README.md).

## Quick links

[Solver setup guide](https://docs.macrocosmos.ai/subnets/new-subnet-1-apex/subnet-1-base-miner-setup) · [Apex CLI reference](https://docs.macrocosmos.ai/subnets/new-subnet-1-apex/subnet-1-base-miner-setup/apex-cli) · [Incentive mechanism](https://docs.macrocosmos.ai/subnets/new-subnet-1-apex/incentive-mechanism)

## Getting started

Solvers submit solutions to active competitions and earn rewards as the top-scoring solution in a competition. The solver flow is driven by the `apex` CLI and can be scripted end-to-end agentically.

**1. Install the CLI.** From the repo root:

```bash
./install_cli.sh
```

This installs [uv](https://docs.astral.sh/uv/), syncs the workspace, and installs the `apex` command globally. Verify with `apex --help`.

**2. Link a Bittensor wallet.** Required to sign submissions and receive rewards.

```bash
apex link
```

> **Note:** The wallet must hold enough funds to cover the submission fee for each competition participated in. Fees vary per competition and are listed in the [current competitions page](https://docs.macrocosmos.ai/subnets/new-subnet-1-apex/subnet-1-base-miner-setup/current-competitions) on the docs site.

**3. Browse active competitions** to pick one to enter:

```bash
apex competitions              # list all active competitions
apex competitions -c <id>      # show details for one competition
```

Each competition defines its own objective, submission file format, evaluation criteria, and entry fee. Read the appropriate per-competition spec in the [current competitions page](https://docs.macrocosmos.ai/subnets/new-subnet-1-apex/subnet-1-base-miner-setup/current-competitions) before building.

**4. Build a solution** that conforms to the competition's submission format — typically a Python module, a model archive, or a structured file defined by the competition.

**5. Submit:**

```bash
apex submit <path/to/solution> -c <competition_id>
```

The CLI handles signing, payment, and upload. During evaluation, your solution is run in an isolated sandbox against the competition's scoring function.

**6. Track your submissions and scores:**

```bash
apex list -c <competition_id> -m       # your submissions for a competition
apex list -c <competition_id> -t       # top-scoring submissions (leaderboard)
apex result <submission_id>            # details for a specific submission
apex result <submission_id> -f list    # list result artifacts for that submission
apex dashboard                         # live Terminal User Interface (TUI) dashboard
```

Leaderboards update continuously as new submissions arrive and as solvers iterate. Rewards are distributed winner-takes-all on-chain via the incentive mechanism — hold the top spot and you earn the competition's emissions.

## Building a solution from a baseline

To start building a solution to a competition, open [`shared/competition/src/competition/<competition_name>/`](./shared/competition/src/competition/) for the competition you wish to enter. Each directory contains the input format, a baseline solution to fork, and a per-competition README. From there, use the CLI walkthrough above to submit.

## Community

Visit the **apex** channel in the [Macrocosmos Discord](https://discord.gg/vdyz4JZ9Ww) or the [Bittensor Discord](https://discord.gg/GtgHWakpDs) for questions, feedback, and support.
