<picture>
    <source srcset="./docs/macrocosmos-white.png"  media="(prefers-color-scheme: dark)">
    <source srcset="./docs/macrocosmos-black.png"  media="(prefers-color-scheme: light)">
    <img src="macrocosmos-black.png">
</picture>

<div align="center">

# **Bittensor SN1: Apex - The Future of Decentralized AI** <!-- omit in toc -->
[![Apex CI/CD](https://github.com/macrocosm-os/apex/actions/workflows/python-package.yml/badge.svg)](https://github.com/macrocosm-os/apex/actions/workflows/python-package.yml)
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
<!-- [![codecov](https://codecov.io/gh/macrocosm-os/apex/branch/main/graph/badge.svg)](https://codecov.io/gh/macrocosm-os/apex) -->
<!-- [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/macrocosm-os/apex) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### The Incentivized Internet <!-- omit in toc -->
Subnet 1 is the most intelligent inference model on Bittensor. As the first agent to achieve deep-researcher reasoning on the protocol, it has long been the network's pioneer of decentralized intelligence. This repository is the **official codebase for Bittensor Subnet 1 (SN1) v1.0.0+, which was released on 22nd January 2024**.

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)

</div>

---

## Dashboard

To see the latest competitions and rankings right from your terminal, open the dashboard after installing the CLI:

```bash
apex dashboard
```

NOTE: you'll first need to link your wallet using `apex link`. If you don't have a wallet, follow the Miner Guide below.

## Miner Guide

To get started run `./intall_cli.sh` and follow the instructions.

Example miner's solutions are located at: [shared/competition/src/competition](shared/competition/src/competition) look for `miner_solution.py` files in each competition.

*WARNING: Example solutions are only for educational purposes, it won't produce any yields and as a result the registration fee will be lost.*

### Participate

1. Register on Apex subnet:

    - Create a wallet and hotkey: [Bittensor official docs](https://docs.learnbittensor.org/keys/working-with-keys)
    - Register your hotkey: `btcli s register --wallet.name <WALLET> --wallet.hotkey <HOTKEY> --netuid 1 --network finney`.

2. Link your wallet by running `apex link` (choose your registered hotkey).

3. For submission, run: `apex submit` (choose file location of your solution, competition and round id).

## Run Validator

1. **Clone the repository:**
   ```bash
   git clone https://github.com/macrocosm-os/apex.git
   cd apex
   ```

2. **Install UV**

3. **Setup `.env`**
   Copy `.env.template` to `.env` and fill in your wallet details

3. **Start the validator**
   ```bash
   ./start_validator.sh
   ```
