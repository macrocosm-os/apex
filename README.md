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

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/macrocosm-os/apex.git
   cd apex
   ```

2. **Install `uv`:**
   Follow the instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv) to install `uv`. For example:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install the project and its development dependencies:**
   ```bash
   uv venv && uv python install 3.11 && uv python pin 3.11 && uv venv --python=3.11 && uv pip install -e '.[dev]'
   ```

4. **Activate python environment:**
  ```bash
  . .venv/bin/activate
  ```

## Run Mainnet Validator

1. Prepare config file:
   ```bash
   cp config/mainnet.yaml.example config/mainnet.yaml
   # Fill in the required values in config/mainnet.yaml
   ```

2. **Run the validator:**
   ```bash
   python validator.py -c config/mainnet.yaml
   ```

## Run Testnet Validator

1. Prepare config file:
   ```bash
   cp config/testnet.yaml.example config/testnet.yaml
   # Fill in the required values in config/testnet.yaml
   ```

2. **Run the validator:**
   ```bash
   python validator.py -c config/testnet.yaml
   ```

## Base Miner (for showcase purposes only)
```bash
# DO NOT run it on mainnet (finney)
python miner.py
```

## Description
Apex structure:
```
.
└── Apex/
    ├── apex/
    │   ├── common/  # Common files used across project.
    │   ├── validator/  # Validator related stuff.
    │   └── services/  # Service modules used by validator.
    ├── config/  # Config files for validator.
    ├── scripts/  # Scripts.
    ├── tests/  # Unit tests.
    └── .github/  # Github actions CI/CD.
```

## Development

### Add packages
```bash
uv add new-package
uv lock
```

### Test mode: Pool of predefined UIDs

One can define a pool of UIDs which will be queried, ignoring all others UIDs in the metagraph.

Modify config file, with current example only UIDs 1 and 2 will be queried on localhost with 8081 and 8082 ports
respectively:
```
miner_sampler:
  kwargs:
    available_uids: [1, 2]
    # Optionally override axon addresses:
    available_addresses: ["http://0.0.0.0:8081", "http://0.0.0.0:8082"]
```
