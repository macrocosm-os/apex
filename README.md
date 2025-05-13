<picture>
    <source srcset="./assets/macrocosmos-white.png"  media="(prefers-color-scheme: dark)">
    <source srcset="./assets/macrocosmos-black.png"  media="(prefers-color-scheme: light)">
    <img src="macrocosmos-black.png">
</picture>

<div align="center">

# **Bittensor SN1** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### The Incentivized Internet <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)

</div>

---

This repository is the **official codebase for Bittensor Subnet 1 (SN1) v1.0.0+, which was released on 22nd January 2024**. To learn more about the Bittensor project and the underlying mechanics, [read here.](https://docs.bittensor.com/).

# Introduction

This repo defines an incentive mechanism to create a distributed conversational AI for Subnet 1 (SN1).

Validators and miners are based on large language models (LLM). The validation process uses **internet-scale datasets and goal-driven behaviour to drive human-like conversations**.


</div>

# Usage

<div align="center">

**[For Validators](./docs/validator.md)** · **[For Miners](./docs/epistula_miner.md)** · **[API Documentation]((./docs/API_docs.md))**


</div>

# Agentic Tasks

Subnet one utilizes the concept of "Tasks" to control the behavior of miners. Validator create a variety of tasks, which include a "challenge" for the miner to solve, and sends them to 100 miners, scoring all the completions they send back.

## Task Descriptions

### 1. **Inference**
A question is given with some pre-seeded information and a random seed. The miner must perform an inference based on this information to provide the correct answer. Completions are scored based on similarity metrics.

### 2. **Multistep Reasoning (MSRv2)**
This task operates in two stages: generative and discriminative. 
In the generative stage, a single miner receives a challenge and generates a response. 
In the discriminative stage, this generated response (or sometimes a validator-provided "real" answer) is presented to a set of discriminator miners. These discriminators must output a score (0-1) assessing the answer. 
Rewards are then calculated: discriminators are rewarded based on how accurately their score reflects the ground truth (i.e., whether the answer was miner-generated or real). The original generator miner is rewarded based on the collective assessment of the discriminators. If a "real" answer was used, this portion of the reward is distributed among other non-discriminating miners.

### 3. **Web Retrieval**
The miner is given a question based on a random web page and must return a scraped website that contains the answer. This requires searching the web to locate the most accurate and reliable source to provide the answer. The miner is scored based on the embedding similarity between the answer it returns and the original website that the validator generated the reference from.

# API Documentation

For detailed information on the available API endpoints, request/response formats, and usage examples, please refer to the [API Documentation](./docs/API_docs.md).

# Contribute
<div align="center">

**[Contribution guide](./assets/CONTRIBUTING.md)**

</div>
