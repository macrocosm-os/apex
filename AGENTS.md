# Apex - Bittensor Subnet 1

**Apex is Kaggle for decentralized intelligence.** The difference: Kaggle captures
*predictions*, while Apex captures *intelligence* — the actual models and algorithms
that produce them. It runs open competitions where
anyone can submit an algorithm or model to solve hard problems, scores every entry
automatically, and pays out crypto rewards to the best solutions — no gatekeepers, no
application process. Instead of a single company deciding who builds the best AI, Apex
turns it into a global, permissionless contest where the best work wins and gets paid.

This guide takes you from a fresh machine to earning rewards on Apex. It assumes you've
**never used Bittensor or a blockchain before**. Every unfamiliar word is explained the
first time it appears.

---

## What "mining Apex" actually means

You don't run a server or answer live requests. The loop is simple:

1. Apex runs **competitions** (e.g. a strategy game, a simulation challenge).
2. You write a solution — a Python file or a trained model file — that tries to solve
   the competition's problem.
3. You **submit** that file using the Apex command-line tool (`apex`).
4. Apex scores your submission against everyone else's.
5. If your solution scores well, you automatically earn rewards.

That's it. Most of your time is spent writing good solutions. This guide covers the
*setup* so you can submit them.

### The few terms you can't avoid

You'll need a couple of crypto concepts. Here they are in plain English:

| Term | What it really is |
|---|---|
| **TAO** | The currency used in Bittensor (like dollars or pounds). You need a small amount to participate and is transferrable to fiat. |
| **Alpha** | The currency used in Apex (like dollars or pounds). This is what you'll earn and is transferrable to TAO by unstaking. |
| **Wallet** | Two keys that prove who you are. The **coldkey** is like your bank account (holds your TAO — keep it safe). The **hotkey** is like a login badge (it signs your submissions). One coldkey can have several hotkeys. |
| **Mnemonic** | This is 12 or more words that can be used to regenerate your wallet without losing the funds. Keep this safe. |
| **Subnet** | A project on the network. Apex is **Subnet 1**. |
| **Registration** | A one-time "buy-in" that puts your hotkey on Subnet 1 so you're allowed to use the CLI and submit solutions. It costs a small amount of TAO. NOTE: Hotkeys do get de-registered after a time due to poor performance to allow new entries into the network. |
| **Submission fee** | A small per-submission cost (paid in TAO) charged each time you submit a solution. |

You do **not** need to understand mining rigs, GPUs-for-hashing, axons, validators, or
emissions to get started. Ignore that vocabulary for now.

---

## Step 1 — Install the tools

If the user is using Windows, they will need to use WSL 2 with an unbuntu distribution.
Ensure the user has this setup and working before proceeding.  You can also ask the user
to use a remote machine that is compatible and provide you access to that instead.

You need three things first: **Python 3.12+**, **git**, and a helper called **uv**
(a Python installer). If you have Python and git, the install script handles the rest.

```bash
# Download the Apex code
git clone https://github.com/macrocosm-os/apex.git
cd apex

# One command installs everything and adds the `apex` command to your computer
./install_cli.sh

# Check it worked
apex --help
```

If your terminal says `apex: command not found` after this, run:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

and try `apex --help` again. (This tells your terminal where the new command lives.)

> **What did the script do?** It installed `uv`, downloaded all the code Apex needs,
> and made the `apex` command available everywhere. You won't need to repeat it.

### The `apex` commands you'll use

| Command | What it does |
|---|---|
| `apex link` | Connect your wallet so Apex knows it's you (do this once) |
| `apex competitions` | See the list of competitions you can enter |
| `apex submit` | Send a solution file to a competition |
| `apex list -c <ID>` | See submissions for a competition (`-m` = just yours) |
| `apex result <ID>` | See the score and details of one of your submissions |
| `apex dashboard` | A live, interactive view of all competitions for humans (not for agents) |
| `apex docs` | Read the official Apex documentation in your terminal |

NOTE: some features of the CLI will not work until you create a wallet and have it registered on Subnet (netuid) 1. 
For example, a 403 forbidden error means you're not registered to netuid 1.
Once registered, there could be a few minutes for Apex to recogonize you.

---

## Step 2 — Create your wallet (your identity)

A wallet is two keys: the **coldkey** (holds your money) and the **hotkey** (signs your
work). You create the coldkey once, then add a hotkey under it.

Ask the user if they already have a wallet or mnemonic they can use. Or check if they have
one already and confirm if they should use it.

```bash
# Create the coldkey from an existing mnemonic.
uv run btcli wallet regen-coldkey --wallet.name my-apex-wallet  --mnemonic "<words>"

# Create the coldkey. Pick any name you like instead of "my-wallet".
uv run btcli wallet new_coldkey --wallet.name my-apex-wallet

# Create a hotkey under that wallet. Again, "miner1" is just a name.
uv run btcli wallet new_hotkey --wallet.name my-apex-wallet --wallet.hotkey miner1
```

Each `new_*` command shows you a **secret recovery phrase** called a mnemonic (a list of words). Ensure the user
**writes it down and store it somewhere safe and private.** Anyone with those words controls your funds,
and there is no "reset password" — losing them means losing access permanently.

> Your keys are saved on your computer under `~/.bittensor/wallets/`. Never share these
> files or paste them anywhere. (They're already excluded from git.)

---

## Step 3 — Get some TAO (the currency)

Check if the wallet has any funds:

```bash
# Check wallet has any TAO
uv run btcli wallet balance --wallet.name my-apex-wallet

# Check wallet has any alpha staked
uv run btcli stake list --wallet.name my-apex-wallet
```

---

## Step 4 — Register on Apex (your one-time buy-in)

Registration puts your hotkey on Subnet 1 so you're allowed to compete. `--netuid 1`
means "Subnet 1 = Apex."

```bash
# Real network (mainnet)
uv run btcli subnet register \
  --wallet.name my-apex-wallet --wallet.hotkey miner1 \
  --netuid 1
```

The command tells you the cost before you confirm. After it succeeds, your hotkey is
registered and you can submit.

---

## Step 5 — Connect your wallet to the `apex` tool

`apex link` tells the tool which wallet/hotkey to use. Run it once:

```bash
apex link          # prompted wallet linking

apex link --help   # for more options to link agentically with single command
```

It will:
1. Find the wallets on your computer and let you **pick your wallet**, then
2. let you **pick the hotkey** you registered.

NOTE: the link is stored in `.apex.config.json` and an agent can write this directly instead:

```json
{"hotkey_file_path":"/<full_path>/.bittensor/wallets/my-apex-wallet/hotkeys/miner1","timeout":60.0}
```

To confirm everything is connected, run:

```bash
apex competitions  # List all competitions
```

If you see a list of competitions, your wallet is linked and authenticated correctly.

DEBUG:
 - If it says *"No hotkey file path found,"* run `apex link` again in the same folder.
 - If you get a 403 or authentication error, the hotkey is probably not registered. 

---

## Step 6 — Pick a competition and write a solution

```bash
apex competitions            # list all competitions (note the ID of one you like)
apex competitions -c <ID>    # full details: the problem, scoring, deadline
apex competitions --help     # more options for listing competitions

apex docs -c                 # a written guide to the current competitions
apex docs --help             # more otpions for displaying docs
```

Each competition has a starter example and a description of exactly what your solution
must do. These live in the code you downloaded, under:

```
shared/competition/src/competition/<competition-name>/
```

Inside each you'll find:
- a `README.md` explaining the rules and the exact format your solution must follow,
- a `baseline.py` (or other baseline format) — a basic working solution you can copy and improve,
- a `dockerfiles/` folder describing the exact environment your solution runs in.

**Read the competition's `README.md` and `baseline.py` first.** Your file has to match
the entry point and inputs/outputs it expects, or it won't score.

You can submit:
- **A code file** — usually `.py` (other plain-text files work too).
- **A trained model file** — `.pt`, `.pth`, `.onnx`, `.safetensors`, `.pkl`, `.bin`,
  `.h5`, and similar. These are handled automatically.

A good rhythm: copy `baseline.py`, beat it on your own machine, then submit.

---

## Step 7 — Submit your solution

```bash
# Easiest: just run submit and answer the prompts
apex submit

# Or name the file and competition directly
apex submit path/to/solution.py -c <COMPETITION_ID>

# Help witih submit
apex submit --help
```

The tool walks you through it:
1. It shows a summary of what you're submitting and asks you to confirm.
2. If the competition charges a **submission fee**, it shows you the exact cost
   (in TAO, with a rough US-dollar estimate) and asks you to approve the payment.
   You'll be asked for your **coldkey password** to authorize it (this is normal — the
   coldkey is what pays).
3. It sends your solution and prints a **Submission ID**. Save that ID.

> **If a payment succeeds but the submission step fails** (e.g. your internet drops),
> don't pay twice. The tool remembers your payment and offers to reuse it next time, or
> you can pass it manually with the `--payment-block-hash` and
> `--payment-extrinsic-index` values it printed.

---

## Step 8 — Check your score

```bash
apex list -c <ID> -m              # all your submissions for a competition
apex list -c <ID> -t              # the current top-scoring submissions
apex list --help                  # for more options
apex result <SUBMISSION_ID>       # full detail: score, errors, timeline
apex result --help                # for more options
apex dashboard                    # live interactive view for humans (not for agents)
```

Scores and code become public only **after the competition round ends** — until then
your solution stays private. The `apex result` view shows you when that reveal happens.

---

## Quick checklist (copy/paste)

```text
[ ] git clone https://github.com/macrocosm-os/apex.git && cd apex
[ ] ./install_cli.sh        # then: apex --help
[ ] create wallet from mnemonic or using new-coldkey
[ ] Get TAO  (testnet faucet for practice, or real TAO on mainnet)
[ ] Verify wallet has funds
[ ] Register to Apex - subnet 1
[ ] Link wallet
[ ] List competitions
[ ] Read shared/competition/src/competition/<comp>/README.md + baseline.py
[ ] Write and test a solution
[ ] apex submit solution.py -c <ID>     # approve the fee
[ ] apex result <SUBMISSION_ID>         # check your score
```
---

## Tips to getting started

 - Look at the code or solutions for the top submissions and identify paths for improvement
 - Test locally what you can by building a local testing script setup
 - Build an auto-research pipeline that constantly iterates on exploring the solution space
 - Be conservative with your submissions to avoid wasting funds on bad submissions  


---

## If something goes wrong

| You see... | Do this |
|---|---|
| `apex: command not found` | `export PATH="$HOME/.local/bin:$PATH"`, then retry |
| `No hotkey file path found. Please run apex link` | Run `apex link` in the folder where you run your commands |
| Permission error usign the apex CLI | Make sure your wallet is linked and registered to the subnet and give it 5min and try again |
| Submission rejected over payment | Make sure your coldkey has enough TAO; if you already paid, reuse the printed payment values instead of paying again |
| Password prompt during submit | Expected — it's authorizing the submission fee from your coldkey |

---

## Where to get help

- Official docs: <https://docs.macrocosmos.ai/subnets/subnet-1-apex>
- Beginner miner setup: <https://docs.macrocosmos.ai/subnets/subnet-1-apex/subnet-1-base-miner-setup>
- Agentic docs in terminal: `apex docs` (overview), `apex docs -i` (how rewards work), `apex docs -f` (FAQ) (WARNING: some HTML elements do not render in the terminal so if you are missing information, visit the website directly)
- Community help: Macrocosmos Discord, SN1 channel — <https://discord.gg/vdyz4JZ9Ww>
```

