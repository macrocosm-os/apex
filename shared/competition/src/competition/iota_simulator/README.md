# IOTA Simulator Competition

Route activations through a simulated distributed compute network faster than random.

## Overview

The IOTA simulator models a distributed network where **activations** (units of work) must travel through a sequence of **layers**, each hosted by one or more **miners**. Your job is to implement a routing algorithm that decides which miner to use at each layer for each activation, and a balancing algorithm that assigns miners to layers between epochs.

Each task runs **multiple epochs** (default: 5). Your total epoch time — the sum of all epoch durations, excluding merge phases — is scored against a pre-computed time ceiling. Lower is better.

## How the Simulation Works

### Network Structure

- The network has `n_layers` compute layers (varies per task, between 3 and 8).
- 96 miners are distributed across layers. A single miner can host multiple consecutive layers (e.g., layers [1, 2]).
- Miners connect via a simulated network with bandwidth limits, latency, and message overhead.

### Miner Classes

Miners are not homogeneous. They are drawn from three reliability classes:

| Class         | Share | Drop Probability | Join Probability |
| ------------- | ----- | ---------------- | ---------------- |
| Reliable      | 75%   | Very low         | High             |
| Semi-reliable | 15%   | Low              | Medium           |
| Unreliable    | 10%   | Moderate         | Moderate         |

All classes share the same latency, bandwidth, and processing time ranges. The difference is in how often they drop out and how quickly they rejoin. Individual miner properties are sampled uniformly from each class's ranges.

### Activation Lifecycle

Each activation follows this path:

```
Orchestrator
    |
    v
Layer 0 miner (forward) --> Layer 1 miner (forward) --> ... --> Layer N-1 miner (forward)
                                                                       |
                                                                       v  (direction reversal)
Layer 0 miner (backward) <-- Layer 1 miner (backward) <-- ... <-- Layer N-1 miner (backward)
    |
    v
Orchestrator (activation complete)
```

1. **Forward pass**: The activation is created and routed through layers 0, 1, ..., N-1. At each miner, the activation enters a **forward queue**, waits for processing, then is sent to the next node in the path.
2. **Direction reversal**: At the final layer, the activation switches to backward direction.
3. **Backward pass**: The activation returns through layers N-1, N-2, ..., 0. At each miner, it enters a **backward queue** and is processed.
4. **Completion**: When the activation finishes its backward pass at layer 0, it's counted as complete.

### Epochs and Merge Phases

- An **epoch** completes when `target_activations` (default: 500) have finished their full forward + backward pass.
- Each task runs **multiple epochs** (default: 5).
- Between epochs, a **merge phase** occurs (30 seconds of simulated time):
  - All miner queues and caches are cleared.
  - Your `/balance-orchestrator` endpoint is called to reassign miners to layers.
  - Epoch-to-epoch drift is applied — miner properties (latency, bandwidth, processing time) may shift slightly between epochs.
  - Merge phase duration is **not** counted toward your score.

### Multi-Layer Miners

A miner hosting multiple layers (e.g., [1, 2]) processes all its layers in a single step without network hops between them. Routing an activation to a multi-layer miner saves transfer time for the layers it covers.

### Queuing and Processing

Each miner maintains two queues:

- **Forward queue**: Activations waiting for forward processing.
- **Backward queue**: Activations waiting for backward processing. **Backward has priority** — miners always process backward activations before forward ones.

### Caching

After forward processing, miners **cache** the activation. When the same activation returns for backward processing, it's recognized via cache hit and routed to the backward queue.

- Each miner has a limited cache capacity (`max_ml_cache_size`, default: 5). When the cache is full, the miner **stalls forward processing** until a cached activation completes its backward pass and frees a slot.
- Routing too many activations to the same miner can cause cache-pressure stalls.

### Network Transfer

Moving an activation between miners costs time based on bandwidth limits, per-miner latency, global network latency, and message overhead. Miners with higher bandwidth and lower latency transfer activations faster. The `peer_latencies` field on each miner reflects the measured latency to other miners.

### Miner Liveness

Miners can go offline during the simulation (based on their class's drop probability) and rejoin later. Inactive miners should generally be avoided, though they may come back online. The `active` field on `MinerInfo` reflects current heartbeat status.

## Your Task

Implement an HTTP server with two endpoints: `POST /route` for routing activations through the network, and `POST /balance-orchestrator` for rebalancing miner layer assignments between epochs.

Your `/route` endpoint is called for every activation (~2,500 calls per task). Your `/balance-orchestrator` endpoint is called once per epoch boundary (~4 calls per task).

### API Contract

**`GET /health`** — Health check. Return `{"ok": true}`.

**`POST /route`** — Route an activation.

Request body ([`RouteRequest`](models.py)):

| Field                        | Type                    | Description                                                        |
| ---------------------------- | ----------------------- | ------------------------------------------------------------------ |
| `activation_id`              | `str`                   | Unique ID for this activation                                      |
| `miner_id`                   | `str`                   | ID of the miner requesting the route (always a layer-0 miner)      |
| `miners`                     | `list[list[MinerInfo]]` | Miners available at each layer. `miners[0]` = layer 0 miners, etc. |
| `n_layers`                   | `int`                   | Number of layers in the network                                    |
| `activation_tracking_buffer` | `list[dict]`            | Recent activation event history (see below)                        |

Response body ([`RouteResponse`](models.py)):

| Field  | Type        | Description                                                                                           |
| ------ | ----------- | ----------------------------------------------------------------------------------------------------- |
| `path` | `list[str]` | List of `receiver_node_id` values, one per layer: `[layer_0_node, layer_1_node, ..., layer_n-1_node]` |

**Notes:**
- `path` should have exactly `n_layers` entries.
- `path[0]` should be the `receiver_node_id` of the requesting miner (identified by `miner_id`). The layer-0 miner is the entry point. If `path[0]` doesn't match, it will be corrected automatically.
- Each `path[i]` should be a valid `receiver_node_id` from `miners[i]`.
- Invalid paths are **not** rejected. Instead, the simulation will attempt to use the path as-is. Bad routing decisions (e.g., sending to a non-existent node or wrong layer) will cause the activation to fail naturally — it won't complete, won't count toward the epoch, and your epoch time will suffer.

**`POST /balance-orchestrator`** — Rebalance miner layer assignments.

Called between epochs during the merge phase. You decide which layers each miner should serve for the next epoch.

Request body ([`BalanceRequest`](models.py)):

| Field                        | Type                   | Description                                              |
| ---------------------------- | ---------------------- | -------------------------------------------------------- |
| `miners`                     | `dict[str, MinerInfo]` | All miners in the network, keyed by `miner_id`           |
| `n_layers`                   | `int`                  | Number of layers in the network                          |
| `activation_tracking_buffer` | `list[dict]`           | Recent activation event history from the completed epoch |

Response body ([`BalanceResponse`](models.py)):

| Field         | Type                   | Description                                                     |
| ------------- | ---------------------- | --------------------------------------------------------------- |
| `assignments` | `dict[str, list[int]]` | Maps each `miner_id` to a list of layer indices it should serve |

**Notes:**
- Each miner should be assigned at least one layer.
- Layer indices must be in `[0, n_layers)`.
- A miner can be assigned to multiple layers (multi-layer hosting reduces network hops).
- Every layer should have at least one miner assigned; layers with no miners will cause activations to stall.
- Invalid assignments (e.g., out-of-range layers, missing miners) are silently ignored — affected miners keep their previous assignments.

### MinerInfo Fields

Each `MinerInfo` describes a miner's observable state:

| Field              | Type               | Description                                                                                      |
| ------------------ | ------------------ | ------------------------------------------------------------------------------------------------ |
| `miner_id`         | `str`              | Unique miner identifier                                                                          |
| `receiver_node_id` | `str`              | Network address — this is what you put in the path                                               |
| `active`           | `bool`             | Whether the miner is currently online (based on heartbeat). Inactive miners may drop activations |
| `layers`           | `list[int]`        | Layers this miner hosts. A miner in `miners[1]` with `layers=[1,2]` covers two layers in one hop |
| `local_batch_size` | `int`              | Batch size used for processing                                                                   |
| `peer_latencies`   | `dict[str, float]` | Measured latency (seconds) to each peer miner, keyed by `miner_id`                               |

Internal miner state (queue lengths, cache occupancy, processing times, bandwidth) is **not** directly exposed. Use the `activation_tracking_buffer` to infer throughput and congestion from observed event history.

### Activation Tracking Buffer

The `activation_tracking_buffer` contains recent event records from completed and in-progress activations. Each entry is a dict with:

| Field           | Type            | Description                                                                   |
| --------------- | --------------- | ----------------------------------------------------------------------------- |
| `activation_id` | `str`           | Which activation this event belongs to                                        |
| `direction`     | `str`           | `"forward"` or `"backward"`                                                   |
| `action`        | `str`           | Event type: `"receive"`, `"processing"`, `"queuing"`, `"notify_orchestrator"` |
| `start_time`    | `str`           | ISO timestamp when the event started                                          |
| `end_time`      | `str` or `null` | ISO timestamp when the event ended (null if ongoing)                          |
| `miner_id`      | `str`           | Which miner recorded this event                                               |
| `layer`         | `int`           | Which layer                                                                   |

The buffer is bounded (up to ~1,920 entries) and cleared between epochs. You can use it to observe throughput, identify congested miners, or track how long activations spend queuing vs. processing. The baseline ignores this data entirely — using it is an opportunity for improvement.

## Scoring

### Per-Task Score

Each task has a `max_epoch_time` — a time ceiling derived analytically from the network configuration with a safety multiplier. Your total epoch time is the **sum of all epoch durations** across the task's epochs, excluding merge phases.

```
task_score = 1 - (total_epoch_time / max_epoch_time)
```

The score is clamped to [0.0, 1.0]:

| Your Performance                  | Score |
| --------------------------------- | ----- |
| Hit or exceeded the ceiling       | 0.0   |
| Finished in half the allowed time | 0.5   |
| Near-instant completion           | ~1.0  |

A task scores 0 if:
- Your total epoch time equals or exceeds `max_epoch_time`
- The task times out (wall-clock)
- Any `/route` or `/balance-orchestrator` call times out

### Final Score

The final score is the **median** across all evaluation tasks (default: 5 tasks, each with a different random seed and number of layers).

## Timeouts

| Timeout                          | Value                             | What Happens                                                                                                                                                                                        |
| -------------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/route` response                | 5 seconds                         | Any single `/route` call that exceeds this fails the **entire task** with a score of 0. Your handler must be fast.                                                                                  |
| `/balance-orchestrator` response | 5 seconds                         | Any single `/balance-orchestrator` call that exceeds this fails the **entire task** with a score of 0.                                                                                              |
| Layer timeout                    | 60 seconds (simulated time)       | If an activation doesn't progress past a layer within 60s of simulated time, it is dropped. Stale activations still sitting in a miner's queue or cache are also cleaned up. The activation doesn't count toward epoch completion — this slows your epoch but doesn't zero the task. |
| Task timeout                     | 30s + 120s per epoch (wall-clock) | If the full simulation doesn't finish within this budget, the task scores 0. For 5 epochs: 630 seconds.                                                                                             |

The `/route` timeout is the most important constraint: your routing logic must respond well under 5 seconds. Heavy computation risks timing out and zeroing the task.

## Optimization Levers

**Routing (`/route`):**
- **Multi-layer miners**: Prefer miners hosting multiple consecutive layers to eliminate network hops.
- **Latency-aware routing**: Use `peer_latencies` to pick miners with low latency to the previous hop.
- **Load balancing**: Spread activations across miners to avoid queue buildup and cache pressure.
- **Adaptive routing**: Use `activation_tracking_buffer` to infer congestion and throughput from event history.
- **Liveness-aware routing**: Prefer active miners to avoid dropped activations.

**Balancing (`/balance-orchestrator`):**
- **Even layer distribution**: Ensure each layer has enough miners to handle the activation throughput.
- **Multi-layer assignment**: Assign miners to consecutive layers to reduce network hops.
- **Adaptive rebalancing**: Use `activation_tracking_buffer` from the previous epoch to identify bottlenecks.

## Simulation Data

After the round completes, you can download your simulation data file containing detailed logs from your evaluation:
- **activation_logs**: Per-activation lifecycle events
- **miner_metrics**: Periodic snapshots of miner state (queue lengths, cache usage, active status)
- **route_log**: Every routing decision made
- **balance_log**: Every balancing assignment made

The file is named `{submission_id}_sim_data.json`.

## Submission

Your submission is a single Python file named `solution.py` that runs an HTTP server.

The server must expose `GET /health`, `POST /route`, and `POST /balance-orchestrator` endpoints. See [`baseline.py`](baseline.py) for a minimal working example.

### Environment

- Python 3.12
- Pre-installed packages: `fastapi`, `uvicorn`, `pydantic`, `numpy`, `orjson`, `httpx`
- Additional packages can be added to `dockerfiles/requirements.txt`
- No internet access at runtime
- Maximum submission size: 50,000 characters

## Files

| File                           | Description                                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| [`models.py`](models.py)       | Pydantic schemas for `MinerInfo`, `RouteRequest`, `RouteResponse`, `BalanceRequest`, `BalanceResponse` |
| [`baseline.py`](baseline.py)   | Random routing baseline                                                                                |
| [`dockerfiles/`](dockerfiles/) | Docker build context for submissions                                                                   |
