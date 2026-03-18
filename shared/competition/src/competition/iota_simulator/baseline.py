"""Baseline submission for the IOTA Simulator competition.

Implements random routing: for each layer, picks a random miner.
This is what the evaluation baseline uses, so submitting this should score ~1.0.
"""

import random
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


class MinerInfo(BaseModel):
    miner_id: str
    receiver_node_id: str
    active: bool
    layers: list[int]
    local_batch_size: int
    peer_latencies: dict[str, float]


class RouteRequest(BaseModel):
    activation_id: str
    miner_id: str
    miners: list[list[MinerInfo]]
    n_layers: int
    activation_tracking_buffer: list[dict]


class RouteResponse(BaseModel):
    path: list[str]


class BalanceRequest(BaseModel):
    miners: dict[str, MinerInfo]
    n_layers: int
    activation_tracking_buffer: list[dict]


class BalanceResponse(BaseModel):
    assignments: dict[str, list[int]]


def make_app() -> FastAPI:
    app = FastAPI(title="IOTA Simulator Baseline")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/route", response_model=RouteResponse)
    def route(request: RouteRequest):
        """Random routing: pick a random active miner at each layer."""
        path = []
        for layer_idx, layer in enumerate(request.miners):
            active_miners = [m for m in layer if m.active]
            if not active_miners:
                active_miners = layer  # fallback to all miners if none active

            if layer_idx == 0:
                # Layer 0: use the entry point miner
                entry = next(
                    (m for m in layer if m.miner_id == request.miner_id),
                    random.choice(active_miners),
                )
                path.append(entry.receiver_node_id)
            else:
                path.append(random.choice(active_miners).receiver_node_id)

        return RouteResponse(path=path)

    @app.post("/balance-orchestrator", response_model=BalanceResponse)
    def balance_orchestrator(request: BalanceRequest):
        """Random balancing: assign each miner to a single random layer."""
        assignments: dict[str, list[int]] = {}
        for miner_id in request.miners:
            layer = random.randint(0, request.n_layers - 1)
            assignments[miner_id] = [layer]
        return BalanceResponse(assignments=assignments)

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(make_app(), host="0.0.0.0", port=args.port)
