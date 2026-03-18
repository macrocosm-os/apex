"""Models for the IOTA Simulator competition HTTP API.

These are the schemas that miners use in their /route endpoint.
"""

from pydantic import BaseModel


class MinerInfo(BaseModel):
    """Information about a single miner node, provided to routing algorithms.

    Note: A miner can host multiple layers simultaneously. The same miner may
    appear in multiple layer lists in the RouteRequest.

    Only fields exposed by the upstream simulator's get_miner_information()
    are included. Internal miner state (queues, cache, processing times,
    bandwidth, latency) is not available.
    """

    miner_id: str
    receiver_node_id: str
    active: bool
    layers: list[int]
    local_batch_size: int
    peer_latencies: dict[str, float]


class RouteRequest(BaseModel):
    """Request payload sent to the miner's /route endpoint."""

    activation_id: str
    miner_id: str
    miners: list[list[MinerInfo]]
    n_layers: int
    activation_tracking_buffer: list[dict]


class RouteResponse(BaseModel):
    """Response from the miner's /route endpoint."""

    path: list[str]  # List of node_ids: [node_id_l0, node_id_l1, ..., node_id_ln]


class BalanceRequest(BaseModel):
    """Request payload sent to the miner's /balance-orchestrator endpoint."""

    miners: dict[str, MinerInfo]  # Keyed by miner_id
    n_layers: int
    activation_tracking_buffer: list[dict]


class BalanceResponse(BaseModel):
    """Response from the miner's /balance-orchestrator endpoint."""

    assignments: dict[str, list[int]]  # miner_id -> list of layer indices
