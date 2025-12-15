from pydantic import BaseModel


class MinerScore(BaseModel):
    """Miner's incentive details"""

    uid: int
    hotkey: str
    weight: float | None = None


class SubnetScores(BaseModel):
    """Details about a subnet's scores (weights)"""

    miner_scores: list[MinerScore]
