from pydantic import BaseModel


class MinerScore(BaseModel):
    """Miner's incentive details"""

    uid: int
    hotkey: str

    # Assigned run_id
    run_id: str

    # The score for the given time window for this run
    # Calculated by: sum(scores within time window) * multipler
    total_score: float

    # Percentage of the incentive_perc assigned to this miner (these total to 1.0 across all miners *in the run*)
    # Calculated by: total_score / all total_scores for the run
    run_weight: float | None = None

    # Overall weight for this hotkey (these total to 1.0 across all miners)
    # Calculated by: weight_in_run * (1 - run's burn rate) * run's incentive_perc
    weight: float | None = None


class RunIncentiveAllocation(BaseModel):
    """Run incentive allocation details"""

    run_id: str

    # Weight of the run that determines percentage of incentive allocated for this run
    incentive_weight: float

    # Percentage of incentive allocated for this run
    # Calculated by: incentive_weight / total_incentive_weight
    incentive_perc: float | None = None

    # How much of the allocated incentive is burned for this run
    burn_factor: float


class SubnetScores(BaseModel):
    """Details about a subnet's scores (weights)"""

    miner_scores: list[MinerScore]
    runs: list[RunIncentiveAllocation]

    # Overall burn factor calculated for the subnet
    # Calculated by: 1 - sum(all miner weights)
    burn_factor: float
