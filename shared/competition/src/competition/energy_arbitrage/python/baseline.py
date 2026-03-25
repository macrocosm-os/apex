"""Default baseline policies: re-export greedy and conservative for compute_baseline()."""

from competition.energy_arbitrage.python.greedy import policy as greedy_policy
from competition.energy_arbitrage.python.conservative import policy as conservative_policy

__all__ = ["greedy_policy", "conservative_policy"]
