"""Conservative baseline policy matching the Rust baselines::conservative module."""

from __future__ import annotations
from typing import TYPE_CHECKING

from competition.energy_arbitrage.python import constants

if TYPE_CHECKING:
    from .challenge import State
    from .policy_view import PolicyView

CHARGE_THRESHOLD: float = 0.95
DISCHARGE_THRESHOLD: float = 1.05
MAX_FLOW_ADJUST_ITERS: int = 64
GLOBAL_SCALE_BSEARCH_ITERS: int = 32
EPS: float = 1e-12


def _compute_flows(challenge: "PolicyView", state: "State", action: list[float]) -> list[float]:
    injections = challenge.compute_total_injections(state, action)
    return [
        sum(challenge.network.ptdf[l][k] * injections[k] for k in range(challenge.network.num_nodes))
        for l in range(challenge.network.num_lines)
    ]


def _most_violated_line(challenge: "PolicyView", flows: list[float]) -> tuple[int, float, float] | None:
    """Returns (line, flow, violation_amount) or None."""
    best = None
    for l, flow in enumerate(flows):
        limit = challenge.network.flow_limits[l]
        violation = abs(flow) - limit
        if violation > constants.EPS_FLOW * limit:
            if best is None or violation > best[2]:
                best = (l, flow, violation)
    return best


def _is_flow_feasible(challenge: "PolicyView", state: "State", action: list[float]) -> bool:
    flows = _compute_flows(challenge, state, action)
    return _most_violated_line(challenge, flows) is None


def _soften_most_violated_line(
    challenge: "PolicyView",
    line: int,
    flow: float,
    violation_amount: float,
    action: list[float],
) -> bool:
    signed_direction = 1.0 if flow > 0 else (-1.0 if flow < 0 else 0.0)
    if abs(signed_direction) <= EPS:
        return False

    worsening_indices = []
    worsening_strength = 0.0
    for i, battery in enumerate(challenge.batteries):
        contribution = challenge.network.ptdf[line][battery.node] * action[i]
        signed_contribution = signed_direction * contribution
        if signed_contribution > EPS:
            worsening_strength += signed_contribution
            worsening_indices.append(i)

    if not worsening_indices or worsening_strength <= EPS:
        return False

    keep = max(0.0, min(1.0 - violation_amount / worsening_strength, 1.0))
    if abs(1.0 - keep) <= EPS:
        return False

    for i in worsening_indices:
        action[i] *= keep
    return True


def _enforce_profit_floor(challenge: "PolicyView", state: "State", action: list[float]) -> list[float]:
    """Scale actions to maintain non-negative total profit."""
    profit = challenge.compute_profit(state, action)
    if state.total_profit + profit >= 0.0:
        return action
    action = list(action)
    while state.total_profit + challenge.compute_profit(state, action) < 0.0:
        for i in range(len(action)):
            if abs(action[i]) < EPS:
                action[i] = 0.0
            else:
                action[i] *= 0.95
    return action


def _enforce_flow_feasibility(challenge: "PolicyView", state: "State", action: list[float]) -> list[float]:
    action = list(action)

    for _ in range(MAX_FLOW_ADJUST_ITERS):
        flows = _compute_flows(challenge, state, action)
        violation = _most_violated_line(challenge, flows)
        if violation is None:
            return action
        line, flow, amount = violation
        if not _soften_most_violated_line(challenge, line, flow, amount, action):
            break

    if _is_flow_feasible(challenge, state, action):
        return action

    zero = [0.0] * len(action)
    if not _is_flow_feasible(challenge, state, zero):
        raise ValueError("Baseline fallback failed: grid is infeasible even with zero battery actions")

    base = action
    low = 0.0
    high = 1.0
    for _ in range(GLOBAL_SCALE_BSEARCH_ITERS):
        mid = 0.5 * (low + high)
        scaled = [mid * u for u in base]
        if _is_flow_feasible(challenge, state, scaled):
            low = mid
        else:
            high = mid

    return [low * u for u in base]


def policy(challenge: "PolicyView", state: "State") -> list[float]:
    """Conservative price-based charge/discharge policy (matches Rust conservative::policy)."""
    if state.time_step >= len(challenge.market.day_ahead_prices):
        raise ValueError(f"Missing day-ahead prices for time_step {state.time_step}")

    da_prices = challenge.market.day_ahead_prices[state.time_step]
    if len(da_prices) != challenge.network.num_nodes:
        raise ValueError(
            f"Day-ahead prices length ({len(da_prices)}) does not match network nodes ({challenge.network.num_nodes})"
        )

    avg_da = sum(da_prices) / len(da_prices)
    action = [0.0] * challenge.num_batteries

    for i, battery in enumerate(challenge.batteries):
        node_price = da_prices[battery.node]
        min_bound, max_bound = state.action_bounds[i]
        can_full_charge = min_bound <= -battery.power_charge_mw + constants.EPS_SOC
        can_full_discharge = max_bound >= battery.power_discharge_mw - constants.EPS_SOC

        if node_price < CHARGE_THRESHOLD * avg_da and can_full_charge:
            action[i] = -battery.power_charge_mw
        elif node_price > DISCHARGE_THRESHOLD * avg_da and can_full_discharge:
            action[i] = battery.power_discharge_mw
        else:
            action[i] = 0.0

        action[i] = max(min_bound, min(action[i], max_bound))

    action = _enforce_flow_feasibility(challenge, state, action)
    return _enforce_profit_floor(challenge, state, action)
