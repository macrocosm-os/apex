"""Greedy baseline policy matching the Rust baselines::greedy module (forward-looking)."""

from __future__ import annotations
from typing import TYPE_CHECKING

from competition.energy_arbitrage.python import constants

if TYPE_CHECKING:
    from .challenge import State
    from .policy_view import PolicyView

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
    """Greedy forward-looking policy (matches Rust greedy::policy)."""
    time_step = state.time_step
    horizon = 12  # Look ahead 3 hours

    best_action = [0.0] * challenge.num_batteries

    da_prices = challenge.market.day_ahead_prices
    current_da = da_prices[time_step][0]  # Approximate with node 0

    # Calculate average future DA price
    end_step = min(time_step + horizon, challenge.num_steps)
    future_sum = 0.0
    future_count = 0.0
    for t in range(time_step + 1, end_step):
        future_sum += da_prices[t][0]
        future_count += 1.0
    future_avg = future_sum / future_count if future_count > 0.0 else current_da

    # Check future congestion risk
    future_congestion_risk = 0.0
    for t in range(time_step + 1, end_step):
        net_load = sum(challenge.exogenous_injections[t])
        if net_load > 100.0:
            future_congestion_risk += 1.0

    for i in range(challenge.num_batteries):
        min_bound, max_bound = state.action_bounds[i]

        threshold_adjust = future_congestion_risk * 2.0

        if current_da < future_avg - 5.0 - threshold_adjust:
            # Charge
            best_action[i] = min_bound
        elif current_da > future_avg + 5.0 + threshold_adjust:
            # Discharge
            best_action[i] = max_bound
        else:
            best_action[i] = 0.0

    # Enforce feasibility
    return _enforce_flow_feasibility(challenge, state, best_action)
