"""PolicyView: restricted view of Challenge data for sandboxed policy execution.

PolicyView exposes only the public information a policy needs to make decisions.
It explicitly excludes: seed, hidden_seed, RT-price generation, market params.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from competition.energy_arbitrage.python import constants
from competition.energy_arbitrage.python.battery import Battery
from competition.energy_arbitrage.python.network import Network

if TYPE_CHECKING:
    from .challenge import State


class MarketView:
    """Read-only market view exposing only day-ahead prices (no RT generation)."""

    __slots__ = ("day_ahead_prices",)

    def __init__(self, day_ahead_prices: list[list[float]]):
        self.day_ahead_prices = day_ahead_prices


class PolicyView:
    """Restricted view of a Challenge instance.

    Provides the same attribute-access patterns as Challenge for legitimate
    solver code (network, batteries, market.day_ahead_prices, etc.) but
    contains no secret material and cannot generate RT prices.
    """

    __slots__ = (
        "num_steps",
        "num_batteries",
        "network",
        "batteries",
        "exogenous_injections",
        "market",
    )

    def __init__(
        self,
        num_steps: int,
        num_batteries: int,
        network: Network,
        batteries: list[Battery],
        exogenous_injections: list[list[float]],
        market: MarketView,
    ):
        self.num_steps = num_steps
        self.num_batteries = num_batteries
        self.network = network
        self.batteries = batteries
        self.exogenous_injections = exogenous_injections
        self.market = market

    def compute_total_injections(self, state: "State", action: list[float]) -> list[float]:
        """Compute total nodal injections (exogenous + batteries) with slack balancing."""
        injections = [0.0] * self.network.num_nodes
        for i in range(self.network.num_nodes):
            if i != self.network.slack_bus:
                injections[i] = state.exogenous_injections[i]
        for battery, a in zip(self.batteries, action):
            injections[battery.node] += a
        total = sum(injections[i] for i in range(self.network.num_nodes) if i != self.network.slack_bus)
        injections[self.network.slack_bus] = -total
        return injections

    def compute_profit(self, state: "State", action: list[float]) -> float:
        """Compute per-step portfolio profit (same formula as Challenge.compute_profit)."""
        dt = constants.DELTA_T
        total_profit = 0.0
        for battery, u in zip(self.batteries, action):
            price = state.rt_prices[battery.node]
            revenue = u * price * dt
            abs_u = abs(u)
            tx_cost = constants.KAPPA_TX * abs_u * dt
            deg_base = (abs_u * dt) / battery.capacity_mwh
            deg_cost = constants.KAPPA_DEG * (deg_base**constants.BETA_DEG)
            total_profit += revenue - tx_cost - deg_cost
        return total_profit

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "num_steps": self.num_steps,
            "num_batteries": self.num_batteries,
            "network": _network_to_dict(self.network),
            "batteries": [_battery_to_dict(b) for b in self.batteries],
            "exogenous_injections": self.exogenous_injections,
            "day_ahead_prices": self.market.day_ahead_prices,
        }

    @staticmethod
    def from_dict(d: dict) -> "PolicyView":
        network = _network_from_dict(d["network"])
        batteries = [_battery_from_dict(b) for b in d["batteries"]]
        market = MarketView(d["day_ahead_prices"])
        return PolicyView(
            num_steps=d["num_steps"],
            num_batteries=d["num_batteries"],
            network=network,
            batteries=batteries,
            exogenous_injections=d["exogenous_injections"],
            market=market,
        )


# ------------------------------------------------------------------
# State serialization (kept here to avoid circular imports)
# ------------------------------------------------------------------


def state_to_dict(state: "State") -> dict:
    return {
        "time_step": state.time_step,
        "socs": state.socs,
        "rt_prices": state.rt_prices,
        "exogenous_injections": state.exogenous_injections,
        "action_bounds": [list(b) for b in state.action_bounds],
        "total_profit": state.total_profit,
    }


def state_from_dict(d: dict) -> "State":
    from .challenge import State

    return State(
        time_step=d["time_step"],
        socs=d["socs"],
        rt_prices=d["rt_prices"],
        exogenous_injections=d["exogenous_injections"],
        action_bounds=[tuple(b) for b in d["action_bounds"]],
        total_profit=d["total_profit"],
    )


# ------------------------------------------------------------------
# Network / Battery serialization helpers
# ------------------------------------------------------------------


def _network_to_dict(net: Network) -> dict:
    return {
        "num_nodes": net.num_nodes,
        "num_lines": net.num_lines,
        "lines": [list(l) for l in net.lines],
        "susceptances": net.susceptances,
        "nominal_flow_limits": net.nominal_flow_limits,
        "flow_limits": net.flow_limits,
        "ptdf": net.ptdf,
        "slack_bus": net.slack_bus,
        "node_incident_lines": net.node_incident_lines,
        "congestion_threshold": net.congestion_threshold,
    }


def _network_from_dict(d: dict) -> Network:
    return Network(
        num_nodes=d["num_nodes"],
        num_lines=d["num_lines"],
        lines=[tuple(l) for l in d["lines"]],
        susceptances=d["susceptances"],
        nominal_flow_limits=d["nominal_flow_limits"],
        flow_limits=d["flow_limits"],
        ptdf=d["ptdf"],
        slack_bus=d["slack_bus"],
        node_incident_lines=d["node_incident_lines"],
        congestion_threshold=d["congestion_threshold"],
    )


def _battery_to_dict(bat: Battery) -> dict:
    return {
        "node": bat.node,
        "capacity_mwh": bat.capacity_mwh,
        "power_charge_mw": bat.power_charge_mw,
        "power_discharge_mw": bat.power_discharge_mw,
        "efficiency_charge": bat.efficiency_charge,
        "efficiency_discharge": bat.efficiency_discharge,
        "soc_min_mwh": bat.soc_min_mwh,
        "soc_max_mwh": bat.soc_max_mwh,
        "soc_initial_mwh": bat.soc_initial_mwh,
    }


def _battery_from_dict(d: dict) -> Battery:
    return Battery(
        node=d["node"],
        capacity_mwh=d["capacity_mwh"],
        power_charge_mw=d["power_charge_mw"],
        power_discharge_mw=d["power_discharge_mw"],
        efficiency_charge=d["efficiency_charge"],
        efficiency_discharge=d["efficiency_discharge"],
        soc_min_mwh=d["soc_min_mwh"],
        soc_max_mwh=d["soc_max_mwh"],
        soc_initial_mwh=d["soc_initial_mwh"],
    )
