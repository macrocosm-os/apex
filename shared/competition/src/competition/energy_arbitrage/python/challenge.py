"""Main Challenge module matching the Rust energy_arbitrage mod.rs."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from competition.energy_arbitrage.python import constants
from competition.energy_arbitrage.python.battery import Battery
from competition.energy_arbitrage.python.market import Market, MarketParams
from competition.energy_arbitrage.python.network import Network
from competition.energy_arbitrage.python.scenarios import Scenario, ScenarioConfig
from competition.energy_arbitrage.python.greedy import policy as greedy_policy
from competition.energy_arbitrage.python.conservative import policy as conservative_policy
from competition.energy_arbitrage.python.policy_view import MarketView, PolicyView


@dataclass
class Track:
    s: Scenario


@dataclass
class Solution:
    schedule: list[list[float]] = field(default_factory=list)


@dataclass
class State:
    """Simulation state visible to innovators."""

    time_step: int
    socs: list[float]
    rt_prices: list[float]
    exogenous_injections: list[float]
    action_bounds: list[tuple[float, float]]
    total_profit: float


class NextRTPrices:
    """Union type for next RT prices: either an override list or a seed to generate from."""

    pass


class NextRTPricesOverride(NextRTPrices):
    def __init__(self, prices: list[float]):
        self.prices = prices


class NextRTPricesGenerate(NextRTPrices):
    def __init__(self, seed: bytes):
        self.seed = seed


# Type alias for policy function — receives a PolicyView, NOT the full Challenge.
PolicyFn = Callable[[PolicyView, State], list[float]]


class Challenge:
    """Energy arbitrage challenge instance."""

    def __init__(
        self,
        seed: bytes,
        hidden_seed: bytes,
        num_steps: int,
        num_batteries: int,
        network: Network,
        batteries: list[Battery],
        exogenous_injections: list[list[float]],
        market: Market,
    ):
        self.seed = seed
        self._hidden_seed = hidden_seed
        self.num_steps = num_steps
        self.num_batteries = num_batteries
        self.network = network
        self.batteries = batteries
        self.exogenous_injections = exogenous_injections
        self.market = market

    @staticmethod
    def generate_instance(seed: bytes, track: Track) -> "Challenge":
        """Generate a challenge instance from seed and track."""
        rng = random.Random()
        rng.seed(seed)

        config: ScenarioConfig = track.s.to_config()

        network = Network.generate_instance(rng, config.num_nodes, config.num_lines, config.gamma_cong)
        batteries = [
            Battery.generate_instance(rng, config.num_nodes, config.heterogeneity) for _ in range(config.num_batteries)
        ]
        exogenous_injections = network.generate_exogenous_injections(rng, config.num_steps)
        market = Market.generate_instance(
            rng,
            MarketParams(
                volatility=config.sigma,
                jump_probability=config.rho_jump,
                tail_index=config.alpha,
            ),
            config.num_nodes,
            config.num_steps,
        )

        # Generate seed and hidden_seed from rng (matches Rust: seed = rng.gen(), hidden_seed = rng.gen())
        generated_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        hidden_seed = bytes([rng.randint(0, 255) for _ in range(32)])

        return Challenge(
            seed=generated_seed,
            hidden_seed=hidden_seed,
            num_steps=config.num_steps,
            num_batteries=config.num_batteries,
            network=network,
            batteries=batteries,
            exogenous_injections=exogenous_injections,
            market=market,
        )

    def to_policy_view(self) -> PolicyView:
        """Create a PolicyView containing only public challenge data."""
        return PolicyView(
            num_steps=self.num_steps,
            num_batteries=self.num_batteries,
            network=self.network,
            batteries=self.batteries,
            exogenous_injections=self.exogenous_injections,
            market=MarketView(self.market.day_ahead_prices),
        )

    def _initial_state(self, rng) -> State:
        congestion = [False] * self.network.num_nodes
        return State(
            time_step=0,
            socs=[b.soc_initial_mwh for b in self.batteries],
            rt_prices=self.market.generate_rt_prices(rng, 0, congestion),
            exogenous_injections=list(self.exogenous_injections[0]),
            action_bounds=[b.compute_action_bounds(b.soc_initial_mwh) for b in self.batteries],
            total_profit=0.0,
        )

    def compute_total_injections(self, state: State, action: list[float]) -> list[float]:
        """Compute total nodal injections (exogenous + batteries) with slack balancing."""
        injections = [0.0] * self.network.num_nodes

        # Add exogenous injections
        for i in range(self.network.num_nodes):
            if i != self.network.slack_bus:
                injections[i] = state.exogenous_injections[i]

        # Add storage injections
        for battery, a in zip(self.batteries, action):
            injections[battery.node] += a

        # Slack bus balances the system
        total = sum(injections[i] for i in range(self.network.num_nodes) if i != self.network.slack_bus)
        injections[self.network.slack_bus] = -total

        return injections

    def compute_profit(self, state: State, action: list[float]) -> float:
        """Compute per-step portfolio profit per spec equation (3.7)."""
        transaction_cost_per_mwh = constants.KAPPA_TX
        degradation_scale = constants.KAPPA_DEG
        degradation_exponent = constants.BETA_DEG
        dt = constants.DELTA_T
        total_profit = 0.0

        for battery, u in zip(self.batteries, action):
            price = state.rt_prices[battery.node]

            # Revenue: u * lambda * dt
            revenue = u * price * dt

            # Friction: phi_b(u) = kappa_tx * |u| * dt + kappa_deg * (|u| * dt / E_bar_b)^beta
            abs_u = abs(u)
            tx_cost = transaction_cost_per_mwh * abs_u * dt
            deg_base = (abs_u * dt) / battery.capacity_mwh
            deg_cost = degradation_scale * (deg_base**degradation_exponent)

            total_profit += revenue - tx_cost - deg_cost

        return total_profit

    def take_step(self, state: State, action: list[float], next_rt_prices: NextRTPrices) -> State:
        """Simulate one time step. Raises ValueError on constraint violation."""
        if len(action) != len(self.batteries):
            raise ValueError(
                f"Action length ({len(action)}) does not match number of batteries ({len(self.batteries)})"
            )

        for i, (a, bounds) in enumerate(zip(action, state.action_bounds)):
            if a < bounds[0] or a > bounds[1]:
                raise ValueError(f"Action ({a}) on battery {i} is out of bounds ({bounds[0]}, {bounds[1]})")

        injections = self.compute_total_injections(state, action)
        flows = self.network.compute_flows(injections)
        self.network.verify_flows(flows)

        next_time_step = state.time_step + 1
        next_total_profit = state.total_profit + self.compute_profit(state, action)

        if next_time_step < self.num_steps:
            if isinstance(next_rt_prices, NextRTPricesOverride):
                if len(next_rt_prices.prices) != self.network.num_nodes:
                    raise ValueError(
                        f"Override RT prices length ({len(next_rt_prices.prices)}) "
                        f"does not match number of nodes ({self.network.num_nodes})"
                    )
                rt_prices = next_rt_prices.prices
            elif isinstance(next_rt_prices, NextRTPricesGenerate):
                rng = random.Random()
                rng.seed(next_rt_prices.seed)
                congestion = self.network.generate_congestion_indicators(rng, state.exogenous_injections)
                rt_prices = self.market.generate_rt_prices(rng, next_time_step, congestion)
            else:
                raise TypeError(f"Unknown NextRTPrices type: {type(next_rt_prices)}")

            next_exogenous = list(self.exogenous_injections[next_time_step])
            next_socs = [self.batteries[i].apply_action_to_soc(action[i], state.socs[i]) for i in range(len(action))]
            next_action_bounds = [self.batteries[i].compute_action_bounds(next_socs[i]) for i in range(len(next_socs))]

            return State(
                time_step=next_time_step,
                socs=next_socs,
                rt_prices=rt_prices,
                exogenous_injections=next_exogenous,
                action_bounds=next_action_bounds,
                total_profit=next_total_profit,
            )
        else:
            return State(
                time_step=next_time_step,
                socs=[],
                rt_prices=[],
                exogenous_injections=[],
                action_bounds=[],
                total_profit=next_total_profit,
            )

    def _simulate(self, policy: PolicyFn) -> tuple[list[list[float]], State]:
        """Run the full rollout loop.

        The policy receives a PolicyView (not the Challenge itself) so it
        cannot access seeds or RT-price generation.
        """
        view = self.to_policy_view()
        rng = random.Random()
        rng.seed(self._hidden_seed)
        state = self._initial_state(rng)
        schedule: list[list[float]] = []

        for _ in range(self.num_steps):
            action = policy(view, state)
            next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
            state = self.take_step(state, action, NextRTPricesGenerate(next_seed))
            schedule.append(action)

        return schedule, state

    def grid_optimize(self, policy: PolicyFn) -> Solution:
        """Run the full rollout loop and return the solution."""
        schedule, _ = self._simulate(policy)
        return Solution(schedule=schedule)

    def evaluate_total_profit(self, solution: Solution) -> float:
        """Evaluate the total profit of a given solution."""

        def replay_policy(challenge: "Challenge", state: State) -> list[float]:
            return list(solution.schedule[state.time_step])

        _, final_state = self._simulate(replay_policy)
        return final_state.total_profit

    def compute_baseline(self) -> tuple[Solution, float]:
        """Compute the best baseline solution (greedy vs conservative)."""
        greedy_schedule, greedy_state = self._simulate(greedy_policy)
        greedy_total_profit = greedy_state.total_profit
        conservative_schedule, conservative_state = self._simulate(conservative_policy)
        conservative_total_profit = conservative_state.total_profit
        print(f"Greedy total profit: {greedy_total_profit}, Conservative total profit: {conservative_total_profit}")
        if greedy_total_profit > conservative_total_profit:
            return Solution(schedule=greedy_schedule), greedy_total_profit
        else:
            return Solution(schedule=conservative_schedule), conservative_total_profit

    def grid_optimize_sandboxed(self, policy_module: str) -> Solution:
        """Run the policy in an isolated subprocess via SandboxedEvaluator.

        The policy never has access to the Challenge object, seeds, or
        RT-price generation machinery.

        Args:
            policy_module: Fully-qualified module name containing a ``policy``
                function, e.g. ``"competition.energy_arbitrage.python.energy_solver_1"``.
        """
        from .sandbox import SandboxedEvaluator

        evaluator = SandboxedEvaluator()
        return evaluator.run(self, policy_module)

    def evaluate_solution(self, solution: Solution) -> int:
        """Evaluate a solution and return quality score."""
        QUALITY_PRECISION = 1_000_000  # match crate::QUALITY_PRECISION
        total_profit = self.evaluate_total_profit(solution)
        _, baseline_total_profit = self.compute_baseline()
        quality = (total_profit - baseline_total_profit) / (baseline_total_profit + 1e-6)
        quality = max(-10.0, min(quality, 10.0)) * QUALITY_PRECISION
        return round(quality)
