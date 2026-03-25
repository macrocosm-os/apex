"""Scenario definitions matching the Rust energy_arbitrage::scenarios module."""

from dataclasses import dataclass
from enum import Enum


@dataclass
class ScenarioConfig:
    num_nodes: int
    num_lines: int
    num_batteries: int
    num_steps: int
    gamma_cong: float
    sigma: float
    rho_jump: float
    alpha: float
    heterogeneity: float


class Scenario(Enum):
    BASELINE = "baseline"
    CONGESTED = "congested"
    MULTIDAY = "multiday"
    DENSE = "dense"
    CAPSTONE = "capstone"

    def to_config(self) -> ScenarioConfig:
        return _SCENARIO_CONFIGS[self]

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(s: str) -> "Scenario":
        s = s.lower()
        for scenario in Scenario:
            if scenario.value == s:
                return scenario
        raise ValueError(f"Invalid scenario type: {s}")


_SCENARIO_CONFIGS = {
    Scenario.BASELINE: ScenarioConfig(
        num_nodes=20,
        num_lines=30,
        num_batteries=10,
        num_steps=96,
        gamma_cong=1.00,
        sigma=0.10,
        rho_jump=0.01,
        alpha=4.0,
        heterogeneity=0.2,
    ),
    Scenario.CONGESTED: ScenarioConfig(
        num_nodes=40,
        num_lines=60,
        num_batteries=20,
        num_steps=96,
        gamma_cong=0.80,
        sigma=0.15,
        rho_jump=0.02,
        alpha=3.5,
        heterogeneity=0.4,
    ),
    Scenario.MULTIDAY: ScenarioConfig(
        num_nodes=80,
        num_lines=120,
        num_batteries=40,
        num_steps=192,
        gamma_cong=0.60,
        sigma=0.20,
        rho_jump=0.03,
        alpha=3.0,
        heterogeneity=0.6,
    ),
    Scenario.DENSE: ScenarioConfig(
        num_nodes=100,
        num_lines=200,
        num_batteries=60,
        num_steps=192,
        gamma_cong=0.50,
        sigma=0.25,
        rho_jump=0.04,
        alpha=2.7,
        heterogeneity=0.8,
    ),
    Scenario.CAPSTONE: ScenarioConfig(
        num_nodes=150,
        num_lines=300,
        num_batteries=100,
        num_steps=192,
        gamma_cong=0.40,
        sigma=0.30,
        rho_jump=0.05,
        alpha=2.5,
        heterogeneity=1.0,
    ),
}
