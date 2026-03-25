"""Battery model matching the Rust energy_arbitrage::battery module."""

from dataclasses import dataclass
from competition.energy_arbitrage.python import constants


@dataclass
class Battery:
    """Battery physical parameters."""

    node: int
    capacity_mwh: float
    power_charge_mw: float
    power_discharge_mw: float
    efficiency_charge: float
    efficiency_discharge: float
    soc_min_mwh: float
    soc_max_mwh: float
    soc_initial_mwh: float

    @staticmethod
    def generate_instance(rng, num_nodes: int, heterogeneity: float) -> "Battery":
        """Generate a battery instance with given parameters."""
        # Uniform random placement
        node = rng.randint(0, num_nodes - 1)

        # Heterogeneity mechanism: M_b = 3^{h(2r_b - 1)} where r_b ~ U(0,1)
        r = rng.random()
        m_factor = 3.0 ** (heterogeneity * (2.0 * r - 1.0))

        capacity = constants.NOMINAL_CAPACITY * m_factor
        power = constants.NOMINAL_POWER * m_factor

        return Battery(
            node=node,
            capacity_mwh=capacity,
            power_charge_mw=power,
            power_discharge_mw=power,
            efficiency_charge=constants.ETA_CHARGE,
            efficiency_discharge=constants.ETA_DISCHARGE,
            soc_min_mwh=constants.E_MIN_FRAC * capacity,
            soc_max_mwh=constants.E_MAX_FRAC * capacity,
            soc_initial_mwh=constants.E_INIT_FRAC * capacity,
        )

    def compute_action_bounds(self, soc: float) -> tuple[float, float]:
        """Return feasible signed action bounds (u_min, u_max).

        u < 0: charge, u > 0: discharge.
        """
        dt = constants.DELTA_T

        headroom = max(self.soc_max_mwh - soc, 0.0)
        available = max(soc - self.soc_min_mwh, 0.0)

        if self.efficiency_charge > 0.0:
            max_charge_from_soc = headroom / (self.efficiency_charge * dt)
        else:
            max_charge_from_soc = 0.0

        if self.efficiency_discharge > 0.0:
            max_discharge_from_soc = available * self.efficiency_discharge / dt
        else:
            max_discharge_from_soc = 0.0

        max_charge = max(min(max_charge_from_soc, self.power_charge_mw), 0.0)
        max_discharge = max(min(max_discharge_from_soc, self.power_discharge_mw), 0.0)

        return (-max_charge, max_discharge)

    def apply_action_to_soc(self, action: float, soc: float) -> float:
        """Apply action to SOC and return new SOC.

        E_{t+1} = E_t + eta_c * c * dt - d * dt / eta_d
        """
        c = max(-action, 0.0)  # charge if negative
        d = max(action, 0.0)  # discharge if positive
        dt = constants.DELTA_T

        new_soc = soc + self.efficiency_charge * c * dt - d * dt / self.efficiency_discharge
        return max(self.soc_min_mwh, min(new_soc, self.soc_max_mwh))
