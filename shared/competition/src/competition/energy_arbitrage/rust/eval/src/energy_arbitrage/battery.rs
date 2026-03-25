use super::constants;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Battery physical parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Battery {
    /// Battery placement
    pub node: usize,
    /// Energy capacity (MWh)
    pub capacity_mwh: f64,
    /// Maximum charge power (MW)
    pub power_charge_mw: f64,
    /// Maximum discharge power (MW)
    pub power_discharge_mw: f64,
    /// Charge efficiency (η^c)
    pub efficiency_charge: f64,
    /// Discharge efficiency (η^d)
    pub efficiency_discharge: f64,
    /// SOC lower bound (MWh)
    pub soc_min_mwh: f64,
    /// SOC upper bound (MWh)
    pub soc_max_mwh: f64,
    /// Initial SOC (MWh)
    pub soc_initial_mwh: f64,
}

impl Battery {
    /// Generate a battery instance with given parameters
    pub(crate) fn generate_instance(
        rng: &mut impl Rng,
        num_nodes: usize,
        heterogeneity: f64,
    ) -> Self {
        // Uniform random placement
        let node = rng.gen_range(0..num_nodes);

        // Heterogeneity mechanism: M_b = 3^{h(2r_b - 1)} where r_b ~ U(0,1)
        let r: f64 = rng.r#gen();
        let m_factor = 3.0_f64.powf(heterogeneity * (2.0 * r - 1.0));

        let capacity = constants::NOMINAL_CAPACITY * m_factor;
        let power = constants::NOMINAL_POWER * m_factor;

        Self {
            node,
            capacity_mwh: capacity,
            power_charge_mw: power,
            power_discharge_mw: power,
            efficiency_charge: constants::ETA_CHARGE,
            efficiency_discharge: constants::ETA_DISCHARGE,
            soc_min_mwh: constants::E_MIN_FRAC * capacity,
            soc_max_mwh: constants::E_MAX_FRAC * capacity,
            soc_initial_mwh: constants::E_INIT_FRAC * capacity,
        }
    }

    /// Return feasible signed action bounds `(u_min, u_max)` for one battery.
    ///
    /// Bounds are computed from:
    /// 1. Nameplate charge/discharge power limits
    /// 2. SOC headroom/availability for the given `soc`
    ///
    /// Sign convention:
    /// - `u < 0`: charge
    /// - `u > 0`: discharge
    ///
    /// This helper does not account for network flow coupling across batteries.
    pub(crate) fn compute_action_bounds(&self, soc: f64) -> (f64, f64) {
        let dt = constants::DELTA_T;

        let headroom = (self.soc_max_mwh - soc).max(0.0);
        let available = (soc - self.soc_min_mwh).max(0.0);

        let max_charge_from_soc = if self.efficiency_charge > 0.0 {
            headroom / (self.efficiency_charge * dt)
        } else {
            0.0
        };
        let max_discharge_from_soc = if self.efficiency_discharge > 0.0 {
            available * self.efficiency_discharge / dt
        } else {
            0.0
        };

        let max_charge = max_charge_from_soc.min(self.power_charge_mw).max(0.0);
        let max_discharge = max_discharge_from_soc.min(self.power_discharge_mw).max(0.0);

        (-max_charge, max_discharge)
    }

    /// Apply action to SOC and return new SOC
    /// E_{t+1} = E_t + η^c * c * Δt - d * Δt / η^d
    pub fn apply_action_to_soc(&self, action: f64, soc: f64) -> f64 {
        let c = (-action).max(0.0); // charge if negative
        let d = action.max(0.0); // discharge if positive
        let dt = constants::DELTA_T;

        let new_soc = soc + self.efficiency_charge * c * dt - d * dt / self.efficiency_discharge;

        new_soc.clamp(self.soc_min_mwh, self.soc_max_mwh)
    }
}
