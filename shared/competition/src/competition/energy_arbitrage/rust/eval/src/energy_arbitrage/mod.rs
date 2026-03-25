mod baselines;
mod battery;
pub use battery::*;
pub mod constants;
mod market;
pub use market::*;
mod network;
pub use network::*;
mod scenarios;
pub use scenarios::*;
mod utils;

use crate::QUALITY_PRECISION;
use anyhow::{anyhow, Result};
use battery::*;
use market::*;
use network::*;
use rand::{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use scenarios::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(not(feature = "hide_verification"))]
pub static CALLED_GRID_OPTIMIZE: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "hide_verification")]
static CALLED_GRID_OPTIMIZE: AtomicBool = AtomicBool::new(true);

impl_kv_string_serde! {
    Track {
        s: Scenario
    }
}

impl_base64_serde! {
    Solution {
        schedule: Vec<Vec<f64>>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self {
            schedule: Vec::new(),
        }
    }
}

/// Simulation state visible to innovators.
/// Contains only what an algorithm needs to make its next decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// Current time step index (0-based)
    pub time_step: usize,
    /// Current state-of-charge for each battery (MWh)
    pub socs: Vec<f64>,
    /// Real-time nodal prices at the current time step ($/MWh)
    pub rt_prices: Vec<f64>,
    /// Exogenous nodal injections at the current time step (MWh)
    pub exogenous_injections: Vec<f64>,
    /// Action bounds for each battery (MWh)
    pub action_bounds: Vec<(f64, f64)>,
    /// Total profit from arbitrage until the current time step ($)
    pub total_profit: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    hidden_seed: [u8; 32],
    pub num_steps: usize,
    pub num_batteries: usize,
    pub network: Network,
    pub batteries: Vec<Battery>,
    pub exogenous_injections: Vec<Vec<f64>>,
    pub market: Market,
}

pub enum NextRTPrices {
    Override(Vec<f64>),
    Generate([u8; 32]),
}

impl Challenge {
    /// Generate a connected network with given parameters
    pub fn generate_instance(seed: &[u8; 32], track: &Track) -> Result<Self> {
        let mut rng = SmallRng::from_seed(StdRng::from_seed(seed.clone()).r#gen());
        let ScenarioConfig {
            num_nodes,
            num_lines,
            num_batteries,
            num_steps,
            gamma_cong,
            sigma,
            rho_jump,
            alpha,
            heterogeneity,
        } = track.s.clone().into();

        let network = Network::generate_instance(&mut rng, num_nodes, num_lines, gamma_cong);
        let batteries = (0..num_batteries)
            .map(|_| Battery::generate_instance(&mut rng, num_nodes, heterogeneity))
            .collect();
        let exogenous_injections = network.generate_exogenous_injections(&mut rng, num_steps);
        let market = Market::generate_instance(
            &mut rng,
            MarketParams {
                volatility: sigma,
                jump_probability: rho_jump,
                tail_index: alpha,
            },
            num_nodes,
            num_steps,
        );
        Ok(Self {
            seed: rng.r#gen(),
            hidden_seed: rng.r#gen(),
            num_steps,
            num_batteries,
            network,
            batteries,
            exogenous_injections,
            market,
        })
    }

    fn initial_state(&self, rng: &mut impl Rng) -> State {
        let congestion: Vec<bool> = vec![false; self.network.num_nodes];
        State {
            time_step: 0,
            socs: self.batteries.iter().map(|b| b.soc_initial_mwh).collect(),
            rt_prices: self.market.generate_rt_prices(rng, 0, &congestion),
            exogenous_injections: self.exogenous_injections[0].clone(),
            action_bounds: self
                .batteries
                .iter()
                .map(|b| b.compute_action_bounds(b.soc_initial_mwh))
                .collect(),
            total_profit: 0.0,
        }
    }

    /// Compute total nodal injections (exogenous + batteries) with slack balancing
    pub fn compute_total_injections(&self, state: &State, action: &[f64]) -> Vec<f64> {
        let mut injections = vec![0.0; self.network.num_nodes];

        // Add exogenous injections
        for i in 0..self.network.num_nodes {
            if i != self.network.slack_bus {
                injections[i] = state.exogenous_injections[i];
            }
        }

        // Add storage injections
        for (battery, action) in self.batteries.iter().zip(action.iter()) {
            injections[battery.node] += action;
        }

        // Slack bus balances the system
        let mut sum = 0.0;
        for i in 0..self.network.num_nodes {
            if i != self.network.slack_bus {
                sum += injections[i];
            }
        }
        injections[self.network.slack_bus] = -sum;

        injections
    }

    /// Compute per-step portfolio profit per spec equation (3.7)
    pub fn compute_profit(&self, state: &State, action: &[f64]) -> f64 {
        let transaction_cost_per_mwh = constants::KAPPA_TX; // Transaction cost ($/MWh) - κ_tx
        let degradation_scale = constants::KAPPA_DEG; // Degradation scale ($) - κ_deg
        let degradation_exponent = constants::BETA_DEG; // Degradation exponent - β
        let dt = constants::DELTA_T;
        let mut total_profit = 0.0;

        for (battery, &u) in self.batteries.iter().zip(action.iter()) {
            if u == 0.0 {
                continue;
            }
            let price = state.rt_prices[battery.node];

            // Revenue: u * λ * Δt
            let revenue = u * price * dt;

            // Friction: φ_b(u) = κ_tx|u|Δt + κ_deg(|u|Δt/E̅_b)^β
            let abs_u = u.abs();
            let tx_cost = transaction_cost_per_mwh * abs_u * dt;
            let deg_base = (abs_u * dt) / battery.capacity_mwh;
            let deg_cost = degradation_scale * deg_base.powf(degradation_exponent);

            total_profit += revenue - tx_cost - deg_cost;
        }

        total_profit
    }

    /// Simulate one time step without commitment (for innovator use).
    ///
    /// Validates the action, applies it to the state, and returns the next state
    /// using the provided RT prices. No seed advancement or commitment chain update.
    ///
    /// `rt_prices_override` is required — innovators must provide their own
    /// price forecast/scenario for the next step.
    ///
    /// Returns `Err` if the action violates any constraint.
    pub fn take_step(
        &self,
        state: &State,
        action: &[f64],
        next_rt_prices: NextRTPrices,
    ) -> Result<State> {
        if action.len() != self.batteries.len() {
            return Err(anyhow!(
                "Action length ({}) does not match number of batteries ({})",
                action.len(),
                self.batteries.len()
            ));
        }
        for (i, (&a, bounds)) in action.iter().zip(state.action_bounds.iter()).enumerate() {
            if a < bounds.0 || a > bounds.1 {
                return Err(anyhow!(
                    "Action ({}) on battery {} is out of bounds ({}, {})",
                    a,
                    i,
                    bounds.0,
                    bounds.1
                ));
            }
        }
        let injections = self.compute_total_injections(state, &action);
        let flows = self.network.compute_flows(&injections);
        self.network.verify_flows(&flows)?;
        let next_time_step = state.time_step + 1;
        let next_total_profit = state.total_profit + self.compute_profit(state, action);
        Ok(if next_time_step < self.num_steps {
            let next_rt_prices = match next_rt_prices {
                NextRTPrices::Override(prices) => {
                    if prices.len() != self.network.num_nodes {
                        return Err(anyhow!(
                            "Override RT prices length ({}) does not match number of nodes ({})",
                            prices.len(),
                            self.network.num_nodes
                        ));
                    }
                    prices
                }
                NextRTPrices::Generate(seed) => {
                    let mut rng = SmallRng::from_seed(seed);
                    let congestion = self
                        .network
                        .generate_congestion_indicators(&mut rng, &state.exogenous_injections);
                    self.market
                        .generate_rt_prices(&mut rng, next_time_step, &congestion)
                }
            };
            let next_exogenous_injections = self.exogenous_injections[next_time_step].clone();
            let next_socs: Vec<f64> = action
                .iter()
                .enumerate()
                .map(|(i, &a)| self.batteries[i].apply_action_to_soc(a, state.socs[i]))
                .collect();
            let next_action_bounds: Vec<(f64, f64)> = next_socs
                .iter()
                .enumerate()
                .map(|(i, &soc)| self.batteries[i].compute_action_bounds(soc))
                .collect();

            State {
                time_step: next_time_step,
                socs: next_socs,
                rt_prices: next_rt_prices,
                exogenous_injections: next_exogenous_injections,
                action_bounds: next_action_bounds,
                total_profit: next_total_profit,
            }
        } else {
            State {
                time_step: next_time_step,
                socs: vec![],
                rt_prices: vec![],
                exogenous_injections: vec![],
                action_bounds: vec![],
                total_profit: next_total_profit,
            }
        })
    }

    /// Run the full rollout loop with commitment chain.
    ///
    /// Calls `policy` at each step to produce actions. If any action violates
    /// constraints, the error propagates immediately (terminates the rollout).
    ///
    /// The commitment chain is handled internally — innovators cannot access or bypass it.
    fn simulate(
        &self,
        policy: &dyn Fn(&Challenge, &State) -> Result<Vec<f64>>,
    ) -> Result<(Vec<Vec<f64>>, State)> {
        let mut rng = SmallRng::from_seed(StdRng::from_seed(self.hidden_seed.clone()).r#gen());
        let mut state = self.initial_state(&mut rng);
        let mut schedule = Vec::with_capacity(self.num_steps);

        for _ in 0..self.num_steps {
            let action = policy(self, &state)?;
            state = self.take_step(&state, &action, NextRTPrices::Generate(rng.r#gen()))?;
            schedule.push(action);
        }

        Ok((schedule, state))
    }

    /// Run the full rollout loop with commitment chain and return the solution.
    pub fn grid_optimize(
        &self,
        policy: &dyn Fn(&Challenge, &State) -> Result<Vec<f64>>,
    ) -> Result<Solution> {
        // Attempt to flip it from false to true
        let already_set = CALLED_GRID_OPTIMIZE.swap(true, Ordering::SeqCst);
        if already_set {
            return Err(anyhow::anyhow!("Can only call grid_optimize once"));
        }

        let (schedule, _) = self.simulate(policy)?;
        Ok(Solution { schedule })
    }

    conditional_pub!(
        fn evaluate_total_profit(&self, solution: &Solution) -> Result<f64> {
            let (_, state) =
                self.simulate(&|_, state: &State| Ok(solution.schedule[state.time_step].clone()))?;
            Ok(state.total_profit)
        }
    );

    conditional_pub!(
        fn compute_baseline(&self) -> Result<(Solution, f64)> {
            let (greedy_schedule, state) = self.simulate(&baselines::greedy::policy)?;
            let greedy_total_profit = state.total_profit;
            let (conservative_schedule, state) = self.simulate(&baselines::conservative::policy)?;
            let conservative_total_profit = state.total_profit;
            println!(
                "Greedy total profit: {}, Conservative total profit: {}",
                greedy_total_profit, conservative_total_profit
            );
            if greedy_total_profit > conservative_total_profit {
                Ok((
                    Solution {
                        schedule: greedy_schedule,
                    },
                    greedy_total_profit,
                ))
            } else {
                Ok((
                    Solution {
                        schedule: conservative_schedule,
                    },
                    conservative_total_profit,
                ))
            }
        }
    );

    conditional_pub!(
        fn evaluate_solution(&self, solution: &Solution) -> Result<i32> {
            let total_profit = self.evaluate_total_profit(solution)?;
            let (_, baseline_total_profit) = self.compute_baseline()?;
            let quality = (total_profit as f64 - baseline_total_profit as f64)
                / (baseline_total_profit as f64 + 1e-6);
            let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
            let quality = quality.round() as i32;
            Ok(quality)
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn challenge_iter() -> impl Iterator<Item = Challenge> {
        (0..5).flat_map(|i| {
            let seed = [i as u8; 32];
            [
                Scenario::BASELINE,
                Scenario::CONGESTED,
                Scenario::MULTIDAY,
                Scenario::DENSE,
                Scenario::CAPSTONE,
            ]
            .into_iter()
            .map(move |s| Challenge::generate_instance(&seed.clone(), &Track { s }).unwrap())
        })
    }

    #[test]
    fn test_zero_profit() {
        fn policy(_: &Challenge, s: &State) -> Result<Vec<f64>> {
            Ok(vec![0.0; s.action_bounds.len()])
        }
        for challenge in challenge_iter() {
            let (schedule, final_state) = challenge.simulate(&policy).unwrap();
            let solution = Solution { schedule };
            let total_profit = challenge.evaluate_total_profit(&solution).unwrap();
            assert_eq!(total_profit, 0.0);
            assert_eq!(total_profit, final_state.total_profit);
        }
    }

    #[test]
    fn test_non_zero_profit() {
        fn policy(_: &Challenge, s: &State) -> Result<Vec<f64>> {
            Ok(vec![0.00001; s.action_bounds.len()])
        }
        for challenge in challenge_iter() {
            let (schedule, final_state) = challenge.simulate(&policy).unwrap();
            let solution = Solution { schedule };
            let total_profit = challenge.evaluate_total_profit(&solution).unwrap();
            assert!(total_profit > 0.0);
            assert_eq!(total_profit, final_state.total_profit);
        }
    }

    #[test]
    fn test_faulty_policy() {
        fn policy(_: &Challenge, _: &State) -> Result<Vec<f64>> {
            Err(anyhow!("Faulty policy"))
        }
        for challenge in challenge_iter() {
            CALLED_GRID_OPTIMIZE.store(false, Ordering::SeqCst);
            let result = challenge.grid_optimize(&policy);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_greedy_baseline() {
        for challenge in challenge_iter() {
            let result = challenge.compute_baseline();
            assert!(result.is_ok());
            let (_, total_profit) = result.unwrap();
            assert!(total_profit > 0.0);
        }
    }

    #[test]
    fn test_policy_independent_rt_prices() {
        for challenge in challenge_iter() {
            let mut rng =
                SmallRng::from_seed(StdRng::from_seed(challenge.hidden_seed.clone()).r#gen());
            let seeds = (0..challenge.num_steps)
                .map(|_| rng.r#gen())
                .collect::<Vec<[u8; 32]>>();
            let initial_state = challenge.initial_state(&mut rng);

            let mut rt_prices1 = Vec::with_capacity(challenge.num_steps);
            let mut state = initial_state.clone();
            for s in seeds.iter() {
                let action = baselines::greedy::policy(&challenge, &state).unwrap();
                state = challenge
                    .take_step(&state, &action, NextRTPrices::Generate(s.clone()))
                    .unwrap();
                rt_prices1.push(state.rt_prices.clone());
            }

            let mut rt_prices2 = Vec::with_capacity(challenge.num_steps);
            let mut state = initial_state.clone();
            for s in seeds.iter() {
                let action = baselines::conservative::policy(&challenge, &state).unwrap();
                state = challenge
                    .take_step(&state, &action, NextRTPrices::Generate(s.clone()))
                    .unwrap();
                rt_prices2.push(state.rt_prices.clone());
            }
            assert_eq!(rt_prices1, rt_prices2);
        }
    }
}
