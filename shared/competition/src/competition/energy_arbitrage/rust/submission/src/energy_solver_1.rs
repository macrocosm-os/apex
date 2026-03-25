use eval::energy_arbitrage::{constants, Challenge, Solution, State};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const CHARGE_THRESHOLD: f64 = 0.95;
const DISCHARGE_THRESHOLD: f64 = 1.05;
const MAX_FLOW_ADJUST_ITERS: usize = 64;
const GLOBAL_SCALE_BSEARCH_ITERS: usize = 32;
const EPS: f64 = 1e-12;

#[derive(Clone, Copy)]
struct Violation {
    line: usize,
    flow: f64,
    amount: f64,
}

fn compute_flows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let injections = challenge.compute_total_injections(state, action);
    (0..challenge.network.num_lines)
        .map(|l| {
            (0..challenge.network.num_nodes)
                .map(|k| challenge.network.ptdf[l][k] * injections[k])
                .sum::<f64>()
        })
        .collect()
}

fn most_violated_line(challenge: &Challenge, flows: &[f64]) -> Option<Violation> {
    let mut best: Option<Violation> = None;
    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        let violation = flow.abs() - limit;
        if violation > constants::EPS_FLOW * limit {
            let candidate = Violation {
                line: l,
                flow,
                amount: violation,
            };
            match best {
                Some(current) if candidate.amount <= current.amount => {}
                _ => best = Some(candidate),
            }
        }
    }
    best
}

fn is_flow_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = compute_flows(challenge, state, action);
    most_violated_line(challenge, &flows).is_none()
}

fn soften_most_violated_line(
    challenge: &Challenge,
    violation: Violation,
    action: &mut [f64],
) -> bool {
    let line = violation.line;
    let signed_direction = violation.flow.signum();
    if signed_direction.abs() <= EPS {
        return false;
    }

    let mut worsening_indices = Vec::new();
    let mut worsening_strength = 0.0;
    for (i, battery) in challenge.batteries.iter().enumerate() {
        let contribution = challenge.network.ptdf[line][battery.node] * action[i];
        let signed_contribution = signed_direction * contribution;
        if signed_contribution > EPS {
            worsening_strength += signed_contribution;
            worsening_indices.push(i);
        }
    }

    if worsening_indices.is_empty() || worsening_strength <= EPS {
        return false;
    }

    let keep = (1.0 - violation.amount / worsening_strength).clamp(0.0, 1.0);
    if (1.0 - keep).abs() <= EPS {
        return false;
    }
    for i in worsening_indices {
        action[i] *= keep;
    }
    true
}

/// Scale actions by the largest α ∈ [0, 1] such that
/// total_profit + P(α) >= 0, where P(α) = α·A − α²·B.
fn enforce_profit_floor(
    challenge: &Challenge,
    state: &State,
    mut action: Vec<f64>,
) -> Vec<f64> {
    let dt = constants::DELTA_T;
    let mut a_coeff = 0.0;
    let mut b_coeff = 0.0;
    for (i, battery) in challenge.batteries.iter().enumerate() {
        let u = action[i];
        let price = state.rt_prices[battery.node];
        a_coeff += u * price * dt - constants::KAPPA_TX * u.abs() * dt;
        let deg_base = u.abs() * dt / battery.capacity_mwh;
        b_coeff += constants::KAPPA_DEG * deg_base * deg_base;
    }

    let step_profit = a_coeff - b_coeff;
    if state.total_profit + step_profit >= 0.0 {
        return action;
    }

    let alpha = if b_coeff > EPS {
        let disc = a_coeff * a_coeff + 4.0 * b_coeff * state.total_profit;
        if disc < 0.0 {
            0.0
        } else {
            ((a_coeff + disc.sqrt()) / (2.0 * b_coeff)).clamp(0.0, 1.0)
        }
    } else if a_coeff < -EPS {
        (state.total_profit / a_coeff.abs()).clamp(0.0, 1.0)
    } else {
        1.0
    };

    for u in action.iter_mut() {
        *u *= alpha;
    }
    action
}

fn enforce_flow_feasibility(
    challenge: &Challenge,
    state: &State,
    mut action: Vec<f64>,
) -> Result<Vec<f64>> {
    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let flows = compute_flows(challenge, state, &action);
        let Some(violation) = most_violated_line(challenge, &flows) else {
            return Ok(action);
        };
        if !soften_most_violated_line(challenge, violation, &mut action) {
            break;
        }
    }

    if is_flow_feasible(challenge, state, &action) {
        return Ok(action);
    }

    let zero = vec![0.0; action.len()];
    if !is_flow_feasible(challenge, state, &zero) {
        return Err(anyhow!(
            "Baseline fallback failed: grid is infeasible even with zero battery actions"
        ));
    }

    let base = action;
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..GLOBAL_SCALE_BSEARCH_ITERS {
        let mid = 0.5 * (low + high);
        let scaled: Vec<f64> = base.iter().map(|u| mid * u).collect();
        if is_flow_feasible(challenge, state, &scaled) {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(base.into_iter().map(|u| low * u).collect())
}

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    // Optionally define hyperparameters here. Example:
    // pub param1: usize,
    // pub param2: f64,
}

pub fn help() {
    println!("No help information provided.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let solution = challenge.grid_optimize(&policy)?;
    save_solution(&solution)?;
    Ok(())
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    if state.time_step >= challenge.market.day_ahead_prices.len() {
        return Err(anyhow!(
            "Missing day-ahead prices for time_step {}",
            state.time_step
        ));
    }
    let da_prices = &challenge.market.day_ahead_prices[state.time_step];
    if da_prices.len() != challenge.network.num_nodes {
        return Err(anyhow!(
            "Day-ahead prices length ({}) does not match network nodes ({})",
            da_prices.len(),
            challenge.network.num_nodes
        ));
    }

    let avg_da = da_prices.iter().sum::<f64>() / da_prices.len() as f64;
    let mut action = vec![0.0; challenge.num_batteries];

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node_price = da_prices[battery.node];
        let (min_bound, max_bound) = state.action_bounds[i];
        let can_full_charge = min_bound <= -battery.power_charge_mw + constants::EPS_SOC;
        let can_full_discharge = max_bound >= battery.power_discharge_mw - constants::EPS_SOC;

        action[i] = if node_price < CHARGE_THRESHOLD * avg_da && can_full_charge {
            -battery.power_charge_mw
        } else if node_price > DISCHARGE_THRESHOLD * avg_da && can_full_discharge {
            battery.power_discharge_mw
        } else {
            0.0
        };

        action[i] = action[i].clamp(min_bound, max_bound);
    }

    let action = enforce_flow_feasibility(challenge, state, action)?;
    Ok(enforce_profit_floor(challenge, state, action))
}
