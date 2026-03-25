use crate::energy_arbitrage::{constants, Challenge, State};
use anyhow::{anyhow, Result};

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

/// Scale actions to maintain non-negative total profit
fn enforce_profit_floor(challenge: &Challenge, state: &State, mut action: Vec<f64>) -> Vec<f64> {
    let mut profit = challenge.compute_profit(state, &action);
    if state.total_profit + profit >= 0.0 {
        return action;
    }
    while state.total_profit + profit < 0.0 {
        action = action
            .into_iter()
            .map(|u| if u.abs() < EPS { 0.0 } else { u * 0.95 })
            .collect();
        profit = challenge.compute_profit(state, &action);
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
