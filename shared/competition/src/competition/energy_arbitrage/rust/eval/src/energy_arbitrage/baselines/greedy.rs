use crate::energy_arbitrage::{constants, Challenge, State};
use anyhow::{anyhow, Result};

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

// --- End Feasibility Helpers ---

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let time_step = state.time_step;
    let horizon = 12; // Look ahead 3 hours

    let mut best_action = vec![0.0; challenge.num_batteries];

    let da_prices = &challenge.market.day_ahead_prices;
    let current_da = da_prices[time_step][0]; // Approximate with node 0

    // Calculate average future DA price
    let end_step = (time_step + horizon).min(challenge.num_steps);
    let mut future_sum = 0.0;
    let mut future_count = 0.0;
    for t in time_step + 1..end_step {
        future_sum += da_prices[t][0];
        future_count += 1.0;
    }
    let future_avg = if future_count > 0.0 {
        future_sum / future_count
    } else {
        current_da
    };

    // Check future congestion
    // We can calculate "Net Load" for the whole grid in future steps
    // High Net Load -> Likely Congestion -> Higher Prices
    let mut future_congestion_risk = 0.0;
    for t in time_step + 1..end_step {
        let net_load: f64 = challenge.exogenous_injections[t].iter().sum();
        if net_load > 100.0 {
            // Arbitrary threshold
            future_congestion_risk += 1.0;
        }
    }

    for i in 0..challenge.num_batteries {
        let (min_bound, max_bound) = state.action_bounds[i];

        // If current price is lower than future average, charge
        // But if future has high congestion risk, price will be even higher, so charge more aggressively
        let threshold_adjust = future_congestion_risk * 2.0;

        if current_da < future_avg - 5.0 - threshold_adjust {
            // Charge
            best_action[i] = min_bound;
        } else if current_da > future_avg + 5.0 + threshold_adjust {
            // Discharge
            best_action[i] = max_bound;
        } else {
            best_action[i] = 0.0;
        }
    }

    // Enforce feasibility
    enforce_flow_feasibility(challenge, state, best_action)
}
