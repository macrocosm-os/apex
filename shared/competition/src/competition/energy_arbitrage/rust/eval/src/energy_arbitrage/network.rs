use super::constants;
use super::utils::invert_matrix;
use anyhow::{anyhow, Result};
use rand::{distributions::Distribution, Rng};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Network topology and DC power flow parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    /// Number of nodes (n)
    pub num_nodes: usize,
    /// Number of lines (L)
    pub num_lines: usize,
    /// Line definitions: (from_node, to_node)
    pub lines: Vec<(usize, usize)>,
    /// Line susceptances (b_l)
    pub susceptances: Vec<f64>,
    /// Nominal line flow limits (MW)
    pub nominal_flow_limits: Vec<f64>,
    /// Effective line flow limits after congestion scaling (MW)
    pub flow_limits: Vec<f64>,
    /// Power Transfer Distribution Factor matrix (L x n)
    pub ptdf: Vec<Vec<f64>>,
    /// Slack bus index
    pub slack_bus: usize,
    /// Incidence: which lines are incident to each node
    pub node_incident_lines: Vec<Vec<usize>>,
    /// Congestion proximity threshold (τ_cong)
    pub congestion_threshold: f64,
}

impl Network {
    /// Generate a connected network with given parameters
    pub(crate) fn generate_instance(
        rng: &mut impl Rng,
        num_nodes: usize,
        num_lines: usize,
        gamma_cong: f64,
    ) -> Self {
        // Start with a spanning tree (n-1 lines), then add extra lines
        let mut lines = Vec::new();
        let mut susceptances = Vec::new();
        let mut nominal_flow_limits = Vec::new();

        // Phase 1: Create spanning tree using random edges
        let mut connected = vec![false; num_nodes];
        connected[0] = true;
        let mut connected_count = 1;

        while connected_count < num_nodes {
            // Pick a random unconnected node
            let unconnected: Vec<usize> = (0..num_nodes).filter(|&i| !connected[i]).collect();
            let new_node = unconnected[rng.gen_range(0..unconnected.len())];

            // Connect to a random connected node
            let connected_nodes: Vec<usize> = (0..num_nodes).filter(|&i| connected[i]).collect();
            let existing = connected_nodes[rng.gen_range(0..connected_nodes.len())];

            let (from, to) = if new_node < existing {
                (new_node, existing)
            } else {
                (existing, new_node)
            };

            lines.push((from, to));
            susceptances.push(constants::BASE_SUSCEPTANCE * (0.8 + 0.4 * rng.r#gen::<f64>()));
            nominal_flow_limits
                .push(constants::NOMINAL_FLOW_LIMIT * (0.8 + 0.4 * rng.r#gen::<f64>()));

            connected[new_node] = true;
            connected_count += 1;
        }

        // Phase 2: Add extra lines to reach target by sampling from non-existing edges
        let extra_needed = num_lines.saturating_sub(lines.len());
        if extra_needed > 0 {
            // Build set of existing edges for O(1) lookup
            let existing: std::collections::HashSet<(usize, usize)> =
                lines.iter().cloned().collect();

            // Build list of all non-existing edges
            let mut candidates: Vec<(usize, usize)> = Vec::new();
            for i in 0..num_nodes {
                for j in (i + 1)..num_nodes {
                    if !existing.contains(&(i, j)) {
                        candidates.push((i, j));
                    }
                }
            }

            // Randomly select using Fisher-Yates partial shuffle
            let to_add = extra_needed.min(candidates.len());
            for k in 0..to_add {
                let idx = rng.gen_range(k..candidates.len());
                candidates.swap(k, idx);

                let (from, to) = candidates[k];
                lines.push((from, to));
                susceptances.push(constants::BASE_SUSCEPTANCE * (0.8 + 0.4 * rng.r#gen::<f64>()));
                nominal_flow_limits
                    .push(constants::NOMINAL_FLOW_LIMIT * (0.8 + 0.4 * rng.r#gen::<f64>()));
            }
        }

        // Compute PTDF matrix
        let slack_bus = constants::SLACK_BUS;
        let ptdf = Self::compute_ptdf(num_nodes, &lines, &susceptances, slack_bus);

        // Apply congestion scaling to get effective limits
        let flow_limits: Vec<f64> = nominal_flow_limits
            .iter()
            .map(|&f| f * gamma_cong)
            .collect();

        // Build node-to-incident-lines mapping
        let mut node_incident_lines = vec![Vec::new(); num_nodes];
        for (l, &(from, to)) in lines.iter().enumerate() {
            node_incident_lines[from].push(l);
            node_incident_lines[to].push(l);
        }

        Self {
            num_nodes,
            num_lines,
            lines,
            susceptances,
            nominal_flow_limits,
            flow_limits,
            ptdf,
            slack_bus,
            node_incident_lines,
            congestion_threshold: constants::TAU_CONG,
        }
    }

    /// Compute PTDF matrix using DC power flow
    fn compute_ptdf(
        num_nodes: usize,
        lines: &[(usize, usize)],
        susceptances: &[f64],
        slack_bus: usize,
    ) -> Vec<Vec<f64>> {
        if num_nodes == 0 || lines.is_empty() {
            return vec![];
        }

        // Build bus susceptance matrix B (n x n)
        let mut b_matrix = vec![vec![0.0; num_nodes]; num_nodes];
        for (l, &(i, j)) in lines.iter().enumerate() {
            let b = susceptances[l];
            b_matrix[i][i] += b;
            b_matrix[j][j] += b;
            b_matrix[i][j] -= b;
            b_matrix[j][i] -= b;
        }

        // Remove slack bus - create reduced (n-1) x (n-1) matrix
        let n_red = num_nodes - 1;
        let mut b_red = vec![vec![0.0; n_red]; n_red];
        let mut row_map = Vec::with_capacity(n_red);
        for i in 0..num_nodes {
            if i != slack_bus {
                row_map.push(i);
            }
        }

        for (ri, &i) in row_map.iter().enumerate() {
            for (rj, &j) in row_map.iter().enumerate() {
                b_red[ri][rj] = b_matrix[i][j];
            }
        }

        // Invert reduced matrix
        let x_red = invert_matrix(&b_red);

        // Build full X matrix (with zeros for slack)
        let mut x = vec![vec![0.0; num_nodes]; num_nodes];
        for (ri, &i) in row_map.iter().enumerate() {
            for (rj, &j) in row_map.iter().enumerate() {
                x[i][j] = x_red[ri][rj];
            }
        }

        // Compute PTDF: PTDF[l,k] = b_l * (X[i,k] - X[j,k])
        let num_lines = lines.len();
        let mut ptdf = vec![vec![0.0; num_nodes]; num_lines];
        for (l, &(i, j)) in lines.iter().enumerate() {
            let b = susceptances[l];
            for k in 0..num_nodes {
                ptdf[l][k] = b * (x[i][k] - x[j][k]);
            }
        }

        ptdf
    }

    /// Generate exogenous nodal injections
    pub(crate) fn generate_exogenous_injections(
        &self,
        rng: &mut impl Rng,
        num_steps: usize,
    ) -> Vec<Vec<f64>> {
        // Generate as low-rank spatiotemporal process
        // Two time factors x two node loading patterns + noise
        let mut injections = vec![vec![0.0; self.num_nodes]; num_steps];

        // Time patterns (sinusoidal load curves)
        let time_pattern1: Vec<f64> = (0..num_steps)
            .map(|t| (2.0 * PI * t as f64 / 96.0).sin())
            .collect();
        let time_pattern2: Vec<f64> = (0..num_steps)
            .map(|t| (2.0 * PI * t as f64 / 48.0 + PI / 4.0).sin())
            .collect();

        // Node patterns (random loadings)
        let node_pattern1: Vec<f64> = (0..self.num_nodes)
            .map(|_| rng.r#gen::<f64>() - 0.5)
            .collect();
        let node_pattern2: Vec<f64> = (0..self.num_nodes)
            .map(|_| rng.r#gen::<f64>() - 0.5)
            .collect();

        // Combine with noise
        let base_load = 200.0; // MW
        let pattern_scale = 60.0;
        let noise_scale = 5.0;

        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..self.num_nodes {
            if i == self.slack_bus {
                continue; // Slack bus injection computed later
            }
            for t in 0..num_steps {
                let pattern = pattern_scale
                    * (node_pattern1[i] * time_pattern1[t] + node_pattern2[i] * time_pattern2[t]);
                let noise = noise_scale * normal.sample(rng);
                injections[t][i] = base_load * (rng.r#gen::<f64>() - 0.5) + pattern + noise;
            }
        }

        // Balance at slack bus: p_s = -Σ_{i≠s} p_i
        for t in 0..num_steps {
            let mut sum = 0.0;
            for i in 0..self.num_nodes {
                if i != self.slack_bus {
                    sum += injections[t][i];
                }
            }
            injections[t][self.slack_bus] = -sum;
        }

        // Per-timestep rescaling: ensure exogenous flows stay within
        // flow_margin * effective_limit for each timestep independently.
        // This avoids the "one-bad-timestep-kills-all" problem of global scaling.
        let flow_margin = 0.85;
        for t in 0..num_steps {
            let flows = self.compute_flows(&injections[t]);
            let mut scale_t: f64 = 1.0;
            for (l, &flow) in flows.iter().enumerate() {
                let limit = self.flow_limits[l] * flow_margin;
                if flow.abs() > limit {
                    scale_t = scale_t.min(limit / flow.abs());
                }
            }
            if scale_t < 1.0 {
                for i in 0..self.num_nodes {
                    if i != self.slack_bus {
                        injections[t][i] *= scale_t;
                    }
                }
                // Re-balance at slack for this timestep
                let mut sum = 0.0;
                for i in 0..self.num_nodes {
                    if i != self.slack_bus {
                        sum += injections[t][i];
                    }
                }
                injections[t][self.slack_bus] = -sum;
            }
        }
        injections
    }

    /// Compute congestion indicators given nodal injections (including slack balancing)
    pub fn compute_flows(&self, injections: &[f64]) -> Vec<f64> {
        (0..self.num_lines)
            .map(|l| {
                (0..self.num_nodes)
                    .map(|k| self.ptdf[l][k] * injections[k])
                    .sum::<f64>()
            })
            .collect()
    }

    /// Verify that the flows are within the flow limits
    pub fn verify_flows(&self, flows: &[f64]) -> Result<()> {
        for (l, &flow) in flows.iter().enumerate() {
            let violation = flow.abs() - self.flow_limits[l];
            if violation > constants::EPS_FLOW * self.flow_limits[l] {
                return Err(anyhow!(
                    "Line {} flow limit violated: |{:.2}| > {:.2}",
                    l,
                    flow.abs(),
                    self.flow_limits[l]
                ));
            }
        }
        Ok(())
    }

    /// Generate congestion indicators given nodal injections (including slack balancing)
    pub fn generate_congestion_indicators(
        &self,
        rng: &mut impl Rng,
        exogenous_injections: &[f64],
    ) -> Vec<bool> {
        let mut indicators = vec![false; self.num_nodes];
        let flows = self.compute_flows(&exogenous_injections);
        for (l, &flow) in flows.iter().enumerate() {
            let p = (flow.abs() / (self.congestion_threshold * self.flow_limits[l])).powf(10.0);
            if p > rng.r#gen::<f64>() {
                let (from, to) = self.lines[l];
                indicators[from] = true;
                indicators[to] = true;
            }
        }
        indicators
    }
}
