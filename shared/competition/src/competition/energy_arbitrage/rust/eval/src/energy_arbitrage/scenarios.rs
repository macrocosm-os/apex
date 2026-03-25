pub struct ScenarioConfig {
    /// Number of nodes (n)
    pub num_nodes: usize,
    /// Number of lines (L)
    pub num_lines: usize,
    /// Number of batteries (m)
    pub num_batteries: usize,
    /// Number of time steps (H)
    pub num_steps: usize,
    /// Congestion scaling factor (γ_cong) - scales line limits
    pub gamma_cong: f64,
    /// Volatility (σ)
    pub sigma: f64,
    /// Jump probability (ρ_jump)
    pub rho_jump: f64,
    /// Pareto tail index (α)
    pub alpha: f64,
    /// Fleet heterogeneity (h) - 0 = identical, 1 = 3x spread
    pub heterogeneity: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scenario {
    /// Track 1: Correctness-and-baseline regime
    /// Small network, nominal limits, low volatility, rare thin-tailed spikes
    BASELINE,

    /// Track 2: Meaningful congestion with increased stochasticity
    /// Tighter limits, more volatility, more diverse fleet
    CONGESTED,

    /// Track 3: Multi-day horizon with larger scale
    /// Longer horizon, larger network, more frequent spikes
    MULTIDAY,

    /// Track 4: High network density with frequent congestion
    /// Dense network, heavy tails, risk management critical
    DENSE,

    /// Track 5: Capstone regime
    /// Largest scale, tightest limits, heaviest tails
    CAPSTONE,
}

impl From<Scenario> for ScenarioConfig {
    fn from(scenario: Scenario) -> Self {
        match scenario {
            Scenario::BASELINE => ScenarioConfig {
                num_nodes: 20,
                num_lines: 30,
                num_batteries: 10,
                num_steps: 96,
                gamma_cong: 1.00,
                sigma: 0.10,
                rho_jump: 0.01,
                alpha: 4.0,
                heterogeneity: 0.2,
            },
            Scenario::CONGESTED => ScenarioConfig {
                num_nodes: 40,
                num_lines: 60,
                num_batteries: 20,
                num_steps: 96,
                gamma_cong: 0.80,
                sigma: 0.15,
                rho_jump: 0.02,
                alpha: 3.5,
                heterogeneity: 0.4,
            },
            Scenario::MULTIDAY => ScenarioConfig {
                num_nodes: 80,
                num_lines: 120,
                num_batteries: 40,
                num_steps: 192,
                gamma_cong: 0.60,
                sigma: 0.20,
                rho_jump: 0.03,
                alpha: 3.0,
                heterogeneity: 0.6,
            },
            Scenario::DENSE => ScenarioConfig {
                num_nodes: 100,
                num_lines: 200,
                num_batteries: 60,
                num_steps: 192,
                gamma_cong: 0.50,
                sigma: 0.25,
                rho_jump: 0.04,
                alpha: 2.7,
                heterogeneity: 0.8,
            },
            Scenario::CAPSTONE => ScenarioConfig {
                num_nodes: 150,
                num_lines: 300,
                num_batteries: 100,
                num_steps: 192,
                gamma_cong: 0.40,
                sigma: 0.30,
                rho_jump: 0.05,
                alpha: 2.5,
                heterogeneity: 1.0,
            },
        }
    }
}

impl std::fmt::Display for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scenario::BASELINE => write!(f, "baseline"),
            Scenario::CONGESTED => write!(f, "congested"),
            Scenario::MULTIDAY => write!(f, "multiday"),
            Scenario::DENSE => write!(f, "dense"),
            Scenario::CAPSTONE => write!(f, "capstone"),
        }
    }
}

impl std::str::FromStr for Scenario {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "baseline" => Ok(Scenario::BASELINE),
            "congested" => Ok(Scenario::CONGESTED),
            "multiday" => Ok(Scenario::MULTIDAY),
            "dense" => Ok(Scenario::DENSE),
            "capstone" => Ok(Scenario::CAPSTONE),
            _ => Err(anyhow::anyhow!("Invalid scenario type: {}", s)),
        }
    }
}
