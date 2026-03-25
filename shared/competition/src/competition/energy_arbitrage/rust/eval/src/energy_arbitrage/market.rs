use super::constants;
use super::utils::*;
use rand::{
    distributions::Distribution,
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Market parameters for price generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketParams {
    /// Volatility (σ)
    pub volatility: f64,
    /// Jump probability (ρ_jump)
    pub jump_probability: f64,
    /// Pareto tail index (α)
    pub tail_index: f64,
}

/// Market parameters for price generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    /// Params
    pub params: MarketParams,
    /// Day-ahead prices (λ_da)
    pub day_ahead_prices: Vec<Vec<f64>>,
}

impl Market {
    /// Generate a market instance with given parameters
    pub(crate) fn generate_instance(
        rng: &mut impl Rng,
        params: MarketParams,
        num_nodes: usize,
        num_steps: usize,
    ) -> Self {
        // Base price curve via Gaussian Process (or simple sinusoidal for efficiency)
        let kernel = GPKernel::new();
        let k = kernel.covariance_matrix(num_steps);
        let l = cholesky(&k);

        // Generate standard normal samples
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z: Vec<f64> = (0..num_steps).map(|_| normal.sample(rng)).collect();

        let mut base_prices = vec![0.0; num_steps];

        for i in 0..num_steps {
            for j in 0..num_steps {
                base_prices[i] += l[i][j] * z[j];
            }
            // Add diurnal pattern based on 15-min steps
            let hour = (i as f64) * constants::DELTA_T;
            base_prices[i] += constants::MEAN_DA_PRICE
                + constants::DA_AMPLITUDE * (2.0 * PI * hour / 24.0 - PI / 2.0).sin();
            base_prices[i] = base_prices[i].max(constants::LAMBDA_DA_MIN);
        }

        // Generate node offsets (correlated AR(1) residual)
        let mut prices = vec![vec![0.0; num_nodes]; num_steps];
        let ar_coef: f64 = 0.8;

        for node in 0..num_nodes {
            let offset: f64 = 5.0 * (rng.r#gen::<f64>() - 0.5);
            let mut residual: f64 = 0.0;

            for t in 0..num_steps {
                residual = ar_coef * residual
                    + (1.0_f64 - ar_coef * ar_coef).sqrt() * 2.0 * normal.sample(rng);
                let price = base_prices[t] + offset + residual;
                prices[t][node] = price.max(constants::LAMBDA_DA_MIN);
            }
        }

        Market {
            params,
            day_ahead_prices: prices,
        }
    }

    /// Generate real-time prices for a given time step
    pub fn generate_rt_prices(
        &self,
        rng: &mut impl Rng,
        time_step: usize,
        congestion_indicators: &[bool],
    ) -> Vec<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();

        let num_nodes = self.day_ahead_prices[0].len();
        let mut prices = Vec::with_capacity(num_nodes);

        // Draw common factor z_t
        let z_common: f64 = normal.sample(rng);
        // Draw z'_t for congestion premium
        let z_prime = normal.sample(rng);
        let zeta = z_prime.max(0.0);

        for i in 0..num_nodes {
            let da_price = self.day_ahead_prices[time_step][i];

            // Draw idiosyncratic shock ε_i,t
            let eps_i = normal.sample(rng);

            // Spatially correlated shock
            let rho = constants::RHO_SPATIAL; // Spatial correlation (ρ_sp)
            let xi_i = rho.sqrt() * z_common + (1.0 - rho).sqrt() * eps_i;

            // Base price with shock
            let mu = constants::MU_BIAS; // RT bias term (μ)
            let sigma = self.params.volatility;
            let mut price = da_price * (1.0 + mu + sigma * xi_i);

            // Congestion premium (uses lagged indicator)
            if congestion_indicators[i] {
                price += constants::GAMMA_PRICE * zeta; // Congestion premium scale (γ_price)
            }

            // Jump component
            let u_jump = rng.r#gen::<f64>();

            if u_jump < self.params.jump_probability {
                let u_pareto = rng.r#gen::<f64>().max(1e-10);
                // Pareto: X = (1-U)^(-1/α), support [1,∞)
                let pareto = (1.0 - u_pareto).powf(-1.0 / self.params.tail_index);
                let jump = da_price * pareto;
                price += jump;
            }

            prices.push(price.clamp(constants::LAMBDA_MIN, constants::LAMBDA_MAX));
        }

        prices
    }
}
