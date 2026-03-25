use super::constants;
use nalgebra::DMatrix;
use std::f64::consts::PI;

/// Gaussian Process kernel for price generation
pub struct GPKernel {
    pub sigma_periodic: f64,
    pub length_periodic: f64,
    pub sigma_se: f64,
    pub length_se: f64,
    pub period_hours: f64,
}

impl GPKernel {
    pub fn new() -> Self {
        Self {
            sigma_periodic: 10.0,
            length_periodic: 2.0,
            sigma_se: 5.0,
            length_se: 4.0,
            period_hours: 24.0,
        }
    }

    pub fn evaluate(&self, t1: f64, t2: f64) -> f64 {
        // Convert step indices to hours
        let h1 = t1 * constants::DELTA_T;
        let h2 = t2 * constants::DELTA_T;
        let tau = (h1 - h2).abs();

        // Periodic component
        let periodic = self.sigma_periodic.powi(2)
            * (-2.0 * (PI * tau / self.period_hours).sin().powi(2) / self.length_periodic.powi(2))
                .exp();

        // Squared exponential component
        let se = self.sigma_se.powi(2) * (-tau.powi(2) / (2.0 * self.length_se.powi(2))).exp();

        periodic + se
    }

    pub fn covariance_matrix(&self, n: usize) -> Vec<Vec<f64>> {
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = self.evaluate(i as f64, j as f64);
                if i == j {
                    k[i][j] += 1e-6; // Numerical stability
                }
            }
        }
        k
    }
}

/// Cholesky decomposition (lower triangular)
pub fn cholesky(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }

    // Build nalgebra matrix
    let matrix = DMatrix::from_fn(n, n, |i, j| a[i][j]);

    // Cholesky decomposition - returns lower triangular L where A = L * L^T
    let chol = matrix
        .cholesky()
        .expect("Covariance matrix should be positive definite");

    // Extract lower triangular factor
    let l_matrix = chol.l();

    // Convert back to Vec<Vec<f64>>
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            l[i][j] = l_matrix[(i, j)];
        }
    }
    l
}

/// Matrix inversion via Cholesky decomposition (for symmetric positive definite matrices)
pub fn invert_matrix(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    use nalgebra::DMatrix;

    let n = a.len();
    if n == 0 {
        return vec![];
    }

    // Build nalgebra matrix
    let matrix = DMatrix::from_fn(n, n, |i, j| a[i][j]);

    // Cholesky decomposition and inverse
    let inverse = matrix
        .cholesky()
        .expect("B_red should be positive definite for a connected network")
        .inverse();

    // Convert back to Vec<Vec<f64>>
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = inverse[(i, j)];
        }
    }
    result
}
