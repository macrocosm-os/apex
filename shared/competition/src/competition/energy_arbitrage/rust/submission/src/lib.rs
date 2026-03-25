// pub mod energy_solver_1;
pub mod energy_solver_2;

#[cfg(test)]
mod tests {
    use eval::energy_arbitrage::{Challenge, State, Track, Scenario};
    use anyhow::Result;

    fn evaluate(name: &str, policy: &dyn Fn(&Challenge, &State) -> Result<Vec<f64>>, master_seed: u64, num_instances: usize) {
        let scenarios = [
            Scenario::BASELINE,
            Scenario::CONGESTED,
            Scenario::MULTIDAY,
            Scenario::DENSE,
            Scenario::CAPSTONE,
        ];

        let mut total_quality = 0i32;
        let mut count = 0;

        for nonce in 0..num_instances {
            let scenario = &scenarios[nonce % scenarios.len()];

            let mut seed = [0u8; 32];
            let bytes = (master_seed ^ (nonce as u64).wrapping_mul(0xdeadbeefcafebabe)).to_le_bytes();
            seed[..8].copy_from_slice(&bytes);

            let challenge = Challenge::generate_instance(&seed, &Track { s: *scenario }).unwrap();
            eval::energy_arbitrage::CALLED_GRID_OPTIMIZE.store(false, std::sync::atomic::Ordering::SeqCst);
            let solution = challenge.grid_optimize(policy).unwrap();
            let my_profit = challenge.evaluate_total_profit(&solution).unwrap();
            let (_, baseline_profit) = challenge.compute_baseline().unwrap();
            let quality_f = (my_profit - baseline_profit) / (baseline_profit + 1e-6);
            let quality = (quality_f.clamp(-10.0, 10.0) * 1_000_000.0).round() as i32;
            println!(
                "{} | {:?} | nonce {} | profit: {:.2} | baseline: {:.2} | quality: {}",
                name, scenario, nonce, my_profit, baseline_profit, quality
            );
            total_quality += quality;
            count += 1;
        }
        println!("{} | avg quality: {}\n", name, total_quality / count);
    }

    #[test]
    // run by: NUM_INSTANCES=10 MASTER_SEED=42 cargo test -p submission compare_algorithms -- --nocapture
    fn compare_algorithms() {
        let master_seed: u64 = std::env::var("MASTER_SEED")
            .unwrap_or_else(|_| "42".to_string())
            .parse()
            .expect("MASTER_SEED must be a valid u64");
        // Remove from environment immediately so submission code cannot read it
        // Safety: single-threaded test context; no other threads read env vars here
        unsafe { std::env::remove_var("MASTER_SEED") };

        let num_instances: usize = std::env::var("NUM_INSTANCES")
            .unwrap_or_else(|_| "10".to_string())
            .parse()
            .expect("NUM_INSTANCES must be a valid usize");
        unsafe { std::env::remove_var("NUM_INSTANCES") };

        // evaluate("energy_solver_1", &crate::energy_solver_1::policy, master_seed, num_instances);
        evaluate("energy_solver_2", &crate::energy_solver_2::policy, master_seed, num_instances);
    }
}
