
# Running in Python

To test the baseline solution, run the below python cmd from `shared/competition/src/competition/energy_arbitrage/python`.  This will use the `energy_solver_2.py` policy.

```sh
python -m test_challenge --seed 123 --num-instances 10 --no-tests --sandbox
```

# Running in RUST

To test the baseline solution, run the below cargo command from `shared/competition/src/competition/energy_arbitrage/rust`.  This will use the `submission/src/energy_solver_2.rs` policy.

```sh
NUM_INSTANCES=10 MASTER_SEED=123 cargo test -p submission compare_algorithms -- --nocapture
```
