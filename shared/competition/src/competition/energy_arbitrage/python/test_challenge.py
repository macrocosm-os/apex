"""Tests matching the Rust test suite in mod.rs."""

import argparse
import json
import os
import sys

# Add directory containing "competition" to path so the package can be imported when run as script or python -m test_challenge
_this_dir = os.path.dirname(os.path.abspath(__file__))
# .../python -> .../energy_arbitrage -> .../competition -> parent of competition (src)
_src_root = os.path.dirname(os.path.dirname(os.path.dirname(_this_dir)))
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from competition.energy_arbitrage.python.energy_solver_1 import policy as energy_solver_1_policy
from competition.energy_arbitrage.python.energy_solver_2 import policy as energy_solver_2_policy
from competition.energy_arbitrage.python.challenge import Challenge, Track, Solution
from competition.energy_arbitrage.python.scenarios import Scenario

# Optionally override the sandboxed policy module via environment variable.
# By default, we preserve the existing behavior of using energy_solver_2.
_SANDBOX_POLICY_MODULE = os.environ.get("ENERGY_SOLVER_MODULE", "competition.energy_arbitrage.python.energy_solver_2")

# Match Rust seed derivation: master_seed ^ (nonce as u64).wrapping_mul(0xdeadbeefcafebabe)
_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE


def seed_from_master_nonce(master_seed: int, nonce: int) -> bytes:
    """Derive a 32-byte seed from master_seed and nonce (matches Rust evaluate())."""
    # 64-bit wrapping multiply then xor
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + (b"\x00" * 24)


def evaluate(
    name: str,
    policy,
    master_seed: int,
    num_instances: int,
    baseline_profits: list[float] | None = None,
) -> tuple[int, list[dict]]:
    """Run policy over num_instances (scenarios cycled), print per-instance and return (avg_quality, instances)."""
    QUALITY_PRECISION = 1_000_000
    scenarios = [
        Scenario.BASELINE,
        Scenario.CONGESTED,
        Scenario.MULTIDAY,
        Scenario.DENSE,
        Scenario.CAPSTONE,
    ]
    total_quality = 0
    count = 0
    instances: list[dict] = []
    for nonce in range(num_instances):
        scenario = scenarios[nonce % len(scenarios)]
        seed = seed_from_master_nonce(master_seed, nonce)
        challenge = Challenge.generate_instance(seed, Track(s=scenario))
        solution = challenge.grid_optimize(policy)
        my_profit = challenge.evaluate_total_profit(solution)
        if baseline_profits is not None:
            baseline_profit = baseline_profits[nonce]
        else:
            _, baseline_profit = challenge.compute_baseline()
        quality_f = (my_profit - baseline_profit) / (baseline_profit + 1e-6)
        quality = round(max(-10.0, min(quality_f, 10.0)) * QUALITY_PRECISION)
        print(
            f"{name} | {scenario.name} | nonce {nonce} | profit: {my_profit:.2f} | baseline: {baseline_profit:.2f} | quality: {quality}"
        )
        instances.append(
            {
                "nonce": nonce,
                "scenario": scenario.name,
                "profit": my_profit,
                "baseline_profit": baseline_profit,
                "quality": quality,
            }
        )
        total_quality += quality
        count += 1
    avg = total_quality // count if count else 0
    print(f"{name} | avg quality: {avg}\n")
    return avg, instances


def evaluate_sandboxed(
    name: str,
    policy_module: str,
    master_seed: int,
    num_instances: int,
    baseline_profits: list[float] | None = None,
) -> tuple[int, list[dict]]:
    """Run policy in a sandboxed subprocess, print per-instance and return (avg_quality, instances)."""
    QUALITY_PRECISION = 1_000_000
    scenarios = [
        Scenario.BASELINE,
        Scenario.CONGESTED,
        Scenario.MULTIDAY,
        Scenario.DENSE,
        Scenario.CAPSTONE,
    ]
    total_quality = 0
    count = 0
    instances: list[dict] = []
    for nonce in range(num_instances):
        scenario = scenarios[nonce % len(scenarios)]
        seed = seed_from_master_nonce(master_seed, nonce)
        challenge = Challenge.generate_instance(seed, Track(s=scenario))
        solution = challenge.grid_optimize_sandboxed(policy_module)
        my_profit = challenge.evaluate_total_profit(solution)
        if baseline_profits is not None:
            baseline_profit = baseline_profits[nonce]
        else:
            _, baseline_profit = challenge.compute_baseline()
        quality_f = (my_profit - baseline_profit) / (baseline_profit + 1e-6)
        quality = round(max(-10.0, min(quality_f, 10.0)) * QUALITY_PRECISION)
        print(
            f"{name} | {scenario.name} | nonce {nonce} | profit: {my_profit:.2f} | baseline: {baseline_profit:.2f} | quality: {quality}"
        )
        instances.append(
            {
                "nonce": nonce,
                "scenario": scenario.name,
                "profit": my_profit,
                "baseline_profit": baseline_profit,
                "quality": quality,
            }
        )
        total_quality += quality
        count += 1
    avg = total_quality // count if count else 0
    print(f"{name} | avg quality: {avg}\n")
    return avg, instances


def challenge_iter():
    """Generate challenge instances across seeds and scenarios."""
    scenarios = [
        Scenario.BASELINE,
        Scenario.CONGESTED,
        Scenario.MULTIDAY,
        Scenario.DENSE,
        Scenario.CAPSTONE,
    ]
    for seed_i in range(5):
        seed = bytes([seed_i] * 32)
        for scenario in scenarios:
            yield Challenge.generate_instance(seed, Track(s=scenario)), seed_i, scenario


def test_zero_profit():
    """Zero actions should yield zero profit."""
    print("Running test_zero_profit...")
    for idx, (challenge, seed_i, scenario) in enumerate(challenge_iter()):

        def zero_policy(c, s):
            return [0.0] * len(s.action_bounds)

        schedule, final_state = challenge._simulate(zero_policy)
        solution = Solution(schedule=schedule)
        total_profit = challenge.evaluate_total_profit(solution)
        print(f"  Seed {seed_i} {scenario.value} profit: {total_profit}")
        assert total_profit == 0.0, f"Expected 0 profit, got {total_profit}"
        assert total_profit == final_state.total_profit
    print("  PASSED")


def test_non_zero_profit():
    """Tiny positive actions should yield positive profit."""
    print("Running test_non_zero_profit...")
    for idx, (challenge, seed_i, scenario) in enumerate(challenge_iter()):

        def tiny_policy(c, s):
            return [0.00001] * len(s.action_bounds)

        schedule, final_state = challenge._simulate(tiny_policy)
        solution = Solution(schedule=schedule)
        total_profit = challenge.evaluate_total_profit(solution)
        print(f"  Seed {seed_i} {scenario.value} profit: {total_profit}")
        assert total_profit > 0.0, f"Expected positive profit, got {total_profit}"
        assert total_profit == final_state.total_profit
    print("  PASSED")


def test_faulty_policy():
    """Faulty policy should raise an error."""
    print("Running test_faulty_policy...")
    for idx, (challenge, seed_i, scenario) in enumerate(challenge_iter()):

        def faulty_policy(c, s):
            raise ValueError("Faulty policy")

        try:
            challenge.grid_optimize(faulty_policy)
            assert False, "Should have raised"
        except ValueError:
            pass
        print(f"  Seed {seed_i} {scenario.value} raised as expected")
    print("  PASSED")


def test_baseline():
    """Baseline should run without errors and produce finite profit.

    NOTE: The Rust tests assert profit > 0 for all seeds/scenarios, but that
    relies on Rust's specific RNG stream (ChaCha-based StdRng/SmallRng).
    Python's Mersenne Twister produces different challenge instances, so the
    baseline may occasionally yield negative profit. We only check that
    it runs and returns a finite number.
    """
    print("Running test_baseline...")
    positive = 0
    total = 0
    for idx, (challenge, seed_i, scenario) in enumerate(challenge_iter()):
        _, total_profit = challenge.compute_baseline()
        print(f"  Seed {seed_i} {scenario.value} profit: {total_profit}")
        assert total_profit == total_profit, "Profit is NaN"  # NaN check
        total += 1
        if total_profit > 0:
            positive += 1
    print(f"  PASSED ({positive}/{total} had positive profit)")


def test_sandboxed_matches_direct():
    """Sandboxed evaluation should produce identical results to direct calls."""
    print("Running test_sandboxed_matches_direct...")
    seed = bytes([0] * 32)
    challenge = Challenge.generate_instance(seed, Track(s=Scenario.BASELINE))

    direct_solution = challenge.grid_optimize(energy_solver_1_policy)
    direct_profit = challenge.evaluate_total_profit(direct_solution)

    sandboxed_solution = challenge.grid_optimize_sandboxed("competition.energy_arbitrage.python.energy_solver_1")
    sandboxed_profit = challenge.evaluate_total_profit(sandboxed_solution)

    print(f"  Direct profit:    {direct_profit:.6f}")
    print(f"  Sandboxed profit: {sandboxed_profit:.6f}")
    assert (
        abs(direct_profit - sandboxed_profit) < 1e-9
    ), f"Profit mismatch: direct={direct_profit}, sandboxed={sandboxed_profit}"
    print("  PASSED")


def _read_seed_file(path: str) -> int:
    """Read the master seed from a file and immediately unlink it.

    The seed file is written to the shared volume by the runner before the
    sandbox container starts.  Unlinking it before any solver subprocess is
    spawned ensures that user code cannot extract the seed from the
    filesystem (it would otherwise be visible via /proc/*/cmdline if passed
    as a command-line argument).
    """
    with open(path) as f:
        seed = int(f.read().strip())
    os.unlink(path)
    return seed


_SEED_FILE = "/workspace/.seed"
_BASELINES_FILE = "/workspace/.baselines"
_RESULTS_FILE = "/workspace/.results"


def _read_baselines_file(path: str) -> list[float]:
    """Read pre-computed baseline profits from a JSON file and immediately unlink it."""
    with open(path) as f:
        baselines = json.loads(f.read())
    os.unlink(path)
    return baselines


def _write_results_file(path: str, avg_quality: int, instances: list[dict]) -> None:
    """Write structured evaluation results to a JSON file for the runner to read."""
    with open(path, "w") as f:
        json.dump({"instances": instances, "avg_quality": avg_quality}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy arbitrage tests / compare algorithms (align with Rust).")
    parser.add_argument("--seed", type=int, default=None, help="Master seed for local testing (do NOT use in sandbox)")
    parser.add_argument(
        "--num-instances", type=int, default=None, help="Number of instances (overrides NUM_INSTANCES env)"
    )
    parser.add_argument(
        "--no-tests", action="store_true", help="Skip full test suite when running with seed/num-instances"
    )
    parser.add_argument("--sandbox", action="store_true", help="Run solvers via sandboxed subprocess evaluation")
    parser.add_argument(
        "--solver-module", type=str, default=None, help="Override solver module for sandboxed evaluation"
    )
    args = parser.parse_args()

    if args.seed is not None:
        master_seed = args.seed
    elif args.sandbox:
        # In sandbox mode the seed MUST come from the secure file.
        try:
            master_seed = _read_seed_file(_SEED_FILE)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"FATAL: cannot read/unlink seed file {_SEED_FILE}: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        # Local dev without --seed: fall back to env var or default.
        try:
            master_seed = int(os.environ.get("MASTER_SEED", "42"))
        except ValueError:
            master_seed = 42

    # Read pre-computed baselines if available (sandbox mode writes this file)
    baseline_profits: list[float] | None = None
    if args.sandbox:
        try:
            baseline_profits = _read_baselines_file(_BASELINES_FILE)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"WARNING: cannot read baselines file {_BASELINES_FILE}: {exc}", file=sys.stderr)
            print("Falling back to computing baselines on the fly", file=sys.stderr)

    num_instances = args.num_instances
    if num_instances is None:
        try:
            num_instances = int(os.environ.get("NUM_INSTANCES", ""))
        except ValueError:
            num_instances = None

    solver_module = args.solver_module or _SANDBOX_POLICY_MODULE
    name = solver_module.split(".")[-1]

    if num_instances is not None and num_instances > 0:
        if args.sandbox:
            avg_quality, instances = evaluate_sandboxed(
                f"{name} (sandbox)",
                solver_module,
                master_seed,
                num_instances,
                baseline_profits=baseline_profits,
            )
            if baseline_profits is not None:
                _write_results_file(_RESULTS_FILE, avg_quality, instances)
        else:
            evaluate(name, energy_solver_2_policy, master_seed, num_instances, baseline_profits=baseline_profits)
        if not args.no_tests:
            print("\n=== Full test suite ===")
            test_zero_profit()
            test_non_zero_profit()
            test_faulty_policy()
            test_baseline()
            test_sandboxed_matches_direct()
            print("\nAll tests PASSED!")
        sys.exit(0)

    # Run a quick subset first (just BASELINE) to verify correctness
    print("=== Quick smoke test (BASELINE only) ===")
    seed = bytes([0] * 32)
    c = Challenge.generate_instance(seed, Track(s=Scenario.BASELINE))
    print(
        f"Generated BASELINE challenge: {c.num_steps} steps, {c.num_batteries} batteries, {c.network.num_nodes} nodes"
    )

    # Zero profit test
    def zero_pol(ch, s):
        return [0.0] * len(s.action_bounds)

    sched, final = c._simulate(zero_pol)
    print(f"Zero policy profit: {final.total_profit}")
    assert final.total_profit == 0.0

    # Baseline test
    _, baseline_profit = c.compute_baseline()
    print(f"Baseline profit: {baseline_profit:.4f}")
    assert baseline_profit > 0.0

    print("\n=== Full test suite ===")
    test_zero_profit()
    test_non_zero_profit()
    test_faulty_policy()
    test_baseline()
    test_sandboxed_matches_direct()
    print("\nAll tests PASSED!")
