"""Sandboxed policy evaluation via subprocess isolation.

SandboxedEvaluator runs the policy function in a child process that only
receives a serialized PolicyView and per-step State.  The hidden seed, RNG
state, and RT-price generation remain exclusively in the evaluator process.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from competition.energy_arbitrage.python.challenge import Challenge, Solution


class SandboxedEvaluator:
    """Evaluate a policy in an isolated subprocess.

    The subprocess receives only a PolicyView (public challenge data) and
    per-step State objects.  It never has access to the Challenge object,
    the hidden seed, or the ability to generate RT prices.
    """

    def __init__(self, timeout_per_step: float = 30.0):
        self.timeout_per_step = timeout_per_step

    def run(self, challenge: "Challenge", policy_module: str) -> "Solution":
        """Run a full simulation with the policy executing in a subprocess.

        Args:
            challenge: The Challenge instance (stays in this process).
            policy_module: Fully-qualified module name containing a ``policy``
                function, e.g. ``"competition.energy_arbitrage.python.energy_solver_1"``.

        Returns:
            A Solution containing the action schedule.
        """
        from competition.energy_arbitrage.python.challenge import NextRTPricesGenerate, Solution
        from competition.energy_arbitrage.python.policy_view import state_to_dict

        view = challenge.to_policy_view()

        worker_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sandbox_worker.py",
        )

        env = os.environ.copy()
        python_paths = [p for p in sys.path if p]
        cwd = os.getcwd()
        if cwd not in python_paths:
            python_paths.insert(0, cwd)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)

        proc = subprocess.Popen(
            [sys.executable, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )

        try:
            self._send(
                proc,
                {
                    "policy_module": policy_module,
                    "sys_path": python_paths,
                },
            )

            self._send(proc, view.to_dict())

            rng = random.Random()
            rng.seed(challenge._hidden_seed)
            state = challenge._initial_state(rng)
            schedule: list[list[float]] = []

            for step in range(challenge.num_steps):
                self._send(proc, {"type": "step", "state": state_to_dict(state)})
                response = self._receive(proc)

                if "error" in response:
                    tb = response.get("traceback", "")
                    raise RuntimeError(f"Policy error at step {step}: {response['error']}\n{tb}")

                action = response["action"]

                next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
                state = challenge.take_step(state, action, NextRTPricesGenerate(next_seed))
                schedule.append(action)

            self._send(proc, {"type": "done"})
            proc.wait(timeout=5)

            return Solution(schedule=schedule)

        except Exception:
            proc.kill()
            stderr_output = proc.stderr.read() if proc.stderr else ""
            if stderr_output:
                sys.stderr.write(f"Sandbox worker stderr:\n{stderr_output}\n")
            raise
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    @staticmethod
    def _send(proc: subprocess.Popen, data: dict) -> None:
        assert proc.stdin is not None
        line = json.dumps(data, separators=(",", ":"))
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    @staticmethod
    def _receive(proc: subprocess.Popen) -> dict:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        if not line:
            stderr_output = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"Sandbox worker exited unexpectedly. stderr:\n{stderr_output}")
        return json.loads(line)
