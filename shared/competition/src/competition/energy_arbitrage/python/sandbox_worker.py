"""Subprocess worker for sandboxed policy evaluation.

This script is launched by SandboxedEvaluator in a separate process.
It reads a PolicyView + State stream from stdin, calls the policy function,
and writes actions back to stdout.  The policy never has access to the
Challenge object, seeds, or RT-price generation machinery.

Protocol (newline-delimited JSON on stdin/stdout):

  stdin  line 1:  {"policy_module": "...", "sys_path": [...]}
  stdin  line 2:  PolicyView dict (sent once)
  stdin  line 3+: {"type": "step", "state": {...}}  |  {"type": "done"}
  stdout line *:  {"action": [...]}  |  {"error": "..."}
"""

import importlib
import json
import sys
import traceback


def main() -> None:
    config_line = sys.stdin.readline()
    if not config_line:
        return
    config = json.loads(config_line)

    for p in config.get("sys_path", []):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    policy_module_name = config["policy_module"]
    mod = importlib.import_module(policy_module_name)
    policy_fn = mod.policy

    from competition.energy_arbitrage.python.policy_view import PolicyView, state_from_dict

    view_line = sys.stdin.readline()
    if not view_line:
        return
    view = PolicyView.from_dict(json.loads(view_line))

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        msg = json.loads(line)
        if msg.get("type") == "done":
            break

        state = state_from_dict(msg["state"])

        try:
            action = policy_fn(view, state)
            out = json.dumps({"action": list(action)}, separators=(",", ":"))
        except Exception as e:
            tb = traceback.format_exc()
            out = json.dumps({"error": f"{type(e).__name__}: {e}", "traceback": tb}, separators=(",", ":"))

        sys.stdout.write(out + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
