# Sandbox Startup Config Guide

This document explains how to use `SandboxStartupConfig` to run optional setup work inside a sandbox before the main competition process starts.

## What It Is

`SandboxStartupConfig` is an optional startup hook attached to `SandboxRunRules`.

- If present, the setup command runs inside the sandbox container.
- The main command starts only after setup completes successfully.
- `sandbox_ready` is emitted after setup, so existing startup checks naturally wait for it.
- If setup fails (non-zero exit code), sandbox startup fails.

## Where To Configure It

Override `get_startup_config()` in your evaluation runner:

```python
from common.models.sandbox import SandboxStartupConfig


class MyEvaluation(BaseEvaluation):
    def get_startup_config(self, sandbox_role: str | None = None) -> SandboxStartupConfig | None:
        if sandbox_role == "compression":
            return SandboxStartupConfig(
                command=["python", "-m", "competition.my_comp.setup.prepare_cache"]
            )
        return None
```

Then pass it to `SandboxRunRules`:

```python
run_rules = SandboxRunRules(
    sandbox_index=0,
    filename="solution.py",
    command=["python", "solution.py", "--port", "8001"],
    startup_config=self.get_startup_config("compression"),
)
```

## Command Forms

`SandboxStartupConfig.command` supports:

- `str`: shell command string
- `list[str]`: argv-style command

Example (`str`):

```python
SandboxStartupConfig(
    command="python -m competition.my_comp.setup.prepare_cache --k 256 --out /workspace/cache.bin"
)
```

Example (`list[str]`, recommended for safer quoting):

```python
SandboxStartupConfig(
    command=[
        "python",
        "-m",
        "competition.my_comp.setup.prepare_cache",
        "--k",
        "256",
        "--out",
        "/workspace/cache.bin",
    ]
)
```

## Running Complex Python Setup (Clustering, Precompute, etc.)

You are not limited to simple shell setup. Use startup command to run any executable available in the sandbox image, including Python scripts/modules that perform heavy computations.

Typical pattern:

1. Put setup code in the competition package image.
2. Run it via startup command.
3. Write artifacts to `/workspace` (shared mount) for the main process to consume.

Example:

```python
SandboxStartupConfig(
    command=[
        "python",
        "-m",
        "competition.matrix_compression.setup.cluster_inputs",
        "--input-dir",
        "/workspace/input",
        "--output",
        "/workspace/precomputed_clusters.npz",
        "--num-clusters",
        "1024",
    ]
)
```

Then your main `solution.py` can load `/workspace/precomputed_clusters.npz`.

## Role-Based Setup

If your evaluation has multiple sandbox roles (for example `compression` and `decompression`), branch on `sandbox_role`:

```python
def get_startup_config(self, sandbox_role: str | None = None) -> SandboxStartupConfig | None:
    if sandbox_role == "compression":
        return SandboxStartupConfig(command=["python", "-m", "competition.x.setup.compress_side"])
    if sandbox_role == "decompression":
        return SandboxStartupConfig(command=["python", "-m", "competition.x.setup.decompress_side"])
    return None
```

## Practical Constraints

- Setup runs under normal sandbox limits (CPU, memory, timeout, network policy).
- Setup counts toward startup time; very heavy setup may require larger startup timeout.
- If setup needs network, ensure run rules allow it (`allow_internet=True` if appropriate).
- Keep setup idempotent when possible so retries are safe.

## When Not To Use Startup Setup

If work is deterministic and can be baked once, prefer image build-time setup (`Dockerfile`) over runtime startup setup.

Use startup setup when data is:

- submission-specific
- round-specific
- dynamic/runtime-dependent

## Troubleshooting

- Check sandbox log file: startup output is written before `sandbox_ready`.
- If startup hangs, confirm setup command exits and does not wait forever.
- If startup fails immediately, run the same command manually in a dev container to verify dependencies and paths.
