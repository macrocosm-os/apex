from typing import Mapping
from pydantic import BaseModel


class SandboxBaseError(Exception):
    def __init__(self, message: str, name: str, description: str):
        self.message = message
        self.name = name
        self.description = description
        super().__init__(self.message)


class SandboxExecutionError(SandboxBaseError):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            name=__class__.__name__,
            description="Sandbox failed to execute the miner code",
        )


class SandboxStartupError(SandboxBaseError):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            name=__class__.__name__,
            description="Sandbox failed to startup",
        )


class SandboxTimeoutError(SandboxBaseError):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            name=__class__.__name__,
            description="Sandbox timed out executing the miner code",
        )


class SandboxOutputReadError(SandboxBaseError):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            name=__class__.__name__,
            description="Failed to read the sandbox output",
        )


class SandboxOutputValidationError(SandboxBaseError):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            name=__class__.__name__,
            description="Sandbox output validation failed",
        )


class SandboxBuildRules(BaseModel):
    dockerfile: str = "Dockerfile"
    tag: str | None = None
    network_mode: str | None = None


class SandboxStartupConfig(BaseModel):
    # Optional setup command executed before the main sandbox command.
    # Supports either a shell string or argv-style list.
    command: str | list[str]
    # Optional Kubernetes node pool for startup setup. If set and different from run_rules.nodepool,
    # Kubernetes sandbox runs startup as a separate pre-job on this pool.
    nodepool: str | None = None


class ReadinessProbeSpec(BaseModel):
    """Kubelet-driven readiness probe for a sandbox container.

    When set on SandboxRunRules, the K8s sandbox attaches this as a readinessProbe on
    the runner container and gates `exit_after_startup=True` on the kubelet flipping
    `containerStatuses[runner].ready` rather than grepping pod stdout for a token.
    Replaces the log-grep readiness path; ignored by the Docker sandbox.

    Defaults match the existing `max_wait_for_ready=60` ceiling at parity
    (failure_threshold * period_seconds = 60s) so this change swaps the readiness
    substrate without changing the cold-start budget.
    """

    # Probe form. Exactly one of http_get / exec_command / tcp_socket should be set.
    http_get: str | None = None  # path, e.g. "/health"
    exec_command: list[str] | None = None  # argv, e.g. ["test", "-f", "/workspace/.ready_to_run"]
    tcp_socket: bool = False  # use the port below for a TCP connection check

    port: int | None = None  # required for http_get and tcp_socket

    initial_delay_seconds: int = 2
    period_seconds: int = 1
    timeout_seconds: int = 2
    failure_threshold: int = 60  # 60 * 1s = 60s budget; bump per-competition on evidence
    success_threshold: int = 1

    # Number of log lines to fetch synchronously on probe failure. None means use the slow-failure fallback
    # (since_time=<pod creation>) instead of a tail.
    failure_log_tail_lines: int | None = 200


class SandboxRunRules(BaseModel):
    sandbox_index: int = 0
    run_timeout_in_seconds: int = 60
    # Seconds to wait for the sandbox container to signal readiness before
    # raising SandboxStartupError. Applies when `exit_after_startup=True`.
    startup_timeout_in_seconds: int = 60
    filename: str = "solution.py"
    command: str | list[str] = "python solution.py"
    startup_config: SandboxStartupConfig | None = None
    mem_limit: str = "1.5g"
    cpu_count: int = 1
    cpu_percent: int | None = None
    cpu_period: int | None = None
    cpu_quota: int | None = None
    seccomp_profile_file: str = "/app/src/worker/seccomp-profile.json"
    network_disabled: bool = True
    network_mode: str = "none"
    ports: Mapping[str, int | list[int] | tuple[str, int] | None] | None = None

    # Internet access control
    allow_internet: bool = False  # Whether to allow internet access
    dns_servers: list[str] | None = None  # Custom DNS servers (None = auto-decide based on allow_internet)
    cap_drop: list[str] | None = None  # Capabilities to drop (None = auto-decide based on allow_internet)
    # Optional Kubernetes node pool target, mapped to node selector/toleration by K8s sandbox.
    # In this cluster, values typically match `component` label (e.g. "sandbox", "app", "ops").
    nodepool: str | None = None

    # K8s image pull policy ("Always", "IfNotPresent", "Never"). Ignored by Docker sandbox.
    image_pull_policy: str = "IfNotPresent"

    # When set, the K8s sandbox attaches this as a readinessProbe on the runner
    # container and uses kubelet readiness (not log-grep) to gate
    # `exit_after_startup=True`. Ignored by Docker sandbox.
    readiness_probe: ReadinessProbeSpec | None = None


class SandboxMetrics(BaseModel):
    execution_time: float = 0
    cpu_usage_seconds: float = 0
    gpu_usage_seconds: float = 0
    network_usage_mb: float = 0  # Total network (rx + tx) for backwards compatibility
    network_rx_mb: float = 0  # Network receive only
    network_tx_mb: float = 0  # Network transmit only
    io_usage_mb: float = 0  # Total I/O (read + write) for backwards compatibility
    io_read_mb: float = 0  # I/O read only
    io_write_mb: float = 0  # I/O write only
    max_memory_mb: float = 0
    max_gpu_memory_mb: float = 0
    max_cpu_percent: float = 0


class SandboxResult(BaseModel):
    sandbox_id: str
    exit_code: int
    timed_out: bool
    startup_time: float
    execution_time: float
    output_path: str
    log_file_name: str
    log_file_path: str
    metrics: SandboxMetrics
