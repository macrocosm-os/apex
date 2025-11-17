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


class SandboxRunRules(BaseModel):
    sandbox_index: int = 0
    run_timeout_in_seconds: int = 60
    filename: str = "solution.py"
    command: str | list[str] = "python solution.py"
    mem_limit: str = "1g"
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


class SandboxMetrics(BaseModel):
    execution_time: float = 0
    cpu_usage_seconds: float = 0
    gpu_usage_seconds: float = 0
    network_usage_mb: float = 0
    io_usage_mb: float = 0
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
