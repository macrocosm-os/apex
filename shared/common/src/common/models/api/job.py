from pydantic import BaseModel
from enum import Enum
from common.models.sandbox import SandboxStartupConfig


class FileType(str, Enum):
    CODE = "code"
    LOG = "log"
    HISTORY = "history"
    ONNX = "onnx"


class JobType(str, Enum):
    EVALUATION = "evaluation"
    ROUND_GENERATION = "round_generation"
    ONNX_CONVERSION = "onnx_conversion"
    SCREEN = "screen"  # Layer-2 behavioural screening in the competition's own screen image


class RoundGenerationPayload(BaseModel):
    request_id: str
    competition_id: int
    competition_pkg: str
    round_number: int
    generator_script_path: str
    generator_module: str
    generator_class_name: str
    generator_args: dict = {}
    round_length_in_days: float | None = None
    nodepool: str | None = None
    startup_config: SandboxStartupConfig | None = None
    # Optional sandbox memory limit (e.g. "13Gi") for heavy round generation such
    # as text_clustering bake mode (mpnet + UMAP + HDBSCAN peaks ~12GB). When None,
    # the worker leaves the SandboxRunRules default (1.5g), which is fine for
    # cheap generators (pool-pick, energy_arbitrage). Set from the competition's
    # input_data_generator_args.sandbox_mem_limit.
    sandbox_mem_limit: str | None = None
    # Optional sandbox CPU core count for heavy round generation. text_clustering
    # bake (mpnet + UMAP + HDBSCAN) is CPU-bound; on the default 1 core a 3-subset
    # bake takes ~45-60 min (UMAP is the bottleneck) and risks the round-gen
    # timeout. When None, the worker leaves the SandboxRunRules default (1). Set
    # from input_data_generator_args.sandbox_cpu_count. The baker also caps native
    # thread pools to the bake_num_threads knob (BakeConfig) — raise that in lockstep.
    sandbox_cpu_count: int | None = None
    # Sandbox kill-timer override in seconds, validated by the scheduler from
    # input_data_generator_args.round_generation_timeout_seconds. The scheduler
    # uses the same value for its result-poll deadline, so both sides of the
    # timeout always agree. When None, the worker falls back to its
    # ROUND_GENERATION_TIMEOUT env default (a spec-resolved timeout still takes
    # precedence on the spec-driven path).
    timeout_seconds: int | None = None
    # Explicit, DB-driven grants for the round-generation sandbox. Set from the
    # competition's input_data_generator_args (inject_secrets /
    # round_generation_allow_internet). Fail-safe defaults: no secrets, no network
    # (the worker derives network_disabled = not allow_internet) — only a
    # competition-metadata change (Neon or the admin endpoint) can grant access,
    # never code drift alone.
    inject_secrets: bool = False
    allow_internet: bool = False
    # Optional sandbox image variant (e.g. "groundtruth") for competitions whose
    # round generation runs a different image than miner eval. Maps to tag
    # sb-{env}-{pkg}-{variant}-{sha} + Dockerfile.{variant}. Set from
    # input_data_generator_args.round_generation_image_variant. None -> the
    # competition's default (miner) image.
    image_variant: str | None = None


class OnnxConversionPayload(BaseModel):
    request_id: str
    competition_id: int
    competition_pkg: str
    submission_id: int
    hotkey: str
    round_number: int
    code_path: str  # S3 key of the source .pt model
    grid_size: int = 32
    converter_script_path: str = "/app/convert_to_onnx.py"
    nodepool: str | None = None
    startup_config: SandboxStartupConfig | None = None


class JobResponse(BaseModel):
    job_type: JobType = JobType.EVALUATION
    submission_id: list[int] = []
    competition_id: int = 0
    competition_name: str = ""
    competition_pkg: str = ""
    round_number: int = 0
    hotkey: list[str] = []
    version: list[int] = []
    input_data: dict = {}
    language: list[str] = []
    submit_metadata: list[dict] = []
    # S3 keys, aligned with `submission_id`; the worker fetches these itself. `raw_code` is the
    # legacy inlined form, kept for rolling-deploy compat with older orchestrators.
    code_paths: list[str] = []
    raw_code: list[str] = []
    # Per-submission cached screening verdict ("passed"/"failed"/None)
    screening_status: list[str | None] = []
    # Which competition_spec_versions row (if any) this job runs under. NULL for
    # legacy EVAL_REGISTRY-driven jobs; set once a spec-driven path is active for
    # the competition. Lets us attribute "which spec ran this job" for replay/audit.
    spec_version_id: int | None = None
    round_generation: RoundGenerationPayload | None = None
    onnx_conversion: OnnxConversionPayload | None = None


class JobResults(BaseModel):
    job_type: JobType = JobType.EVALUATION
    submission_id: int | None = None
    eval_metadata: dict = {}
    eval_error: str | None = None
    eval_time_in_seconds: float | None = None
    eval_raw_score: float | None = None
    eval_score: float | None = None
    eval_at: str | None = None
    round_generation_request_id: str | None = None
    round_generation_tasks: list[dict] | None = None
    round_generation_sandbox_data: dict | None = None
    round_generation_round_length_in_days: float | None = None
    onnx_conversion_request_id: str | None = None
    # Base64-encoded ONNX file bytes; None if conversion failed.
    onnx_conversion_payload_b64: str | None = None
    # Layer-2 screen verdict: "passed" / "failed" (with an optional reason). Set for SCREEN jobs.
    screening_status: str | None = None
    screening_reason: str | None = None


class JobFile(BaseModel):
    submission_id: int
    file_type: str
    file_name: str
    file_content: str


class JobReject(BaseModel):
    submission_id: int
    reason: str
