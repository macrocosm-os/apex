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
    allow_internet: bool = False


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


class JobFile(BaseModel):
    submission_id: int
    file_type: str
    file_name: str
    file_content: str


class JobReject(BaseModel):
    submission_id: int
    reason: str
