from pydantic import BaseModel
from enum import Enum
from common.models.sandbox import SandboxStartupConfig


class FileType(str, Enum):
    CODE = "code"
    LOG = "log"
    HISTORY = "history"


class JobType(str, Enum):
    EVALUATION = "evaluation"
    ROUND_GENERATION = "round_generation"


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
    round_generation: RoundGenerationPayload | None = None


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
    round_generation_round_length_in_days: float | None = None


class JobFile(BaseModel):
    submission_id: int
    file_type: str
    file_name: str
    file_content: str


class JobReject(BaseModel):
    submission_id: int
    reason: str
