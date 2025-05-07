import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    """Enum for job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResult(BaseModel):
    """Model for job result."""

    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    result: Optional[List[str]] = None
    error: Optional[str] = None


class JobStore:
    """Store for background jobs."""

    def __init__(self):
        """Initialize the job store."""
        self._jobs: Dict[str, JobResult] = {}

    def delete_job(self, job_id: str) -> None:
        """Delete a job by its ID."""
        if job_id in self._jobs:
            del self._jobs[job_id]

    def create_job(self) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        now = datetime.now()
        self._jobs[job_id] = JobResult(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        return job_id

    def get_job(self, job_id: str) -> Optional[JobResult]:
        """Get a job by its ID."""
        return self._jobs.get(job_id)

    def update_job_status(self, job_id: str, status: JobStatus) -> None:
        """Update the status of a job."""
        if job_id in self._jobs:
            self._jobs[job_id].status = status
            self._jobs[job_id].updated_at = datetime.now()

    def update_job_result(self, job_id: str, result: List[str]) -> None:
        """Update the result of a job."""
        if job_id in self._jobs:
            self._jobs[job_id].result = result
            self._jobs[job_id].status = JobStatus.COMPLETED
            self._jobs[job_id].updated_at = datetime.now()

    def update_job_error(self, job_id: str, error: str) -> None:
        """Update the error of a job."""
        if job_id in self._jobs:
            self._jobs[job_id].error = error
            self._jobs[job_id].status = JobStatus.FAILED
            self._jobs[job_id].updated_at = datetime.now()


# Create a singleton instance
job_store = JobStore()


async def process_chain_of_thought_job(job_id: str, orchestrator, messages: List[Dict[str, str]]) -> None:
    """Process a chain of thought job in the background."""
    try:
        job_store.update_job_status(job_id, JobStatus.RUNNING)

        # Collect all chunks from the orchestrator
        chunks = []
        async for chunk in orchestrator.run(messages=messages):
            chunks.append(chunk)

        # Update the job with the result
        job_store.update_job_result(job_id, chunks)
    except Exception as e:
        # Update the job with the error
        job_store.update_job_error(job_id, str(e))
