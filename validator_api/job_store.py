import sqlite3
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
    """Store for background jobs using SQLite database."""

    def __init__(self, db_path: str = "jobs.db"):
        """Initialize the job store with SQLite database."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database and create the jobs table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    result TEXT,
                    error TEXT
                )
            """)
            conn.commit()

    def delete_job(self, job_id: str) -> None:
        """Delete a job by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.commit()

    def create_job(self) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        now = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (job_id, JobStatus.PENDING, now, now),
            )
            conn.commit()
        return job_id

    def get_job(self, job_id: str) -> Optional[JobResult]:
        """Get a job by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            )
            row = cursor.fetchone()
            
        if row is None:
            return None
            
        # Convert the result string to List[str] if it exists
        result = eval(row['result']) if row['result'] is not None else None
        
        return JobResult(
            job_id=row['job_id'],
            status=JobStatus(row['status']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            result=result,
            error=row['error']
        )

    def update_job_status(self, job_id: str, status: JobStatus) -> None:
        """Update the status of a job."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs 
                SET status = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, datetime.now(), job_id),
            )
            conn.commit()

    def update_job_result(self, job_id: str, result: List[str]) -> None:
        """Update the result of a job."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs 
                SET result = ?, status = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (str(result), JobStatus.COMPLETED, datetime.now(), job_id),
            )
            conn.commit()

    def update_job_error(self, job_id: str, error: str) -> None:
        """Update the error of a job."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs 
                SET error = ?, status = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (error, JobStatus.FAILED, datetime.now(), job_id),
            )
            conn.commit()


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