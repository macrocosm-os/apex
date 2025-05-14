import sqlite3
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel
import json


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
    result: Optional[List[dict]] = None
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    seq_id INT NOT NULL,
                    chunk TEXT,
                    error TEXT,
                    PRIMARY KEY (seq_id, job_id)
                )
            """
            )
            conn.commit()

    def create_job(self) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        now = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, created_at, updated_at, seq_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, JobStatus.PENDING, now, now, 0),
            )
            conn.commit()
        return job_id

    def get_job(self, job_id: str) -> Optional[JobResult]:
        """Get a job by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ? ORDER BY seq_id ASC", (job_id,))
            rows = cursor.fetchall()

        if rows is None:
            return None

        # Convert the result string to List[dict] if it exists
        result = [{"seq_id": row["seq_id"], "chunk": row["chunk"]} for row in rows if row["chunk"]]
        row = rows[-1]
        return JobResult(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            result=result,
            error=row["error"],
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

    def insert_job_chunk(self, job_id: str, chunk: str, seq_id: int, created_at: str) -> None:
        """Insert a chunk row."""
        choices_list = json.loads(chunk.split("data:")[-1])["choices"]
        content_list = [choice["delta"] for choice in choices_list]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (chunk, status, updated_at, seq_id, job_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (json.dumps(content_list), JobStatus.RUNNING, datetime.now(), seq_id, job_id, created_at),
            )
            conn.commit()

    def insert_job_error(self, job_id: str, error: str, seq_id: int, created_at: str) -> None:
        """Insert an error row."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (chunk, status, updated_at, seq_id, job_id, created_at, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (None, JobStatus.RUNNING, datetime.now(), seq_id, job_id, created_at, error),
            )
            conn.commit()


# Create a singleton instance
job_store = JobStore()


async def process_chain_of_thought_job(
    job_id: str, orchestrator, messages: List[Dict[str, str]], created_at: str
) -> None:
    """Process a chain of thought job in the background with incremental result updates."""

    job_store.update_job_status(job_id, JobStatus.RUNNING)

    seq_id = 1
    async for chunk in orchestrator.run(messages=messages):
        # Immediately store each chunk with its sequence number
        try:
            job_store.insert_job_chunk(job_id=job_id, chunk=chunk, seq_id=seq_id, created_at=created_at)
        except Exception as e:
            # Capture and store any errors encountered during processing
            job_store.insert_job_error(job_id, str(e), seq_id, created_at)
        seq_id += 1

    # Mark the job as completed after all chunks are processed
    job_store.update_job_status(job_id, JobStatus.COMPLETED)
