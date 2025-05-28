import json
import random
from enum import Enum
from typing import Any, Iterable
import sqlite3, json, os, contextlib
from datetime import datetime

from loguru import logger
from pydantic import BaseModel

from shared import settings
from validator_api.deep_research.utils import parse_llm_json

shared_settings = settings.shared_settings

SUCCESS_RATE_MIN = 0.9
MIN_CHUNKS = 514

SQLITE_PATH = os.getenv("UID_TRACKER_DB", "uid_tracker.sqlite")


def connect_db(db_path: str = SQLITE_PATH):
    # Autocommit.
    con = sqlite3.connect(db_path, isolation_level=None)
    # Cheap concurrent reads.
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    return con


class CompletionFormat(str, Enum):
    JSON = "json"
    UNKNOWN = None
    # Not supported yet.
    # YAML = "yaml"
    # XML = "xml"
    # HTML = "html"


class TaskType(str, Enum):
    Inference = "Chain-of-Thought"
    WebRetrieval = "WebRetrieval"
    Unknown = None


class Uid(BaseModel):
    uid: int
    hkey: str
    requests_per_task: dict[TaskType, int]
    success_per_task: dict[TaskType, int]

    async def success_rate(self, task_name: TaskType | str) -> float:
        if not isinstance(task_name, TaskType):
            try:
                task_name = TaskType(task_name)
            except ValueError:
                return 0.0

        if task_name not in self.success_per_task:
            return 0.0

        if task_name not in self.requests_per_task:
            return 0.0

        if self.requests_per_task[task_name] == 0:
            return 0.0

        return self.success_per_task[task_name] / self.requests_per_task[task_name]


class UidTracker(BaseModel):
    uids: dict[int, Uid] = {}

    def model_post_init(self, __context: object) -> None:
        self.resync()

    def resync(self):
        # self.load_from_sqlite()
        hotkeys = shared_settings.METAGRAPH.hotkeys
        for uid in range(int(shared_settings.METAGRAPH.n)):
            if uid in self.uids and hotkeys[uid] == self.uids[uid].hkey:
                continue

            self.uids[uid] = Uid(
                uid=uid,
                hkey=hotkeys[uid],
                requests_per_task={task: 0 for task in TaskType},
                success_per_task={task: 0 for task in TaskType},
            )
        self.save_to_sqlite()

    @staticmethod
    async def body_to_task(body: dict[str, Any]) -> TaskType:
        task_name = body.get("task")
        if body.get("test_time_inference", False) or task_name == "Chain-of-Thought":
            return TaskType.Inference
        elif task_name == "InferenceTask":
            return TaskType.Inference
        elif task_name == "WebRetrieval":
            return TaskType.WebRetrieval
        else:
            return TaskType.Unknown

    async def set_query_attempt(self, uids: list[int] | int, task_name: TaskType | str):
        if not isinstance(task_name, TaskType):
            task_name = TaskType(task_name)
        if not isinstance(uids, Iterable):
            uids = [uids]
        for uid in uids:
            self.uids[uid].requests_per_task[task_name] = self.uids[uid].requests_per_task.get(task_name, 0) + 1
            logger.debug(f"Setting query attempt for UID {uid} and task {task_name}")

    async def set_query_success(self, uids: list[int] | int, task_name: TaskType | str):
        if not isinstance(task_name, TaskType):
            task_name = TaskType(task_name)
        if not isinstance(uids, Iterable):
            uids = [uids]
        for uid in uids:
            self.uids[uid].success_per_task[task_name] = self.uids[uid].success_per_task.get(task_name, 0) + 1
            logger.debug(f"Setting query success for UID {uid} and task {task_name}")

    async def sample_reliable(
        self, task: TaskType | str, amount: int, success_rate: float = SUCCESS_RATE_MIN, add_random_extra: bool = True
    ) -> dict[int, Uid]:
        if not isinstance(task, TaskType):
            try:
                task = TaskType(task)
            except ValueError:
                return []

        uid_success_rates: list[tuple[Uid, float]] = []
        for uid in self.uids.values():
            rate = await uid.success_rate(task)
            uid_success_rates.append((uid, rate))

        # Filter UIDs by success rate.
        reliable_uids: list[Uid] = [uid for uid, rate in uid_success_rates if rate > success_rate]

        # Sample random UIDs with success rate > success_rate.
        sampled_reliable = random.sample(reliable_uids, min(amount, len(reliable_uids)))

        # If not enough UIDs, add the remaining ones with the highest success rate.
        if add_random_extra and len(sampled_reliable) < amount:
            remaining_uids = [(uid, rate) for uid, rate in uid_success_rates if uid not in sampled_reliable]
            # Sort by success rate.
            remaining_uids.sort(key=lambda x: x[1], reverse=True)
            sampled_reliable.extend(uid for uid, _ in remaining_uids[: amount - len(sampled_reliable)])

        return {uid_info.uid: uid_info for uid_info in sampled_reliable}

    async def score_uid_chunks(
        self,
        uids: list[int],
        chunks: list[list[str]],
        task_name: TaskType | str,
        format: CompletionFormat | None,
    ):
        if not isinstance(task_name, TaskType):
            try:
                task_name = TaskType(task_name)
            except ValueError:
                return

        if not isinstance(format, CompletionFormat):
            try:
                format = CompletionFormat(format)
            except ValueError:
                format = CompletionFormat.UNKNOWN

        # Only JSON is supported at this moment.
        if format != CompletionFormat.JSON:
            return

        await self.set_query_attempt(uids, task_name)
        for idx, uid in enumerate(uids):
            if len(chunks[idx]) <= MIN_CHUNKS:
                continue
            completion = "".join(chunks[idx])
            try:
                parse_llm_json(completion, allow_empty=False)
                await self.set_query_success(uid, task_name)
            except json.JSONDecodeError:
                pass

    def save_to_sqlite(self, db_path: str = SQLITE_PATH) -> None:
        """Persist the current UID stats to SQLite.

        Uses an UPSERT (`ON CONFLICT ... DO UPDATE`) so concurrent writers
        cannot collide on the PRIMARY KEY (uid).  A short IMMEDIATE
        transaction guarantees readers never see a partially-written table
        but lets other processes keep reading while we write.
        """
        ts = datetime.utcnow().isoformat(timespec="seconds")

        with contextlib.closing(connect_db(db_path)) as con:
            con.execute("BEGIN IMMEDIATE")            # lock for writing, readers still allowed (WAL mode)
            try:
                # Ensure tables exist.
                con.execute(
                    """
                    CREATE TABLE IF NOT EXISTS uids (
                        uid INTEGER PRIMARY KEY,
                        hkey TEXT NOT NULL,
                        requests_per_task TEXT NOT NULL,
                        success_per_task TEXT NOT NULL
                    )
                    """
                )
                con.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        val TEXT NOT NULL
                    )
                    """
                )

                # Upsert every UID row.
                for uid_obj in self.uids.values():
                    con.execute(
                        """
                        INSERT INTO uids (uid, hkey, requests_per_task, success_per_task)
                        VALUES (?,?,?,?)
                        ON CONFLICT(uid) DO UPDATE SET
                            hkey = excluded.hkey,
                            requests_per_task = excluded.requests_per_task,
                            success_per_task = excluded.success_per_task
                        """,
                        (
                            uid_obj.uid,
                            uid_obj.hkey,
                            json.dumps(uid_obj.requests_per_task),
                            json.dumps(uid_obj.success_per_task),
                        ),
                    )

                # Record last-save timestamp.
                con.execute(
                    """
                    INSERT INTO metadata (key, val)
                    VALUES ('last_saved', ?)
                    ON CONFLICT(key) DO UPDATE SET val = excluded.val
                    """,
                    (ts,),
                )

                con.commit()
            except Exception:
                con.rollback()
                raise

    def load_from_sqlite(self, db_path: str = SQLITE_PATH) -> None:
        with contextlib.closing(connect_db(db_path)) as con:
            rows = con.execute("""
                SELECT uid,hkey,requests_per_task,success_per_task FROM uids
            """).fetchall()
            if not rows:
                return
            self.uids.clear()
            for uid, hkey, req_json, succ_json in rows:
                self.uids[uid] = Uid(
                    uid=uid,
                    hkey=hkey,
                    requests_per_task=json.loads(req_json),
                    success_per_task=json.loads(succ_json),
                )


# TODO: Move to FastAPI lifespan.
uid_tracker = UidTracker()
uid_tracker.resync()
