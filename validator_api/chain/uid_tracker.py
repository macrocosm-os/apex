import asyncio
import contextlib
import json
import os
import random
import sqlite3
from datetime import datetime
from enum import Enum
from typing import Any, Iterable

from loguru import logger
from pydantic import BaseModel

from shared import settings
from validator_api.deep_research.utils import parse_llm_json

shared_settings = settings.shared_settings

SUCCESS_RATE_MIN = 0.95
MIN_CHUNKS = 514

SQLITE_PATH = os.getenv("UID_TRACKER_DB", "uid_tracker.sqlite")


def connect_db(db_path: str = SQLITE_PATH):
    con = sqlite3.connect(db_path, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS uids (
            uid TEXT PRIMARY KEY,
            hkey TEXT NOT NULL,
            requests_per_task TEXT NOT NULL,
            success_per_task TEXT NOT NULL
        )
    """
    )
    return con


class CompletionFormat(str, Enum):
    JSON = "json"
    STR = "str"
    UNKNOWN = None
    # Not supported yet.
    # YAML = "yaml"
    # XML = "xml"
    # HTML = "html"


class TaskType(str, Enum):
    Inference = "InferenceTask"
    WebRetrieval = "WebRetrieval"
    Unknown = None


class Uid(BaseModel):
    uid: int
    hkey: str
    all_uids: list[int]
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
    bind_uid_coldkey: bool = True
    uids: dict[int, Uid] = {}
    write_frequency: float = 10
    track_counter: float = 0

    def model_post_init(self, __context: object) -> None:
        self.resync()

    def resync(self):
        self.load_from_sqlite()
        hotkeys = shared_settings.METAGRAPH.hotkeys
        for uid in range(int(shared_settings.METAGRAPH.n)):
            if uid in self.uids and hotkeys[uid] == self.uids[uid].hkey:
                self.uids[uid].all_uids = self.get_all_coldkey_uids(uid)
                continue

            self.uids[uid] = Uid(
                uid=uid,
                hkey=hotkeys[uid],
                all_uids=self.get_all_coldkey_uids(uid),
                requests_per_task={task: 0 for task in TaskType},
                success_per_task={task: 0 for task in TaskType},
            )
        # self.save_to_sqlite()

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

    def get_all_coldkey_uids(self, uid: int) -> list[int]:
        """Get all UIDs assosiated with given uid and its coldkey."""
        # Metagraph is lazily initialized.
        coldkeys = shared_settings.METAGRAPH.coldkeys
        uid_ckey = coldkeys[uid]
        all_uids = [related_uid for related_uid, ckey in enumerate(coldkeys) if ckey == uid_ckey]
        return all_uids

    async def success_rate_per_coldkey(self) -> dict[str, float]:
        coldkey_rate: dict[str, float] = {}
        for sr in self.uids.values():
            coldkey = shared_settings.METAGRAPH.coldkeys[sr.uid]
            if coldkey in coldkey_rate:
                coldkey_rate[coldkey] = round(sr.success_rate(TaskType.Inference), 2)
        return coldkey_rate

    async def set_query_attempt(self, uids: list[int] | int, task_name: TaskType | str):
        """Set query attempt, in case of success `set_query_success` should be called for the given uid."""
        if not isinstance(task_name, TaskType):
            task_name = TaskType(task_name)

        if not isinstance(uids, Iterable):
            uids = [uids]

        for uid in uids:
            if self.bind_uid_coldkey:
                for uid_assosiated in self.uids[uid].all_uids:
                    # Update all assosiated UIDs with given coldkey.
                    value = self.uids[uid_assosiated].requests_per_task.get(task_name, 0) + 1
                    self.uids[uid_assosiated].requests_per_task[task_name] = value
                    logger.debug(f"Setting query attempt for task {task_name} and UIDs {self.uids[uid].all_uids}")
            else:
                self.uids[uid].requests_per_task[task_name] = self.uids[uid].requests_per_task.get(task_name, 0) + 1
                logger.debug(f"Setting query attempt for task {task_name} and UID {uid}")

    async def set_query_success(self, uids: list[int] | int, task_name: TaskType | str):
        if not isinstance(task_name, TaskType):
            task_name = TaskType(task_name)

        if not isinstance(uids, Iterable):
            uids = [uids]

        self.track_counter += 1
        if self.track_counter % self.write_frequency == 0:
            await asyncio.to_thread(self.save_to_sqlite)
            coldkey_rate = await self.success_rate_per_coldkey()
            logger.debug(f"Success rate per coldkey: {coldkey_rate}")

        if self.track_counter % self.write_frequency == self.write_frequency - 1:
            await asyncio.to_thread(self.resync)

        for uid in uids:
            if self.bind_uid_coldkey:
                for uid_assosiated in self.uids[uid].all_uids:
                    # Update all assosiated UIDs with given coldkey.
                    value = self.uids[uid_assosiated].success_per_task.get(task_name, 0) + 1
                    self.uids[uid_assosiated].success_per_task[task_name] = value
                logger.debug(f"Setting query success for task {task_name} and UIDs {self.uids[uid].all_uids}")
            else:
                self.uids[uid].success_per_task[task_name] = self.uids[uid].success_per_task.get(task_name, 0) + 1
                logger.debug(f"Setting query success for task {task_name} and UID {uid}")

    async def sample_reliable(
        self, task: TaskType | str, amount: int, success_rate: float = SUCCESS_RATE_MIN, add_random_extra: bool = True
    ) -> dict[int, Uid]:
        if not isinstance(task, TaskType):
            try:
                task = TaskType(task)
            except ValueError:
                return {}

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
        min_chunks: int = 514,
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

        if format == CompletionFormat.JSON:
            await self.set_query_attempt(uids, task_name)
            for idx, uid in enumerate(uids):
                if len(chunks[idx]) <= min_chunks:
                    continue
                completion = "".join(chunks[idx])
                try:
                    parse_llm_json(completion, allow_empty=False)
                    await self.set_query_success(uid, task_name)
                except json.JSONDecodeError:
                    pass
        elif format == CompletionFormat.STR:
            await self.set_query_attempt(uids, task_name)
            for idx, uid in enumerate(uids):
                if len(chunks[idx]) <= min_chunks:
                    continue
                await self.set_query_success(uid, task_name)
        else:
            logger.error(f"Unsupported completion format: {format}")
            return

    def save_to_sqlite(self, db_path: str = SQLITE_PATH) -> None:
        """Persist the current UID stats to SQLite.

        Uses an UPSERT (`ON CONFLICT ... DO UPDATE`) so concurrent writers
        cannot collide on the PRIMARY KEY (uid). A short IMMEDIATE
        transaction guarantees readers never see a partially-written table
        but lets other processes keep reading while we write.
        """
        ts = datetime.utcnow().isoformat(timespec="seconds")

        with contextlib.closing(connect_db(db_path)) as con:
            # Lock for writing, readers still allowed (WAL mode).
            con.execute("BEGIN IMMEDIATE")
            try:
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
                logger.debug("Successfully updated UID tracker database")
            except Exception:
                con.rollback()
                logger.debug("Failed to update UID tracker database")
                raise

    def load_from_sqlite(self, db_path: str = SQLITE_PATH) -> None:
        with contextlib.closing(connect_db(db_path)) as con:
            try:
                rows = con.execute("SELECT uid, hkey, requests_per_task, success_per_task FROM uids").fetchall()
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    # Empty DB, nothing to load.
                    return
                raise

            if not rows:
                return

            self.uids.clear()
            for uid, hkey, req_json, succ_json in rows:
                self.uids[int(uid)] = Uid(
                    uid=uid,
                    hkey=hkey,
                    all_uids=self.get_all_coldkey_uids(int(uid)),
                    requests_per_task=json.loads(req_json),
                    success_per_task=json.loads(succ_json),
                )
        logger.debug("Loaded from database")


# TODO: Move to FastAPI lifespan.
uid_tracker = UidTracker()
uid_tracker.resync()
