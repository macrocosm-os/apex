import asyncio
import json
import time
from pathlib import Path
from typing import Literal, TypedDict

import aiosqlite
from loguru import logger

from apex.common.models import MinerDiscriminatorResults


class DiscriminatorItem(TypedDict):
    """Item placed on the queue that represents one discriminator result row."""

    kind: Literal["discriminator"]
    data: MinerDiscriminatorResults


class LoggerDB:
    _COMMIT_FREQ = 60
    _COMMIT_CHANGES = 1000

    def __init__(self, db_path: Path | str = "results.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: asyncio.Queue[DiscriminatorItem | object] = asyncio.Queue(maxsize=10_000)
        self._SHUTDOWN = object()
        self._closing = asyncio.Event()

    async def start_loop(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA wal_autocheckpoint=1000;
                PRAGMA synchronous=NORMAL;
                PRAGMA busy_timeout=5000;

                CREATE TABLE IF NOT EXISTS discriminator_results (
                    query TEXT,
                    generator_hotkey TEXT,
                    generator_result TEXT,
                    generator_score REAL,
                    discriminator_hotkeys TEXT,  -- JSON array of strings
                    discriminator_results TEXT,  -- JSON array of strings
                    discriminator_scores TEXT,  -- JSON array of floats
                    timestamp INTEGER,  -- Unix timestamp when row was added
                    processed INTEGER DEFAULT 0,
                    PRIMARY KEY (query, generator_hotkey)
                );

                CREATE INDEX IF NOT EXISTS idx_discriminator_processed
                ON discriminator_results(processed);
                """
            )
            await db.commit()

            last_commit = time.monotonic()
            last_changes = db.total_changes
            while True:
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=20.0)
                except TimeoutError:
                    if db.total_changes != last_changes:
                        await db.commit()
                        last_commit = time.monotonic()
                        last_changes = db.total_changes
                    continue

                if item is self._SHUTDOWN:
                    self._queue.task_done()
                    await db.commit()
                    await db.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    break

                try:
                    await self.add_entry(db, item=item)
                except Exception:
                    await db.rollback()
                finally:
                    self._queue.task_done()

                commit_changes = (db.total_changes - last_changes) >= self._COMMIT_CHANGES
                commit_timer = time.monotonic() - last_commit >= self._COMMIT_FREQ
                if commit_changes or commit_timer:
                    logger.debug(f"Commiting scores to the {self.db_path}")
                    await db.commit()
                    last_commit = time.monotonic()
                    last_changes = db.total_changes

    async def add_entry(self, db: aiosqlite.Connection, item: DiscriminatorItem | object) -> None:
        if isinstance(item, dict) and item.get("kind") == "discriminator":
            row: MinerDiscriminatorResults = item["data"]

            await db.execute(
                """
                INSERT INTO discriminator_results (
                    query, generator_hotkey, generator_result, generator_score,
                    discriminator_hotkeys, discriminator_results, discriminator_scores, timestamp
                ) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(query, generator_hotkey) DO UPDATE SET
                    generator_result = excluded.generator_result,
                    generator_score = excluded.generator_score,
                    discriminator_hotkeys = excluded.discriminator_hotkeys,
                    discriminator_results = excluded.discriminator_results,
                    discriminator_scores = excluded.discriminator_scores,
                    timestamp = excluded.timestamp
                """,
                (
                    row.query,
                    row.generator_hotkey,
                    row.generator_result,
                    row.generator_score,
                    json.dumps(row.discriminator_hotkeys),
                    json.dumps(row.discriminator_results),
                    json.dumps(row.discriminator_scores),
                    int(time.time()),
                ),
            )

    async def log(self, row: MinerDiscriminatorResults) -> None:
        if self._closing.is_set():
            logger.error("Database is shutting down")
            return

        await self._queue.put({"kind": "discriminator", "data": row})

    async def shutdown(self) -> None:
        self._closing.set()
        await self._queue.join()
        await self._queue.put(self._SHUTDOWN)
