import asyncio
import json
import time
from pathlib import Path
from typing import Literal, TypedDict

import aiosqlite

from apex.common.models import MinerDiscriminatorResults


class _DiscriminatorItem(TypedDict):
    """Item placed on the queue that represents one discriminator result row."""

    kind: Literal["discriminator"]
    data: MinerDiscriminatorResults


class LoggerDB:
    def __init__(self, db_path: Path | str = "results.db"):
        self.db_path = Path(db_path)
        self._queue: asyncio.Queue[_DiscriminatorItem | object] = asyncio.Queue(maxsize=10_000)
        self._SHUTDOWN = object()

    async def start_loop(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(
                """
                PRAGMA journal_mode = WAL;

                -- Results coming back from a discriminator step.
                -- Store *all* fields from DiscriminatorQueryResults; list fields are serialized as JSON.
                CREATE TABLE IF NOT EXISTS discriminator_results (
                    query                 TEXT,
                    generator_hotkey      TEXT,
                    generator_result      TEXT,
                    generator_score       REAL,
                    discriminator_hotkeys TEXT,  -- JSON array of strings
                    discriminator_results TEXT,  -- JSON array of strings
                    discriminator_scores  TEXT,  -- JSON array of floats
                    timestamp            INTEGER,  -- Unix timestamp when row was added
                    processed            INTEGER DEFAULT 0,
                    PRIMARY KEY (query, generator_hotkey)
                );
                """
            )
            await db.commit()

            while True:
                item = await self._queue.get()
                if item is self._SHUTDOWN:
                    break

                if isinstance(item, dict) and item.get("kind") == "discriminator":
                    row: MinerDiscriminatorResults = item["data"]

                    await db.execute(
                        "INSERT OR REPLACE INTO discriminator_results VALUES (?,?,?,?,?,?,?,?,0)",
                        (
                            row.query,
                            row.generator_hotkey,
                            row.generator_result,
                            row.generator_score,
                            json.dumps(row.discriminator_hotkeys),
                            json.dumps(row.discriminator_results),
                            json.dumps(row.discriminator_scores),
                            int(time.time()),  # Current Unix timestamp
                        ),
                    )

                # flush every 1 000 rows or on demand
                if self._queue.empty() or db.total_changes % 1000 == 0:
                    await db.commit()
                    await db.execute("PRAGMA wal_checkpoint(FULL);")
                self._queue.task_done()

    async def log(self, row: MinerDiscriminatorResults) -> None:
        await self._queue.put({"kind": "discriminator", "data": row})

    async def shutdown(self) -> None:
        await self._queue.put(self._SHUTDOWN)
