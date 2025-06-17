import asyncio
import contextlib
import os
import random
import socket
import sqlite3
import time
from typing import Any, AsyncGenerator

import numpy as np
import openai
from loguru import logger

from shared import settings
from shared.epistula import make_openai_query
from validator_api.chain.uid_tracker import SQLITE_PATH, CompletionFormat, TaskType, UidTracker, connect_db
from validator_api.deep_research.orchestrator_v2 import MODEL_ID

shared_settings = settings.shared_settings
STEP = 256
TIMEOUT_CALIBRATION = 180


@contextlib.contextmanager
def maybe_acquire_lock(worker_id: str, ttl_sec: int = 600):
    """Try to grab writer lock. If we return True, caller is the writer."""
    with contextlib.closing(connect_db()) as con, con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration_lock(
                id          INTEGER PRIMARY KEY CHECK(id = 1),
                holder      TEXT,
                acquired_at INTEGER
            )"""
        )
        # Clean up stale lock (no update for ttl)
        con.execute(
            """
            DELETE FROM calibration_lock
            WHERE id = 1 AND strftime('%s','now') - acquired_at > ?
        """,
            (ttl_sec,),
        )
        try:
            con.execute(
                "INSERT INTO calibration_lock(id, holder, acquired_at) VALUES (1, ?, strftime('%s','now'))",
                (worker_id,),
            )
            yield True
        except sqlite3.IntegrityError:
            yield False


async def generate_json_query() -> dict[str, Any]:
    async def _generate_prompt() -> str:
        year, era = random.randint(100, 2025), random.choice(["BCE", "CE"])
        year2 = random.randint(600, 2025)
        content = f"""\
Provide a detailed JSON response containing a list the most powerful countries/kingdoms (up to 10) and their rulers in the year {year} {era}.
Each ruler should include the ruler's name and a brief description.
In addition to that, provide a long and detailed report of the most significant discovery in the year {year2} CE, with a description of the discovery, founders biography and its impact on the world.
Answer only with JSON, do not include any other text.

<example input>
countries and their rulers in the year 2020 CE, significant discovery in the year 2021.
</example input>
<example output>
{{
    "countries": [
        {{
            "country": "Some country 1",
            "ruler": {{
                "name": "Ruler's name",
                "description": "Ruler's description."
            }}
        }},
        {{
            "country": "Some country 2",
            "ruler": {{
                "name": "Ruler's name",
                "description": "Ruler's description."
            }}
        }}
    ],
    "discovery": {{
        "year": {year2},
        "description": "Detailed description of the discovery, founders biography and its impact on the world"
    }}
}}
"""
        return content

    messages = [{"role": "system", "content": "You are a historian specializing in world history."}]
    history_len = random.randint(3, 8)
    for idx in range(history_len):
        if idx < history_len:
            messages.append({"role": "assistant", "content": ""})
        content = await _generate_prompt()
        messages.append({"role": "user", "content": content})

    return {
        "model": MODEL_ID,
        "messages": messages,
        "stream": True,
        "task": "InferenceTask",
        "sampling_parameters": {
            "temperature": 0.7,
            "max_new_tokens": 6144,
            "top_p": 0.95,
            "timeout": TIMEOUT_CALIBRATION,
        },
    }


async def _collector(idx: int, response_task: asyncio.Task) -> tuple[list[str], list[float]]:
    """Stream collector - returns **only** the list[str] chunks for one UID."""
    chunks: list[str] = []
    timings: list[float] = []
    timer_start = time.monotonic()
    try:
        stream = await response_task
        if stream is None or isinstance(stream, Exception):
            return chunks, timings

        async for chunk in stream:
            if not chunk.choices:
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                chunks.append(content)
                timings.append(time.monotonic() - timer_start)

    except (openai.APIConnectionError, asyncio.CancelledError):
        # Treat as empty result.
        pass
    except Exception as e:
        logger.error(f"Collector error for miner index {idx}: {e}")
    return chunks, timings


async def _run_single_calibration(uid_tracker: UidTracker) -> None:
    """Execute one complete calibration cycle.

    Args:
        uid_tracker: The in-memory tracker whose statistics are updated.
    """
    uid_tracker.resync()

    all_uids: list[int] = [
        int(uid) for uid in shared_settings.METAGRAPH.uids if shared_settings.METAGRAPH.stake[uid] * 0.05 < 1000
    ]

    # Sample single uid from each coldkey.
    tracked_uids: list[int] = []
    tracked_coldkeys: set[str] = set()
    for uid in all_uids:
        ckey = shared_settings.METAGRAPH.coldkeys[uid]
        if ckey in tracked_coldkeys:
            continue

        tracked_coldkeys.add(ckey)
        tracked_uids.append(uid)

    logger.debug(f"Starting network calibration for {len(tracked_uids)} UIDs.")

    for start in range(0, len(tracked_uids), STEP):
        uids = tracked_uids[start : start + STEP]

        body = await generate_json_query()

        response_tasks: list[asyncio.Task[AsyncGenerator[Any, None] | tuple[list[str], list[float]]]] = [
            asyncio.create_task(
                make_openai_query(
                    shared_settings.METAGRAPH,
                    shared_settings.WALLET,
                    timeout_seconds=TIMEOUT_CALIBRATION,
                    body=body,
                    uid=uid,
                    stream=True,
                )
            )
            for uid in uids
        ]

        collector_tasks: list[asyncio.Task[tuple[list[str], list[float]]]] = [
            asyncio.create_task(_collector(idx, rt)) for idx, rt in enumerate(response_tasks)
        ]
        results: list[tuple[list[str], list[float]]] = await asyncio.gather(*collector_tasks, return_exceptions=False)

        chunks_list: list[list[str]] = [res[0] for res in results]
        timings_list: list[list[float]] = [res[1] for res in results]

        tps_values: list[float] = [
            (len(chks) / tms[-1]) if chks and tms and tms[-1] > 0 else 0.0
            for chks, tms in zip(chunks_list, timings_list)
        ]

        if tps_values:
            mean_tps = np.mean(tps_values)
            logger.debug(
                f"[Calibration] TPS mean: {mean_tps:.2f}, "
                f"min: {min(tps_values):.2f}, "
                f"max: {max(tps_values):.2f}"
            )

        await uid_tracker.score_uid_chunks(
            uids=uids,
            chunks=chunks_list,
            task_name=TaskType.Inference,
            format=CompletionFormat.JSON,
        )

        await asyncio.sleep(15)

    logger.debug(f"Network calibration completed for {len(tracked_uids)} UIDs and {len(tracked_coldkeys)} coldkeys")


async def periodic_network_calibration(
    uid_tracker: UidTracker,
    interval_hours: int = 48,
    write_results: bool | None = None,
):
    """Run calibration at fixed intervals with single-writer semantics.

    The first worker that acquires the SQLite lock becomes the *writer*,
    performs calibration, and persists the tracker via
    :pyfunc:`UidTracker.save_to_sqlite`. Other workers act as *readers*:
    they block until the writer finishes, then load the fresh snapshot with
    :pyfunc:`UidTracker.load_from_sqlite`.

    Args:
        uid_tracker: The tracker instance local to this worker process.
        interval_hours: Interval between global calibration cycles.
        write_results: Deprecated. Retained so existing calls stay valid.
    """
    # Stagger startup to reduce lock contention bursts.
    # await asyncio.sleep(random.randint(0, 120))

    worker_id = f"{socket.gethostname()}:{os.getpid()}"

    while True:
        try:
            with maybe_acquire_lock(worker_id) as is_writer:
                if is_writer:
                    logger.info(f"{worker_id} acquired writer lock")
                    try:
                        await _run_single_calibration(uid_tracker)
                        uid_tracker.save_to_sqlite()
                    finally:
                        with contextlib.closing(sqlite3.connect(SQLITE_PATH)) as con, con:
                            con.execute("DELETE FROM calibration_lock WHERE id = 1")
                        logger.info(f"{worker_id} released writer lock")
                        success_rates = [await sr.success_rate(TaskType.Inference) for sr in uid_tracker.uids.values()]
                        logger.debug(f"Success rate average (writer): {np.mean(success_rates):.2f}")
                else:
                    logger.info(f"{worker_id} acting as reader; waiting for snapshot")
                    while True:
                        await asyncio.sleep(30)
                        with sqlite3.connect(SQLITE_PATH) as con:
                            row = con.execute("SELECT 1 FROM calibration_lock WHERE id = 1").fetchone()
                            if row is None:
                                break

                    uid_tracker.load_from_sqlite()
                    logger.info(f"{worker_id} loaded fresh uid_tracker snapshot")
                    success_rates = [await sr.success_rate(TaskType.Inference) for sr in uid_tracker.uids.values()]
                    logger.debug(f"Success rate average (reader): {np.mean(success_rates):.2f}")
        except BaseException as exc:  # pylint: disable=broad-except
            logger.exception(f"Calibration error on {worker_id}: {exc}")

        logger.debug(f"Waiting {interval_hours:.2f}h before next calibration")
        await asyncio.sleep(interval_hours * 3600)


if __name__ == "__main__":
    # Example usage of the periodic calibration function.
    uid_tracker = UidTracker()
    uid_tracker.resync()
    asyncio.run(periodic_network_calibration(uid_tracker, write_results=True))
