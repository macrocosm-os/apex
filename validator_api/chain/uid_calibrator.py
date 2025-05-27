import asyncio
import json
import random
import time
from datetime import datetime
from typing import Any, AsyncGenerator

import numpy as np
import openai
from loguru import logger

from shared import settings
from shared.epistula import make_openai_query
from validator_api.chain.uid_tracker import CompletionFormat, TaskType, UidTracker
from validator_api.deep_research.orchestrator_v2 import MODEL_ID

shared_settings = settings.shared_settings

TIMEOUT_CALIBRATION = 150


async def generate_json_query() -> dict[str, Any]:
    year, era = random.randint(1, 2025), random.choice(["BCE", "CE"])
    year2 = random.randint(600, 2025)
    content = f"""\
Provide a detailed JSON response containing a list the most powerful countries/kingdoms (up to 5) and their rulers in the year {year} {era}.
Each ruler should include the ruler's name and a brief description.
In addition to that, provide a long and detailed report of the most significant discovery in the year {year2} CE, with a description of the discovery, founders and its impact on the world.
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
        "description": "Detailed description of the discovery."
    }}
}}
"""
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a historian specializing in world history."},
            {"role": "user", "content": content},
        ],
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
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                chunks.append(content)
                timings.append(time.monotonic() - timer_start)

    except (openai.APIConnectionError, asyncio.CancelledError):
        # Treat as empty result.
        pass
    except Exception as e:
        logger.exception(f"Collector error for miner index {idx}: {e}")
    return chunks, timings


async def periodic_network_calibration(
    uid_tracker: UidTracker,
    interval_hours: int = 24,
    write_results: bool = True,
):
    """Periodically queries the subnet to assess UID reliability."""
    # TODO: Add docstring.
    STEP = 100
    while True:
        try:
            uid_tracker.resync()
            # Filter out validators.
            all_uids: list[int] = [
                int(uid) for uid in shared_settings.METAGRAPH.uids if shared_settings.METAGRAPH.stake[uid] * 0.05 < 1000
            ]
            logger.debug(f"Starting network calibration for {len(all_uids)} UIDs.")

            for start in range(0, len(all_uids), STEP):
                uids = all_uids[start : start + STEP]

                body = await generate_json_query()

                # Fire off all queries concurrently.
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

                # Collect the streams.
                collector_tasks: list[asyncio.Task[tuple[list[str], list[float]]]] = [
                    asyncio.create_task(_collector(i, rt)) for i, rt in enumerate(response_tasks)
                ]
                results: list[tuple[list[str], list[float]]] = await asyncio.gather(
                    *collector_tasks, return_exceptions=False
                )

                # Split into parallel lists so indices align with `uids`.
                chunks_list: list[list[str]] = [res[0] for res in results]
                timings_list: list[list[float]] = [res[1] for res in results]

                # Tokens per second statistics.
                tps_values: list[float] = []
                for chunks, timings in zip(chunks_list, timings_list):
                    # Need at least one timing point to measure duration.
                    if chunks and timings and timings[-1] > 0:
                        tps_values.append(len(chunks) / timings[-1])
                    else:
                        tps_values.append(0.0)

                if tps_values:
                    mean_tps = np.mean(tps_values)
                    min_tps = min(tps_values)
                    max_tps = max(tps_values)
                    logger.debug(f"Calibration TPS - mean: {mean_tps:.2f}, min: {min_tps:.2f}, max: {max_tps:.2f}")

                # Feed into the tracker.
                await uid_tracker.score_uid_chunks(
                    uids=uids,
                    chunks=chunks_list,
                    task_name=TaskType.Inference,
                    format=CompletionFormat.JSON,
                )

                # Write results to a JSONL file if enabled.
                # TODO: Move to SQLite.
                if write_results:
                    with open("calibration_results.jsonl", "a+") as f:
                        for idx, (uid, tps) in enumerate(zip(uids, tps_values)):
                            success_rate = await uid_tracker.uids[uid].success_rate(TaskType.Inference)
                            result: dict[str, Any] = {
                                "uid": uid,
                                "success_rate": success_rate,
                                "tps": round(tps, 2),
                                "token_amount": len(chunks_list[idx]),
                                "date": datetime.now().isoformat(),
                                "hotkey": uid_tracker.uids[uid].hkey,
                            }
                            f.write(json.dumps(result) + "\n")

                await asyncio.sleep(30)

            logger.debug(f"Network calibration completed for UIDs: {len(all_uids)}")
        except Exception as e:
            logger.error(f"Error during periodic network calibration: {e}")

        logger.debug(f"Waiting for next calibration in {interval_hours:.2f}h")
        await asyncio.sleep(interval_hours * 3600)


if __name__ == "__main__":
    # Example usage of the periodic calibration function.
    uid_tracker = UidTracker()
    uid_tracker.resync()
    asyncio.run(periodic_network_calibration(uid_tracker, write_results=True))
