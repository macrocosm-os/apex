import json
from datetime import datetime
from pathlib import Path

from apex.common.models import MinerDiscriminatorResults, MinerGeneratorResults


class LoggerLocal:
    def __init__(self, filepath: str = "debug/logs.jsonl"):
        self._debug_file_path = Path(filepath)
        self._debug_file_path.parent.mkdir(exist_ok=True)

    async def log(
        self,
        query: str,
        ground_truth: int,
        reference: str | None,
        generator_results: MinerGeneratorResults | None,
        discriminator_results: MinerDiscriminatorResults | None,
    ) -> None:
        day = datetime.now().strftime("%Y-%m-%d")
        filepath = Path(f"{self._debug_file_path.with_suffix('')}-{day}.jsonl")
        record: dict[str, str | int | list[str] | list[float] | None] = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "ground_truth": ground_truth,
            "reference": reference,
            "generators": generator_results.generator_results if generator_results else [],
            "generator_hotkeys": generator_results.generator_hotkeys if generator_results else [],
            "discriminator_results": discriminator_results.discriminator_results if discriminator_results else [],
            "discriminator_scores": discriminator_results.discriminator_scores if discriminator_results else [],
            "generator_hotkey": discriminator_results.generator_hotkey if discriminator_results else "",
        }

        with filepath.open("a+") as fh:
            fh.write(f"{json.dumps(record)}\n")
