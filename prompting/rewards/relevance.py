import asyncio
import time
from typing import ClassVar, Optional

import numpy as np
from angle_emb import AnglE
from pydantic import ConfigDict
from scipy import spatial

from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from shared import settings
from shared.dendrite import DendriteResponseEvent

shared_settings = settings.shared_settings


class RelevanceRewardModel(BaseRewardModel):
    threshold: Optional[float] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Lazy singleton.
    _embedding_model: ClassVar[Optional[AnglE]] = None
    _model_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def _get_model(cls) -> AnglE:
        """Lazy-load AnglE once; afterwards just return the cached instance."""
        if cls._embedding_model is not None:
            return cls._embedding_model

        async with cls._model_lock:
            if cls._embedding_model is None:
                loop = asyncio.get_running_loop()
                model = await loop.run_in_executor(
                    None,
                    lambda: AnglE.from_pretrained(
                        "WhereIsAI/UAE-Large-V1",
                        pooling_strategy="cls",
                        device=shared_settings.NEURON_DEVICE,
                    ).to(shared_settings.NEURON_DEVICE),
                )
                cls._embedding_model = model

        return cls._embedding_model

    async def reward(
        self,
        reference: str,
        response_event: DendriteResponseEvent,
        model_manager=None,
        **kwargs,
    ) -> BatchRewardOutput:
        """Cosine-similarity reward (0–1) between `reference` and each completion."""
        if not reference:
            raise ValueError("Reference is empty")

        model = await self._get_model()

        ref_vec = model.encode(reference, to_numpy=True).flatten()
        empty_vec = model.encode("", to_numpy=True).flatten()
        baseline = 1 - spatial.distance.cosine(ref_vec, empty_vec)

        rewards, timings = [], []
        for text in response_event.completions:
            if not text:
                rewards.append(0.0)
                timings.append(0.0)
                continue

            t0 = time.time()
            vec = model.encode(text, to_numpy=True).flatten()
            raw = 1 - spatial.distance.cosine(ref_vec, vec)
            rewards.append(raw - baseline)
            timings.append(time.time() - t0)

        return BatchRewardOutput(
            rewards=np.clip(rewards, 0, 1),
            timings=np.array(timings),
            threshold=self.threshold,
        )