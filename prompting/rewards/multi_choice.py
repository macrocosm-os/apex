import json
import re
import time

import numpy as np
from pydantic import Field, model_validator

from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from shared.dendrite import DendriteResponseEvent


class MultiChoiceRewardModel(BaseRewardModel):
    choices: tuple[str, ...] = Field(default=("A", "B", "C", "D"))
    json_penalty: float = Field(default=0.9)
    choice_map: dict[str, str] = Field(default={})

    @model_validator(mode="after")
    def init_choice_map(self):
        self.choice_map = {choice.lower(): choice for choice in self.choices}
        return self

    @property
    def name(self) -> str:
        return "multiple_choice"

    @staticmethod
    def safe_load_json(json_string: str) -> dict[str, float]:
        cleaned_json_string = re.sub(r",(\s*[}\]])", r"\1", json_string.strip())
        cleaned_json_string = re.sub(r'"\s*\n\s*"', r'""', cleaned_json_string)
        try:
            return {k.upper(): v for k, v in json.loads(cleaned_json_string).items()}
        except Exception:
            return None

    def process_predictions(self, predictions: dict[str, float]) -> dict[str, float]:
        if not all(isinstance(value, (int, float)) for value in predictions.values()):
            raise ValueError("Values must be numeric")

        valid_choices = {
            self.choice_map[k.lower()]: float(v) for k, v in predictions.items() if k.lower() in self.choice_map
        }

        if any(v < 0 for v in valid_choices.values()):
            raise ValueError(f"Negative values are not allowed: {valid_choices}")

        total = sum(valid_choices.values())
        if np.isclose(total, 0.0):
            raise ValueError(f"Values sum up to 0, total={total}")

        if not np.isclose(total, 1.0):
            valid_choices = {k: v / total for k, v in valid_choices.items()}

        return {choice: valid_choices.get(choice, 0.0) for choice in self.choices}

    def letter_reward(self, reference: str, completion: str) -> float:
        matches = [word.upper() for word in re.findall(r"\w+", completion) if word.upper() in self.choices]
        return float(matches[-1] == reference.upper()) if matches else 0.0

    def logit_reward(self, reference: str, completion: str) -> float:
        try:
            loaded_json = self.safe_load_json(completion)
            if not loaded_json:
                return None
            valid_choices = self.process_predictions(loaded_json)
            return valid_choices.get(reference.upper(), 0.0)
        except ValueError:
            return None

    async def reward(self, reference: str, response_event: DendriteResponseEvent, **kwargs) -> BatchRewardOutput:
        rewards = []
        timings = []

        for completion in response_event.completions:
            start_time = time.perf_counter()

            reward = self.logit_reward(reference, completion)
            if reward is None:
                reward = self.letter_reward(reference, completion) * self.json_penalty

            timings.append(time.perf_counter() - start_time)
            rewards.append(reward)

        return BatchRewardOutput(rewards=np.asarray(rewards), timings=np.asarray(timings))
