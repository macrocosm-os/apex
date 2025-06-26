import time
from dataclasses import dataclass, field

from prompting.tasks.base_task import BaseTextTask
from shared.base import DatasetEntry
from shared.dendrite import DendriteResponseEvent


@dataclass
class ScoringConfig:
    task: BaseTextTask
    response: DendriteResponseEvent
    dataset_entry: DatasetEntry
    block: int
    step: int
    task_id: str
    created_at: float = field(default_factory=time.time)
