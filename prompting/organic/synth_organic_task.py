from dataclasses import dataclass

from prompting.organic.organic_task import OrganicTask


@dataclass
class SynthOrganicTask(OrganicTask):
    """Task with defined reward and penalty mechanisms for synthetic organic prompts."""

    name = "synthetic-organic"

    reward_definition = [
        dict(name="relevance", weight=0.8),
        dict(name="rouge", ngram="rouge-1", metric="f", weight=0.2),
    ]

    penalty_definition = [
        dict(name="relevance", weight=0.5),
    ]

    cleaning_pipeline = []

    def __init__(self, context: dict, reference: str):
        self.context = context
        self.messages = context["messages"]
        self.roles = context["roles"]
        self.query = context["messages"][-1]
        self.topic = "Organic"
        self.reference = reference
        self.subtopic = ""
        self.tags = [""]
