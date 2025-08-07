from pydantic import BaseModel


class QueryTask(BaseModel):
    """Represents one unit of work for the pipeline."""

    query_id: str
    # None means generate automatically.
    query: str | None


class MinerGeneratorResults(BaseModel):
    query: str
    generator_hotkeys: list[str]
    generator_results: list[str]


class MinerDiscriminatorResults(BaseModel):
    query: str
    generator_hotkey: str
    generator_result: str
    generator_score: float
    discriminator_hotkeys: list[str]
    discriminator_results: list[str]
    discriminator_scores: list[float]
    timestamp: int
