# mypy: disable-error-code=no-redef
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field


class ConfigClass(BaseModel):
    class_loc: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class Config(BaseModel):
    chain: ConfigClass = Field(default_factory=ConfigClass)
    websearch: ConfigClass = Field(default_factory=ConfigClass)
    logger_db: ConfigClass = Field(default_factory=ConfigClass)
    weight_syncer: ConfigClass = Field(default_factory=ConfigClass)
    miner_sampler: ConfigClass = Field(default_factory=ConfigClass)
    miner_scorer: ConfigClass = Field(default_factory=ConfigClass)
    llm: ConfigClass = Field(default_factory=ConfigClass)
    deep_research: ConfigClass = Field(default_factory=ConfigClass)
    pipeline: ConfigClass = Field(default_factory=ConfigClass)
    logging: ConfigClass = Field(default_factory=ConfigClass)

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        with Path(path).open("r") as file:
            config_data = yaml.safe_load(file)

        config = cls.model_validate(config_data)
        return config
