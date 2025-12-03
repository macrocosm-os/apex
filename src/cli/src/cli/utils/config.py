from pathlib import Path
from pydantic import BaseModel


class Config(BaseModel):
    hotkey_file_path: str | None = None
    timeout: float = 60.0

    @classmethod
    def load_config(cls, config_file_path: Path = Path(".apex.config.json")) -> "Config":
        if not config_file_path.exists():
            return cls()
        with open(config_file_path, "r") as f:
            return cls.model_validate_json(f.read())

    def save_config(self, config_file_path: Path = Path(".apex.config.json")) -> None:
        with open(config_file_path, "w") as f:
            f.write(self.model_dump_json())
