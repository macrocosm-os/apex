from pydantic import BaseModel, model_validator


class BaseErrorModel(BaseModel):
    error_name: str | None = None
    error_dict: dict | None = None


class SpecVersionError(BaseModel):
    expected_version: int
    actual_version: str
    message: str = "Spec version mismatch"

    @model_validator(mode="after")
    def make_message(self):
        self.message = f"Spec version mismatch. Expected: {self.expected_version}, Received: {self.actual_version}"
        return self
