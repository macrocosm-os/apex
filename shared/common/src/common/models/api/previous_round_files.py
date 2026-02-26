from pydantic import BaseModel


class PreviousRoundInputFileEntry(BaseModel):
    task_name: str
    download_url: str
    expires_in_seconds: int


class PreviousRoundInputFilesResponse(BaseModel):
    files: list[PreviousRoundInputFileEntry]
    total_count: int
    limit: int
    offset: int
