from pydantic import BaseModel


class FilePagination(BaseModel):
    start_idx: int
    end_idx: int
    lines: int
    next_start_idx: int | None


class ChunkedFileData(BaseModel):
    data: str
    pagination: FilePagination
