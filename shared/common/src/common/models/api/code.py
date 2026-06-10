from datetime import datetime
from pydantic import BaseModel

from common.models.api.file import FilePagination


class CodeRequest(BaseModel):
    competition_id: int
    round_number: int | None = None
    hotkey: str
    version: int | None = None
    start_idx: int = 0


class CodeResponse(BaseModel):
    version: int
    language: str
    code: str
    submit_at: datetime
    pagination: FilePagination
    is_binary: bool = False
    # For binary files, the contents are served via a presigned S3 download URL rather than
    # inlined (base64) in `code`, to avoid loading large model weights into the server's memory.
    # When `download_url` is set, `code` is empty and clients should fetch the file from the URL.
    download_url: str | None = None
    expires_in_seconds: int | None = None
