from pydantic import BaseModel


class Pagination(BaseModel):
    start_idx: int
    count: int
    total: int
    has_more: bool
