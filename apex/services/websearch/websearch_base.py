from pydantic import BaseModel


class Website(BaseModel):
    query: str
    url: str | None
    content: str | None
    score: float | None = None
    title: str = ""
    response_time: float | None = None


class WebSearchBase:
    async def search(self, query: str, max_results: int) -> list[Website]:
        raise NotImplementedError
