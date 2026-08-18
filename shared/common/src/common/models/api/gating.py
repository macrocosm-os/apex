from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class GatedReason(str, Enum):
    """Machine-readable reasons a reveal-gated resource returned 403."""

    ROUND_NOT_COMPLETED = "round_not_completed"
    CODE_NOT_REVEALED = "code_not_revealed"
    CODE_NOT_ACCESSIBLE = "code_not_accessible"


class GatedResourceError(BaseModel):
    """Standard 403 `detail` body for reveal-gated resources.

    Every gated 403 carries the same shape so the FE can branch on `reason`
    and render "locked until" states without parsing prose. Fields that don't
    apply to a given gate stay None rather than being dropped, keeping the
    envelope stable across endpoints.
    """

    reason: GatedReason
    message: str
    round_state: Optional[str] = None
    round_end_at: Optional[datetime] = None
    reveal_at: Optional[datetime] = None

    def as_detail(self) -> dict:
        """JSON-safe dict for HTTPException(detail=...)."""
        return self.model_dump(mode="json")
