from pydantic import BaseModel
from typing import Optional


class TournamentRequest(BaseModel):
    competition_id: int
    round_number: Optional[int] = None


class TournamentWinner(BaseModel):
    submission_id: int
    hotkey: str
    coldkey: Optional[str] = None


class TournamentParticipant(BaseModel):
    submission_id: int
    hotkey: str
    coldkey: Optional[str] = None
    seed: int
    current_bracket_round: int
    alive: bool
    rounds_survived: int
    eliminated_in_round: Optional[int] = None
    had_bye: bool = False


class TournamentMatchSide(BaseModel):
    submission_id: int
    hotkey: str
    seed: int


class TournamentMatch(BaseModel):
    bracket_round: int
    submission_a: TournamentMatchSide
    submission_b: Optional[TournamentMatchSide] = None
    winner_submission_id: Optional[int] = None
    outcome: str  # "won" | "tied" | "bye" | "no_result"


class TournamentResponse(BaseModel):
    competition_id: int
    round_number: int
    round_id: int
    format: str  # "single_elim" (future: "round_robin")
    state: str  # round.state
    bracket_size: int
    total_bracket_rounds: int
    current_bracket_round: Optional[int] = None
    winner: Optional[TournamentWinner] = None
    participants: list[TournamentParticipant]
    matches: list[TournamentMatch]


class TournamentCache(BaseModel):
    response: TournamentResponse
