# rl_baseline.py - RL-enabled Battleship player
# Drop-in replacement for baseline.py that uses a trained RL model for shooting.

from typing import Dict, Optional, Any, List
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Import everything from battleship baseline to avoid code duplication
from ..battleship.baseline import (
    DEFAULT_SHIPS,
    BoardRequest,
    BoardResponse,
    NextMoveRequest,
    NextMoveResponse,
    Board,
    RandomShooter,
    generate_board,
)

# Import RLShooter if available
try:
    from .train.agent import RLShooter
except ImportError:
    RLShooter = None


class PlayerSession(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    game_id: str
    size: int
    board: Board
    shooter: Any  # Union[RandomShooter, RLShooter]
    pending_result_for_logging: Optional[Dict] = None
    history: List[Dict] = Field(default_factory=list)


def make_app(model_path: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Battleship RL Player API")
    SESSIONS: Dict[str, PlayerSession] = {}

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/board", response_model=BoardResponse)
    def board(req: BoardRequest):
        """
        Create or return the board for this game_id (idempotent).
        The validator is the *client* and will store the board and run the match.
        """
        size = req.size
        ships_spec = req.ships_spec or DEFAULT_SHIPS
        sess = SESSIONS.get(req.game_id)
        if sess is None:
            board = generate_board(size=size, ships_spec=ships_spec)

            if RLShooter is not None and model_path and os.path.exists(model_path):
                shooter = RLShooter(size=size, model_path=model_path)
            else:
                shooter = RandomShooter(size=size)

            sess = PlayerSession(game_id=req.game_id, size=size, board=board, shooter=shooter)
            SESSIONS[req.game_id] = sess
        return {"game_id": req.game_id, "board": sess.board.to_payload()}

    @app.post("/next-move", response_model=NextMoveResponse)
    def next_move(req: NextMoveRequest):
        """
        Return the next shot coordinate.
        Uses the pending result passed earlier via /result (if any) for logging.
        """
        sess = SESSIONS.get(req.game_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="Unknown game_id")

        # Store result of previous shot for logging
        if req.result is not None:
            sess.pending_result_for_logging = {
                "x": req.result.x,
                "y": req.result.y,
                "hit": req.result.hit,
                "sunk": req.result.sunk,
            }
            sess.history.append({"type": "result", **sess.pending_result_for_logging})

        # Get next shot
        x, y = sess.shooter.next_shot(sess.pending_result_for_logging)
        sess.pending_result_for_logging = None
        sess.history.append({"type": "shot", "x": x, "y": y})
        return {"x": x, "y": y}

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", type=str, default=None, help="Path to trained model checkpoint")
    args = parser.parse_args()
    uvicorn.run(make_app(model_path=args.model), host="0.0.0.0", port=args.port)
