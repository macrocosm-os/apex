from typing import Dict, Set, Tuple, Optional, Any
from textual.widgets import Static
from textual.containers import Vertical, Horizontal
from textual.binding import Binding
from textual.message import Message


Coord = Tuple[int, int]


class Ship:
    """Represents a ship on the board."""

    def __init__(self, name: str, length: int, cells: Set[Coord]):
        self.name = name
        self.length = length
        self.cells = cells
        self.hits: Set[Coord] = set()

    def is_sunk(self) -> bool:
        return self.cells == self.hits


class BoardManager:
    """Own board model. Coordinates are 0‑based (x, y). x = col, y = row."""

    def __init__(self, size: int = 10, ships_spec: Optional[Dict[str, int]] = None):
        self.size = size
        self.ships: Dict[str, Ship] = {}
        self.occupied: Dict[Coord, str] = {}  # cell -> ship name
        self.misses: Set[Coord] = set()  # track missed shots

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def receive_shot(self, x: int, y: int) -> Tuple[bool, Optional[str]]:
        """
        Process a shot at (x, y) on *this* board.
        Returns (hit, sunk_ship_name). sunk_ship_name is None unless the shot just sank a ship.
        """
        if not self.in_bounds(x, y):
            raise ValueError("Shot is out of bounds")
        name = self.occupied.get((x, y))
        if not name:
            self.misses.add((x, y))  # record the miss
            return False, None
        ship = self.ships[name]
        ship.hits.add((x, y))
        if ship.is_sunk():
            return True, ship.name
        return True, None

    def all_ships_sunk(self) -> bool:
        return all(s.is_sunk() for s in self.ships.values())

    def render(self, reveal: bool = True) -> str:
        """String visualization using Rich markup. If reveal=False, ships are hidden (opponent view)."""
        grid = [["░" for _ in range(self.size)] for _ in range(self.size)]
        # place ships
        for (x, y), name in self.occupied.items():
            n = name[0] if name != "Cruiser" else "R"  # rename cruiser to 'R' for readability
            grid[y][x] = n if reveal else "░"
        # mark hits
        for ship in self.ships.values():
            for x, y in ship.hits:
                grid[y][x] = "X"
        # mark misses
        for x, y in self.misses:
            grid[y][x] = "▒"
        # to string with headers
        header = "   " + " ".join(f"{x:2d}" for x in range(self.size))
        rows = [header]
        for y in range(self.size):
            colored_cells = []
            for c in grid[y]:
                if c == "X":
                    colored_cells.append(f"[red]{c:>2}[/red]")
                elif c in ("░", "▒"):
                    colored_cells.append(f"[blue]{c:>2}[/blue]")
                else:
                    colored_cells.append(f"[yellow]{c:>2}[/yellow]")
            rows.append(f"{y:2d} " + " ".join(colored_cells))
        return "\n".join(rows)

    @classmethod
    def from_log_ships(cls, ships_dict: Dict[str, Any], size: int) -> "BoardManager":
        """Create a board from log ships data."""
        b = cls(size=size)
        for name, s in ships_dict.items():
            cells = set((int(x), int(y)) for x, y in s.get("cells", []))
            ship = Ship(name=name, length=len(cells) or int(s.get("length", 0)), cells=cells)
            b.ships[name] = ship
            for c in cells:
                b.occupied[c] = name
        return b


def infer_board_size_from_log(log_obj: Dict[str, Any]) -> int:
    """Infer board size from log data."""
    maxx = maxy = -1
    for side_key in ("p1", "p2"):
        side = log_obj.get(side_key, {})
        # ship cells
        for ship in side.get("ships", {}).values():
            for x, y in ship.get("cells", []):
                maxx = max(maxx, int(x))
                maxy = max(maxy, int(y))
        # shots
        for x, y in side.get("shot_history", []):
            maxx = max(maxx, int(x))
            maxy = max(maxy, int(y))
    sz = max(maxx, maxy) + 1
    return sz if sz > 0 else 10


class BattleshipWidget(Vertical):
    """Widget that animates a battleship game replay from JSON log data."""

    BINDINGS = [
        Binding("escape", "close_replay", "Close", show=False),
        Binding("space", "toggle_pause", "Pause/Resume", show=False),
        Binding("right", "step_forward", "Step Forward", show=False),
        Binding("left", "step_back", "Step Back", show=False),
        Binding("up", "speed_up", "Speed Up", show=False),
        Binding("down", "slow_down", "Slow Down", show=False),
    ]

    def __init__(
        self, log_data: Dict[str, Any], delay_seconds: float = 0.5, submitter_player: Optional[int] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.log_data = log_data
        self.delay_seconds = delay_seconds
        self.min_delay = 0.05  # Minimum delay (fastest speed)
        self.max_delay = 2.0  # Maximum delay (slowest speed)
        self.submitter_player = submitter_player  # 1 or 2, None if unknown
        self.board1: Optional[BoardManager] = None
        self.board2: Optional[BoardManager] = None
        self.p1_moves: list[Coord] = []
        self.p2_moves: list[Coord] = []
        self.game_id: str = ""
        self.winner_from_log: Optional[str] = None
        self.header_widget: Optional[Static] = None
        self.board1_widget: Optional[Static] = None
        self.board2_widget: Optional[Static] = None
        self.message_widget: Optional[Static] = None
        self.legend_widget: Optional[Static] = None
        self.animation_timer = None
        self.is_playing = False
        self.current_turn = 0
        self.i1 = 0
        self.i2 = 0
        self.current_player = "Player 1"
        self.last_msg = ""
        self.game_over = False
        # History for step back functionality
        self.history: list[Dict[str, Any]] = []  # Store state snapshots

    def compose(self):
        """Compose the widget with child widgets."""
        self.header_widget = Static("", id="battleship_header")
        self.message_widget = Static("", id="battleship_message")
        self.board1_widget = Static("", id="battleship_board1")
        self.board2_widget = Static("", id="battleship_board2")
        self.legend_widget = Static("", id="battleship_legend")

        yield self.header_widget
        yield self.message_widget
        with Horizontal():
            with Vertical():
                yield self.board1_widget
            with Vertical():
                yield self.board2_widget
        yield self.legend_widget

    def on_mount(self) -> None:
        """Initialize the widget when mounted."""
        self.initialize_game()
        # Make widget focusable to receive key events
        self.can_focus = True
        self.focus()

    def initialize_game(self) -> None:
        """Initialize the game from log data."""
        try:
            # Build boards
            size = int(self.log_data.get("board_size") or infer_board_size_from_log(self.log_data))
            p1_meta = self.log_data.get("p1", {})
            p2_meta = self.log_data.get("p2", {})
            self.board1 = BoardManager.from_log_ships(p1_meta.get("ships", {}), size=size)
            self.board2 = BoardManager.from_log_ships(p2_meta.get("ships", {}), size=size)

            self.p1_moves = [(int(x), int(y)) for x, y in p1_meta.get("shot_history", [])]
            self.p2_moves = [(int(x), int(y)) for x, y in p2_meta.get("shot_history", [])]

            self.game_id = self.log_data.get("game_id", "unknown")
            self.winner_from_log = self.log_data.get("winner")

            # Reset animation state
            self.current_turn = 0
            self.i1 = 0
            self.i2 = 0
            self.current_player = "Player 1"
            self.last_msg = ""
            self.game_over = False
            self.history = []  # Reset history

            # Save initial state to history
            self.save_state_to_history()

            # Show initial state
            self.update_display()
            self.update_legend()

            # Start animation after a short delay to show initial state
            self.set_timer(0.5, self.start_animation)
        except Exception as e:
            if self.message_widget:
                self.message_widget.update(f"[red]Error initializing game: {e}[/red]")

    def start_animation(self) -> None:
        """Start the animation loop."""
        if self.is_playing:
            return
        self.is_playing = True
        self.game_over = False
        self.update_legend()
        # Start the first move immediately
        self.process_next_move()

    def save_state_to_history(self) -> None:
        """Save current game state to history for step back functionality."""
        if not self.board1 or not self.board2:
            return
        # Deep copy board states
        state = {
            "turn": self.current_turn,
            "i1": self.i1,
            "i2": self.i2,
            "current_player": self.current_player,
            "last_msg": self.last_msg,
            "game_over": self.game_over,
            "board1_ships": {
                name: {"hits": set(ship.hits), "cells": ship.cells} for name, ship in self.board1.ships.items()
            },
            "board1_misses": set(self.board1.misses),
            "board2_ships": {
                name: {"hits": set(ship.hits), "cells": ship.cells} for name, ship in self.board2.ships.items()
            },
            "board2_misses": set(self.board2.misses),
        }
        self.history.append(state)

    def restore_state_from_history(self, state: Dict[str, Any]) -> None:
        """Restore game state from history."""
        self.current_turn = state["turn"]
        self.i1 = state["i1"]
        self.i2 = state["i2"]
        self.current_player = state["current_player"]
        self.last_msg = state["last_msg"]
        self.game_over = state["game_over"]

        # Restore board1
        if self.board1:
            for name, ship_data in state["board1_ships"].items():
                if name in self.board1.ships:
                    self.board1.ships[name].hits = set(ship_data["hits"])
            self.board1.misses = set(state["board1_misses"])

        # Restore board2
        if self.board2:
            for name, ship_data in state["board2_ships"].items():
                if name in self.board2.ships:
                    self.board2.ships[name].hits = set(ship_data["hits"])
            self.board2.misses = set(state["board2_misses"])

    def process_next_move(self, manual_step: bool = False) -> None:
        """Process the next move in the game.

        Args:
            manual_step: If True, process move even when not playing (for manual stepping)
        """
        if not manual_step and (not self.is_playing or self.game_over):
            return

        # Save state before making move (for step back)
        self.save_state_to_history()

        # Check if game is over
        if self.current_player == "Player 1":
            if self.i1 >= len(self.p1_moves):
                # Game over
                self.is_playing = False
                self.game_over = True
                self.show_final_state()
                return
            x, y = self.p1_moves[self.i1]
            self.i1 += 1
            if not self.board2:
                return
            hit, sunk = self.board2.receive_shot(x, y)
            self.last_msg = f"Player 1 shoots {x},{y} -> {'HIT' if hit else 'MISS'}" + (
                f", sank {sunk}" if sunk else ""
            )
            if self.board2.all_ships_sunk():
                self.is_playing = False
                self.game_over = True
                self.current_turn += 1
                self.update_display()
                self.show_final_state(f"Winner: Player 1 in {self.current_turn} turns.")
                return
            self.current_player = "Player 2"
        else:
            if self.i2 >= len(self.p2_moves):
                # Game over
                self.is_playing = False
                self.game_over = True
                self.show_final_state()
                return
            x, y = self.p2_moves[self.i2]
            self.i2 += 1
            if not self.board1:
                return
            hit, sunk = self.board1.receive_shot(x, y)
            self.last_msg = f"Player 2 shoots {x},{y} -> {'HIT' if hit else 'MISS'}" + (
                f", sank {sunk}" if sunk else ""
            )
            if self.board1.all_ships_sunk():
                self.is_playing = False
                self.game_over = True
                self.current_turn += 1
                self.update_display()
                self.show_final_state(f"Winner: Player 2 in {self.current_turn} turns.")
                return
            self.current_player = "Player 1"

        self.current_turn += 1
        self.update_display()

        # Schedule next move (only if playing automatically)
        if self.is_playing and not self.game_over and not manual_step:
            self.animation_timer = self.set_timer(self.delay_seconds, self.process_next_move)

    def update_display(self) -> None:
        """Update the display with current game state."""
        if not all([self.header_widget, self.board1_widget, self.board2_widget, self.message_widget]):
            return

        if not self.board1 or not self.board2:
            return

        # Update header
        header_text = f"[bold cyan]Game ID:[/bold cyan] {self.game_id}\n"
        header_text += f"[bold cyan]Turn:[/bold cyan] {self.current_turn}\n"
        header_text += f"[bold cyan]Next:[/bold cyan] {self.current_player}"
        self.header_widget.update(header_text)

        # Update boards with submitter/opponent labels
        if self.submitter_player == 1:
            board1_label = "submitter"
            board2_label = "opponent"
        elif self.submitter_player == 2:
            board1_label = "opponent"
            board2_label = "submitter"
        else:
            # Fallback to "self-view" if submitter_player is unknown
            board1_label = "self-view"
            board2_label = "self-view"

        board1_text = f"[bold]--- Player 1 Board ({board1_label}) ---[/bold]\n"
        board1_text += self.board1.render(reveal=True)
        self.board1_widget.update(board1_text)

        board2_text = f"[bold]--- Player 2 Board ({board2_label}) ---[/bold]\n"
        board2_text += self.board2.render(reveal=True)
        self.board2_widget.update(board2_text)

        # Update message
        if self.last_msg:
            self.message_widget.update(f"[dim]{self.last_msg}[/dim]")
        else:
            self.message_widget.update("")

        # Update legend
        self.update_legend()

    def show_final_state(self, winner_msg: Optional[str] = None) -> None:
        """Show the final game state."""
        self.update_display()
        if winner_msg:
            if self.message_widget:
                self.message_widget.update(f"[bold green]{winner_msg}[/bold green]")
        elif self.winner_from_log:
            if self.message_widget:
                total_turns = self.log_data.get("turns", self.current_turn)
                self.message_widget.update(
                    f"[bold green]Replay complete. Winner (from log): {self.winner_from_log}. Total turns: {total_turns}.[/bold green]"
                )

    def on_unmount(self) -> None:
        """Stop animation when widget is unmounted."""
        self.is_playing = False
        self.game_over = True
        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer = None

    def _notify(self, message: str, severity: str = "information", timeout: float = 1.0) -> None:
        """Safely show a notification."""
        try:
            if hasattr(self, "app") and self.app:
                self.app.notify(message, severity=severity, timeout=timeout)
        except Exception:
            pass  # Silently fail if notification can't be shown

    def update_legend(self) -> None:
        """Update the command legend display."""
        if not self.legend_widget:
            return

        status = "Playing" if self.is_playing else "Paused"
        speed_percent = (
            int((2.0 - self.delay_seconds) / (2.0 - self.min_delay) * 100) if self.delay_seconds > 0 else 100
        )
        speed_percent = max(0, min(100, speed_percent))

        legend_text = f"""[dim]Controls:[/dim]
[bold]ESC[/bold] - Close  |  [bold]Space[/bold] - {status}  |  [bold]←[/bold] - Step Back  |  [bold]→[/bold] - Step Forward  |  [bold]↑[/bold] - Speed Up  |  [bold]↓[/bold] - Slow Down
[dim]Speed: {speed_percent}%[/dim]"""
        self.legend_widget.update(legend_text)

    def action_close_replay(self) -> None:
        """Close the replay widget (ESC key)."""
        # Stop animation
        self.is_playing = False
        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer = None

        # Show notification
        self._notify("Replay closed", severity="information", timeout=2)

        # Post message to parent before removing
        self.post_message(BattleshipWidgetClosed())

        # Remove the widget
        try:
            self.remove()
        except Exception:
            pass

    def action_toggle_pause(self) -> None:
        """Toggle pause/resume (Space key)."""
        if self.game_over:
            return

        if self.is_playing:
            # Pause
            self.is_playing = False
            if self.animation_timer:
                self.animation_timer.stop()
                self.animation_timer = None
            self._notify("Replay paused", severity="information", timeout=2)
        else:
            # Resume
            self.is_playing = True
            self.process_next_move()
            self._notify("Replay resumed", severity="information", timeout=2)

        self.update_legend()

    def action_step_forward(self) -> None:
        """Step forward one move (Right arrow key)."""
        if self.game_over:
            return

        # Pause if playing
        was_playing = self.is_playing
        if self.is_playing:
            self.is_playing = False
            if self.animation_timer:
                self.animation_timer.stop()
                self.animation_timer = None

        # Process one move manually (bypass is_playing check)
        self.process_next_move(manual_step=True)

        # Don't auto-resume if it was paused
        if not was_playing:
            self.is_playing = False

        self.update_legend()

    def action_step_back(self) -> None:
        """Step back one move (Left arrow key)."""
        if self.game_over or len(self.history) < 2:
            return  # Need at least 2 states (initial + one move)

        # Pause if playing
        if self.is_playing:
            self.is_playing = False
            if self.animation_timer:
                self.animation_timer.stop()
                self.animation_timer = None

        # Remove current state and restore previous
        self.history.pop()  # Remove current state
        if self.history:
            previous_state = self.history[-1]
            self.restore_state_from_history(previous_state)
            self.update_display()

        self.update_legend()

    def action_speed_up(self) -> None:
        """Speed up the replay (Up arrow key)."""
        # Decrease delay (faster)
        old_delay = self.delay_seconds
        self.delay_seconds = max(self.min_delay, self.delay_seconds * 0.7)
        # If currently playing, restart timer with new delay
        if self.is_playing and not self.game_over:
            if self.animation_timer:
                self.animation_timer.stop()
            self.animation_timer = self.set_timer(self.delay_seconds, self.process_next_move)

        if self.delay_seconds < old_delay:
            speed_percent = int((2.0 - self.delay_seconds) / (2.0 - self.min_delay) * 100)
            speed_percent = max(0, min(100, speed_percent))

        self.update_legend()

    def action_slow_down(self) -> None:
        """Slow down the replay (Down arrow key)."""
        # Increase delay (slower)
        old_delay = self.delay_seconds
        self.delay_seconds = min(self.max_delay, self.delay_seconds * 1.4)
        # If currently playing, restart timer with new delay
        if self.is_playing and not self.game_over:
            if self.animation_timer:
                self.animation_timer.stop()
            self.animation_timer = self.set_timer(self.delay_seconds, self.process_next_move)

        if self.delay_seconds > old_delay:
            speed_percent = int((2.0 - self.delay_seconds) / (2.0 - self.min_delay) * 100)
            speed_percent = max(0, min(100, speed_percent))

        self.update_legend()


class BattleshipWidgetClosed(Message):
    """Message sent when battleship widget is closed."""

    pass
