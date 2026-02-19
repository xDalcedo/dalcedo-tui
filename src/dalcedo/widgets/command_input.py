"""Command input widget with history support."""

from __future__ import annotations

from textual.binding import Binding
from textual.message import Message
from textual.widgets import Input


class QuerySubmitted(Message):
    """Emitted when user submits a query."""

    def __init__(self, query: str) -> None:
        self.query = query
        super().__init__()


class CommandInput(Input):
    """Input widget with command detection and history."""

    BINDINGS = [
        Binding("up", "history_previous", "Previous", show=False),
        Binding("down", "history_next", "Next", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.history: list[str] = []
        self.history_index: int = -1
        self._temp_input: str = ""

    def action_submit(self) -> None:
        """Handle Enter key."""
        value = self.value.strip()
        if value:
            # Add to history (avoid duplicates)
            if not self.history or self.history[-1] != value:
                self.history.append(value)
            self.history_index = -1
            self._temp_input = ""

            # Post message
            self.post_message(QuerySubmitted(value))

            # Clear input
            self.value = ""

    def action_history_previous(self) -> None:
        """Navigate to previous history entry."""
        if not self.history:
            return

        # Save current input when starting to navigate
        if self.history_index == -1:
            self._temp_input = self.value

        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.value = self.history[-(self.history_index + 1)]
            # Move cursor to end
            self.cursor_position = len(self.value)

    def action_history_next(self) -> None:
        """Navigate to next history entry."""
        if self.history_index > 0:
            self.history_index -= 1
            self.value = self.history[-(self.history_index + 1)]
            self.cursor_position = len(self.value)
        elif self.history_index == 0:
            self.history_index = -1
            self.value = self._temp_input
            self.cursor_position = len(self.value)
