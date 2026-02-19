"""Conversation model for managing chat history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class ConversationTurn:
    """Single turn in the conversation."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sql: str | None = None
    result_summary: str | None = None


class Conversation:
    """Manages conversation history with context limits."""

    def __init__(self, max_turns: int = 20):
        self.turns: list[ConversationTurn] = []
        self.max_turns = max_turns

    def add_user_message(self, content: str) -> None:
        """Add user message."""
        self.turns.append(ConversationTurn(role="user", content=content))
        self._trim_history()

    def add_assistant_message(
        self,
        content: str,
        sql: str | None = None,
        result_summary: str | None = None,
    ) -> None:
        """Add assistant response."""
        self.turns.append(
            ConversationTurn(
                role="assistant",
                content=content,
                sql=sql,
                result_summary=result_summary,
            )
        )
        self._trim_history()

    def update_last_assistant(
        self,
        sql: str | None = None,
        result_summary: str | None = None,
    ) -> None:
        """Update the last assistant turn with SQL and results."""
        for turn in reversed(self.turns):
            if turn.role == "assistant":
                if sql:
                    turn.sql = sql
                if result_summary:
                    turn.result_summary = result_summary
                break

    def clear(self) -> None:
        """Reset conversation history."""
        self.turns.clear()

    def to_messages(self) -> list[dict]:
        """Convert to Anthropic API message format."""
        messages = []
        for turn in self.turns:
            content = turn.content
            # Include SQL and results in context for assistant turns
            if turn.role == "assistant" and turn.sql:
                content = f"{content}\n\nSQL:\n```sql\n{turn.sql}\n```"
                if turn.result_summary:
                    content = f"{content}\n\nResult: {turn.result_summary}"
            messages.append({"role": turn.role, "content": content})
        return messages

    def _trim_history(self) -> None:
        """Keep only recent turns to manage context window."""
        if len(self.turns) > self.max_turns:
            # Keep first turn (establishes context) and most recent turns
            self.turns = self.turns[:2] + self.turns[-(self.max_turns - 2) :]

    def __len__(self) -> int:
        return len(self.turns)
