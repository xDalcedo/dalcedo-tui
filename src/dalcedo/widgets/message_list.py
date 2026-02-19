"""Message list widget for chat display."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Collapsible, Static


@dataclass
class ChatMessage:
    """A chat message."""

    role: Literal["user", "assistant", "system"]
    content: str
    sql: str | None = None
    query_result: str | None = None
    timing: str | None = None  # Timing summary (e.g., "Total: 1234ms | LLM: 1000ms | ...")
    status: str | None = None  # Loading status (e.g., "Executing query...")
    is_loading: bool = False
    is_error: bool = False


class MessageBox(Vertical):
    """Single message display with optional collapsible SQL."""

    DEFAULT_CSS = """
    MessageBox {
        height: auto;
        padding: 1 2;
        margin: 1 0;
    }

    MessageBox.user-message {
        background: $primary-darken-3;
        border-left: thick $primary;
    }

    MessageBox.assistant-message {
        background: $surface-darken-1;
        border-left: thick $secondary;
    }

    MessageBox.system-message {
        background: $surface;
        border-left: thick $warning;
    }

    MessageBox.loading {
        /* Loading state - content stays readable, status shown below */
    }

    MessageBox.error {
        background: $error 20%;
        border-left: thick $error;
    }

    MessageBox Collapsible {
        margin-top: 1;
    }

    MessageBox Collapsible Static {
        padding: 1 2;
    }
    """

    def __init__(self, message: ChatMessage, **kwargs) -> None:
        super().__init__(**kwargs)
        self.message = message
        self._apply_classes()

    def _apply_classes(self) -> None:
        """Apply role-specific CSS classes."""
        self.remove_class("user-message", "assistant-message", "system-message", "loading", "error")
        msg = self.message
        if msg.role == "user":
            self.add_class("user-message")
        elif msg.role == "system":
            self.add_class("system-message")
        else:
            self.add_class("assistant-message")
            if msg.is_loading:
                self.add_class("loading")
            elif msg.is_error:
                self.add_class("error")

    def compose(self):
        """Compose the message widgets."""
        msg = self.message

        if msg.role == "user":
            yield Static(Text(msg.content))
        elif msg.role == "system":
            yield Static(Text(msg.content, style="dim"))
        else:  # assistant
            if msg.is_error:
                yield Static(Text(msg.content, style="red"))
            elif msg.is_loading:
                # Show content readable, with status indicator below
                if msg.content:
                    yield Static(Markdown(msg.content))
                if msg.status:
                    yield Static(Text(f"\n⏳ {msg.status}", style="dim italic"))
                elif not msg.content:
                    yield Static(Text("Thinking...", style="italic dim"))
            else:
                yield Static(Markdown(msg.content))
                if msg.timing:
                    yield Static(Text(f"\n⏱ {msg.timing}", style="dim italic"))
                if msg.sql:
                    yield Collapsible(
                        Static(Syntax(msg.sql, "sql", theme="monokai", line_numbers=False)),
                        title="Show SQL",
                        collapsed=True,
                    )
                if msg.query_result:
                    yield Collapsible(
                        Static(Text(msg.query_result, style="dim")),
                        title="Show raw results",
                        collapsed=True,
                    )

    def update_message(self, message: ChatMessage) -> None:
        """Update the message by replacing this widget."""
        self.message = message
        self._apply_classes()

        # Remove all children
        for child in list(self.children):
            child.remove()

        # Re-compose
        msg = self.message

        if msg.role == "user":
            self.mount(Static(Text(msg.content)))
        elif msg.role == "system":
            self.mount(Static(Text(msg.content, style="dim")))
        else:  # assistant
            if msg.is_error:
                self.mount(Static(Text(msg.content, style="red")))
            elif msg.is_loading:
                # Show content readable, with status indicator below
                if msg.content:
                    self.mount(Static(Markdown(msg.content)))
                if msg.status:
                    self.mount(Static(Text(f"\n⏳ {msg.status}", style="dim italic")))
                elif not msg.content:
                    self.mount(Static(Text("Thinking...", style="italic dim")))
            else:
                self.mount(Static(Markdown(msg.content)))
                if msg.timing:
                    self.mount(Static(Text(f"\n⏱ {msg.timing}", style="dim italic")))
                if msg.sql:
                    self.mount(
                        Collapsible(
                            Static(Syntax(msg.sql, "sql", theme="monokai", line_numbers=False)),
                            title="Show SQL",
                            collapsed=True,
                        )
                    )
                if msg.query_result:
                    self.mount(
                        Collapsible(
                            Static(Text(msg.query_result, style="dim")),
                            title="Show raw results",
                            collapsed=True,
                        )
                    )


class MessageList(VerticalScroll):
    """Scrollable message container."""

    DEFAULT_CSS = """
    MessageList {
        height: 1fr;
        padding: 1;
    }
    """

    def add_message(self, message: ChatMessage) -> None:
        """Add a new message to the list."""
        box = MessageBox(message)
        self.mount(box)
        self.scroll_end(animate=False)

    def add_user_message(self, content: str) -> None:
        """Convenience method to add a user message."""
        self.add_message(ChatMessage(role="user", content=content))

    def add_assistant_message(
        self, content: str, sql: str | None = None, is_loading: bool = False
    ) -> None:
        """Convenience method to add an assistant message."""
        self.add_message(
            ChatMessage(role="assistant", content=content, sql=sql, is_loading=is_loading)
        )

    def add_system_message(self, content: str) -> None:
        """Convenience method to add a system message."""
        self.add_message(ChatMessage(role="system", content=content))

    def update_last_message(self, message: ChatMessage) -> None:
        """Update the most recent message."""
        children = list(self.query(MessageBox))
        if children:
            children[-1].update_message(message)

    def clear(self) -> None:
        """Remove all messages."""
        for child in self.query(MessageBox):
            child.remove()
