"""Structured logging and tracing for agent execution."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from xdg_base_dirs import xdg_cache_home


class Tracer:
    """Structured tracer for agent execution.

    Logs events as JSON lines to a file for debugging and analysis.
    Each user question gets a unique trace_id that flows through all events.
    """

    def __init__(self, trace_file: Path | None = None):
        """Initialize tracer.

        Args:
            trace_file: Path to trace file. Defaults to ~/.cache/dalcedo/traces.jsonl
        """
        if trace_file is None:
            cache_dir = xdg_cache_home() / "dalcedo"
            cache_dir.mkdir(parents=True, exist_ok=True)
            trace_file = cache_dir / "traces.jsonl"

        self.trace_file = trace_file
        self.trace_id: str | None = None
        self.question: str | None = None
        self.start_time: datetime | None = None

    def start_trace(self, question: str) -> str:
        """Start a new trace for a user question.

        Args:
            question: The user's question

        Returns:
            The generated trace_id
        """
        self.trace_id = uuid.uuid4().hex[:12]
        self.question = question
        self.start_time = datetime.now()

        self._log_event(
            event_type="trace_start",
            question=question,
        )

        return self.trace_id

    def log_iteration_start(self, iteration: int, mode: str) -> None:
        """Log the start of an agent iteration."""
        self._log_event(
            event_type="iteration_start",
            iteration=iteration,
            mode=mode,
        )

    def log_llm_call(
        self,
        iteration: int,
        duration_ms: float,
        input_tokens: int,
        output_tokens: int,
        stop_reason: str,
        tool_calls: list[str] | None = None,
        has_content: bool = False,
    ) -> None:
        """Log an LLM API call."""
        self._log_event(
            event_type="llm_call",
            iteration=iteration,
            duration_ms=round(duration_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            has_content=has_content,
        )

    def log_tool_call(
        self,
        iteration: int,
        tool_name: str,
        tool_args: dict[str, Any],
        duration_ms: float,
        is_error: bool = False,
        row_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Log a tool execution."""
        # Sanitize args - truncate long SQL
        sanitized_args = {}
        for key, value in tool_args.items():
            if key == "sql" and isinstance(value, str) and len(value) > 500:
                sanitized_args[key] = value[:500] + "..."
            else:
                sanitized_args[key] = value

        self._log_event(
            event_type="tool_call",
            iteration=iteration,
            tool_name=tool_name,
            tool_args=sanitized_args,
            duration_ms=round(duration_ms, 2),
            is_error=is_error,
            row_count=row_count,
            error_message=error_message,
        )

    def log_trace_end(
        self,
        total_ms: float,
        llm_ms: float,
        tool_ms: float,
        iterations: int,
        input_tokens: int,
        output_tokens: int,
        sql_queries: int,
        error: str | None = None,
        answer_length: int = 0,
    ) -> None:
        """Log the end of a trace."""
        self._log_event(
            event_type="trace_end",
            total_ms=round(total_ms, 2),
            llm_ms=round(llm_ms, 2),
            tool_ms=round(tool_ms, 2),
            iterations=iterations,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            sql_queries=sql_queries,
            error=error,
            answer_length=answer_length,
        )

        # Reset state
        self.trace_id = None
        self.question = None
        self.start_time = None

    def log_error(self, error_type: str, message: str, iteration: int | None = None) -> None:
        """Log an error event."""
        self._log_event(
            event_type="error",
            error_type=error_type,
            message=message[:500] if len(message) > 500 else message,
            iteration=iteration,
        )

    def _log_event(self, event_type: str, **kwargs: Any) -> None:
        """Write an event to the trace file."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "trace_id": self.trace_id,
            "event_type": event_type,
            **{k: v for k, v in kwargs.items() if v is not None},
        }

        try:
            with open(self.trace_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            # Don't let tracing errors break the application
            pass


# Global tracer instance
_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def set_tracer(tracer: Tracer) -> None:
    """Set a custom tracer (useful for testing)."""
    global _tracer
    _tracer = tracer
