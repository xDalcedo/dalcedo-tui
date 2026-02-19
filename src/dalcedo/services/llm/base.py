"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


# Custom exceptions for LLM errors
class LLMError(Exception):
    """Base exception for LLM-related errors."""

    def __init__(self, message: str, provider: str = "LLM"):
        self.provider = provider
        super().__init__(message)


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded."""

    def __init__(self, provider: str = "LLM", retry_after: int | None = None):
        self.retry_after = retry_after
        if retry_after:
            message = f"{provider} rate limit exceeded. Try again in {retry_after} seconds."
        else:
            message = f"{provider} rate limit exceeded. Please wait and try again."
        super().__init__(message, provider)


class LLMQuotaExceededError(LLMError):
    """Raised when billing quota/credits are exhausted."""

    def __init__(self, provider: str = "LLM"):
        message = f"{provider} quota exceeded. Please check your billing/credits at the provider's website."
        super().__init__(message, provider)


class LLMAuthenticationError(LLMError):
    """Raised when API key is invalid."""

    def __init__(self, provider: str = "LLM"):
        message = f"{provider} authentication failed. Please check your API key in /login."
        super().__init__(message, provider)


class LLMConnectionError(LLMError):
    """Raised when unable to connect to the API."""

    def __init__(self, provider: str = "LLM"):
        message = f"Unable to connect to {provider}. Please check your internet connection."
        super().__init__(message, provider)


class LLMServiceUnavailableError(LLMError):
    """Raised when the LLM service is down or overloaded."""

    def __init__(self, provider: str = "LLM"):
        message = f"{provider} is currently unavailable. Please try again later."
        super().__init__(message, provider)


class LLMInternalError(LLMError):
    """Raised when the LLM API returns an internal server error."""

    def __init__(self, provider: str = "LLM", request_id: str | None = None):
        self.request_id = request_id
        if request_id:
            message = (
                f"{provider} internal error (request_id: {request_id}). "
                "This is a temporary issue on the provider's side. Please try again."
            )
        else:
            message = (
                f"{provider} internal error. "
                "This is a temporary issue on the provider's side. Please try again."
            )
        super().__init__(message, provider)


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str | None  # Text response (None if only tool calls)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn", "tool_use", "max_tokens"
    input_tokens: int = 0  # Tokens in the request
    output_tokens: int = 0  # Tokens in the response
    raw_response: Any = None  # Provider-specific response for debugging


@dataclass
class StreamEvent:
    """Event from streaming response."""

    type: str  # "text_delta", "tool_use", "done"
    text: str = ""  # For text_delta events
    tool_call: ToolCall | None = None  # For tool_use events
    response: LLMResponse | None = None  # For done events (final response)


# Tool definitions that work across providers (JSON Schema format)
AGENT_TOOLS = [
    {
        "name": "execute_sql",
        "description": "Execute a SQL query against the local DuckDB database and return results. Use this to answer data questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to execute (DuckDB/PostgreSQL syntax)",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this query is needed",
                },
            },
            "required": ["sql", "reason"],
        },
    },
    {
        "name": "get_table_schema",
        "description": "Get detailed schema information for a specific table, including column types and descriptions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table to inspect",
                },
            },
            "required": ["table_name"],
        },
    },
    {
        "name": "get_sample_data",
        "description": "Get sample rows from a table to understand its content and data patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table to sample",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of rows to return (default 5, max 20)",
                    "default": 5,
                },
            },
            "required": ["table_name"],
        },
    },
]


class BaseLLMService(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def test_connection(self) -> bool:
        """Verify API credentials are valid."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat message and get a response (no tools)."""
        pass

    @abstractmethod
    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat message with tool definitions."""
        pass

    @abstractmethod
    async def chat_with_tools_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a chat response with tool definitions.

        Yields StreamEvent objects:
        - type="text_delta": Partial text content (use event.text)
        - type="tool_use": Tool call detected (use event.tool_call)
        - type="done": Stream complete (use event.response for final LLMResponse)
        """
        pass
        # Make this an async generator
        if False:
            yield StreamEvent(type="done")

    @abstractmethod
    def format_tool_results(self, results: list[ToolResult]) -> list[dict]:
        """Format tool results for the next message (provider-specific)."""
        pass

    @abstractmethod
    def format_assistant_message(self, response: LLMResponse) -> dict:
        """Format the assistant's response for conversation history."""
        pass
