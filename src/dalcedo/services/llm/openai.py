"""OpenAI LLM provider implementation."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from openai import AsyncOpenAI
from openai import (
    APIConnectionError as OpenAIConnectionError,
    APIStatusError as OpenAIStatusError,
    AuthenticationError as OpenAIAuthError,
    RateLimitError as OpenAIRateLimitError,
)

from dalcedo.services.llm.base import (
    BaseLLMService,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMResponse,
    LLMServiceUnavailableError,
    StreamEvent,
    ToolCall,
    ToolResult,
)


class OpenAILLMService(BaseLLMService):
    """OpenAI API implementation."""

    PROVIDER = "OpenAI"

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    def _translate_exception(self, e: Exception) -> Exception:
        """Translate OpenAI-specific exceptions to our custom exceptions."""
        if isinstance(e, OpenAIAuthError):
            return LLMAuthenticationError(self.PROVIDER)
        if isinstance(e, OpenAIRateLimitError):
            # Check for quota exceeded (usually in the error message)
            error_msg = str(e).lower()
            if "quota" in error_msg or "billing" in error_msg or "insufficient" in error_msg:
                return LLMQuotaExceededError(self.PROVIDER)
            # Extract retry-after if available
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after_str = e.response.headers.get("retry-after")
                if retry_after_str:
                    try:
                        retry_after = int(retry_after_str)
                    except ValueError:
                        pass
            return LLMRateLimitError(self.PROVIDER, retry_after)
        if isinstance(e, OpenAIConnectionError):
            return LLMConnectionError(self.PROVIDER)
        if isinstance(e, OpenAIStatusError):
            # Check for service unavailable (503)
            if e.status_code == 503:
                return LLMServiceUnavailableError(self.PROVIDER)
            # Check for billing/quota issues (402, 429 with specific messages)
            if e.status_code == 402:
                return LLMQuotaExceededError(self.PROVIDER)
        return e

    async def test_connection(self) -> bool:
        """Verify API key is valid."""
        try:
            await self.client.chat.completions.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            raise self._translate_exception(e) from e

    async def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat message and get a response."""
        openai_messages = self._format_messages(messages, system)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=openai_messages,
            )
            return self._normalize_response(response)
        except Exception as e:
            raise self._translate_exception(e) from e

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat message with tool definitions."""
        openai_messages = self._format_messages(messages, system)
        openai_tools = self._format_tools(tools)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=openai_messages,
                tools=openai_tools,
            )
            return self._normalize_response(response)
        except Exception as e:
            raise self._translate_exception(e) from e

    async def chat_with_tools_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a chat response with tool definitions."""
        openai_messages = self._format_messages(messages, system)
        openai_tools = self._format_tools(tools)

        # Track state during streaming
        content_text = ""
        tool_calls: list[ToolCall] = []
        tool_call_args: dict[int, str] = {}  # index -> accumulated arguments
        tool_call_info: dict[int, dict] = {}  # index -> {id, name}
        finish_reason = None

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=openai_messages,
                tools=openai_tools,
                stream=True,
            )
        except Exception as e:
            raise self._translate_exception(e) from e

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                # Check for finish reason
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta

                # Handle text content
                if delta and delta.content:
                    content_text += delta.content
                    yield StreamEvent(type="text_delta", text=delta.content)

                # Handle tool calls
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index

                        # Initialize tool call info if this is a new one
                        if idx not in tool_call_info and tc_delta.id:
                            tool_call_info[idx] = {
                                "id": tc_delta.id,
                                "name": tc_delta.function.name if tc_delta.function else "",
                            }
                            tool_call_args[idx] = ""

                        # Accumulate function arguments
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_call_args[idx] = (
                                tool_call_args.get(idx, "") + tc_delta.function.arguments
                            )
        except Exception as e:
            raise self._translate_exception(e) from e

        # Process completed tool calls
        for idx in sorted(tool_call_info.keys()):
            info = tool_call_info[idx]
            args_str = tool_call_args.get(idx, "{}")
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {}

            tool_call = ToolCall(
                id=info["id"],
                name=info["name"],
                arguments=args,
            )
            tool_calls.append(tool_call)
            yield StreamEvent(type="tool_use", tool_call=tool_call)

        # Normalize stop reason
        if finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        # Yield final done event
        final_response = LLMResponse(
            content=content_text if content_text else None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )
        yield StreamEvent(type="done", response=final_response)

    def _format_messages(self, messages: list[dict], system: str | None) -> list[dict]:
        """Format messages for OpenAI API."""
        openai_messages = []

        # Add system message if provided
        if system:
            openai_messages.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg.get("content")

            if role == "user":
                # Handle tool results from previous turn
                if (
                    isinstance(content, list)
                    and content
                    and content[0].get("type") == "tool_result"
                ):
                    for tool_result in content:
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_result["tool_use_id"],
                                "content": tool_result["content"],
                            }
                        )
                else:
                    openai_messages.append({"role": "user", "content": content})

            elif role == "assistant":
                # Handle assistant messages with tool calls
                if isinstance(content, list):
                    text_parts = []
                    tool_calls = []

                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_calls.append(
                                {
                                    "id": block["id"],
                                    "type": "function",
                                    "function": {
                                        "name": block["name"],
                                        "arguments": json.dumps(block["input"]),
                                    },
                                }
                            )

                    assistant_msg: dict[str, Any] = {"role": "assistant"}
                    if text_parts:
                        assistant_msg["content"] = "\n".join(text_parts)
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    openai_messages.append(assistant_msg)
                else:
                    openai_messages.append({"role": "assistant", "content": content})

        return openai_messages

    def _format_tools(self, tools: list[dict]) -> list[dict]:
        """Convert tools to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    def _normalize_response(self, response) -> LLMResponse:
        """Convert OpenAI response to normalized format."""
        choice = response.choices[0]
        message = choice.message

        content = message.content
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        # Map finish reason
        stop_reason = "end_turn"
        if choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif choice.finish_reason == "length":
            stop_reason = "max_tokens"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw_response=response,
        )

    def format_tool_results(self, results: list[ToolResult]) -> list[dict]:
        """Format tool results for OpenAI's expected format.

        Note: For OpenAI, tool results are formatted as separate messages
        in _format_messages, but we return them in Anthropic format here
        since the agent uses a consistent interface.
        """
        return [
            {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
            for result in results
        ]

    def format_assistant_message(self, response: LLMResponse) -> dict:
        """Format assistant message for conversation history.

        Uses a normalized format that _format_messages can convert.
        """
        content = []

        if response.content:
            content.append({"type": "text", "text": response.content})

        for tc in response.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )

        return {"role": "assistant", "content": content}
