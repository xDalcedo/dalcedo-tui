"""Anthropic Claude LLM provider implementation."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic
from anthropic import (
    APIConnectionError as AnthropicConnectionError,
    APIStatusError as AnthropicStatusError,
    AuthenticationError as AnthropicAuthError,
    RateLimitError as AnthropicRateLimitError,
)

from dalcedo.services.llm.base import (
    BaseLLMService,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInternalError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMResponse,
    LLMServiceUnavailableError,
    StreamEvent,
    ToolCall,
    ToolResult,
)


class AnthropicLLMService(BaseLLMService):
    """Anthropic Claude API implementation."""

    PROVIDER = "Anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    def _translate_exception(self, e: Exception) -> Exception:
        """Translate Anthropic-specific exceptions to our custom exceptions."""
        if isinstance(e, AnthropicAuthError):
            return LLMAuthenticationError(self.PROVIDER)
        if isinstance(e, AnthropicRateLimitError):
            # Check for quota exceeded (usually in the error message or status)
            error_msg = str(e).lower()
            if "quota" in error_msg or "credit" in error_msg or "billing" in error_msg:
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
        if isinstance(e, AnthropicConnectionError):
            return LLMConnectionError(self.PROVIDER)
        if isinstance(e, AnthropicStatusError):
            # Extract request_id if available
            request_id = None
            if hasattr(e, "response") and e.response:
                request_id = e.response.headers.get("request-id")

            # Check for internal server error (500)
            if e.status_code == 500:
                return LLMInternalError(self.PROVIDER, request_id)
            # Check for overloaded (529) or service unavailable (503)
            if e.status_code in (503, 529):
                return LLMServiceUnavailableError(self.PROVIDER)
            # Check for billing/quota issues (402)
            if e.status_code == 402:
                return LLMQuotaExceededError(self.PROVIDER)
        return e

    async def test_connection(self) -> bool:
        """Verify API key is valid."""
        try:
            await self.client.messages.create(
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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        try:
            response = await self.client.messages.create(**kwargs)
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
        # Convert tools to Anthropic format
        anthropic_tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"],
            }
            for tool in tools
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "tools": anthropic_tools,
        }
        if system:
            kwargs["system"] = system

        try:
            response = await self.client.messages.create(**kwargs)
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
        anthropic_tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"],
            }
            for tool in tools
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "tools": anthropic_tools,
        }
        if system:
            kwargs["system"] = system

        # Track state during streaming
        content_text = ""
        tool_calls: list[ToolCall] = []
        current_tool_id: str | None = None
        current_tool_name: str | None = None
        current_tool_input: str = ""
        stop_reason = "end_turn"
        input_tokens = 0
        output_tokens = 0

        try:
            async with self.client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        # Capture input tokens from the message start
                        if hasattr(event, "message") and hasattr(event.message, "usage"):
                            input_tokens = event.message.usage.input_tokens

                    elif event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool_id = block.id
                            current_tool_name = block.name
                            current_tool_input = ""

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            content_text += delta.text
                            yield StreamEvent(type="text_delta", text=delta.text)
                        elif delta.type == "input_json_delta":
                            current_tool_input += delta.partial_json

                    elif event.type == "content_block_stop":
                        if current_tool_id and current_tool_name:
                            # Parse the accumulated tool input
                            try:
                                tool_args = (
                                    json.loads(current_tool_input) if current_tool_input else {}
                                )
                            except json.JSONDecodeError:
                                tool_args = {}

                            tool_call = ToolCall(
                                id=current_tool_id,
                                name=current_tool_name,
                                arguments=tool_args,
                            )
                            tool_calls.append(tool_call)
                            yield StreamEvent(type="tool_use", tool_call=tool_call)

                            # Reset tool state
                            current_tool_id = None
                            current_tool_name = None
                            current_tool_input = ""

                    elif event.type == "message_stop":
                        pass

                    elif event.type == "message_delta":
                        if hasattr(event.delta, "stop_reason") and event.delta.stop_reason:
                            stop_reason = event.delta.stop_reason
                        # Capture output tokens from message delta
                        if hasattr(event, "usage") and event.usage:
                            output_tokens = event.usage.output_tokens
        except Exception as e:
            raise self._translate_exception(e) from e

        # Normalize stop reason
        if stop_reason == "tool_use":
            normalized_stop = "tool_use"
        elif stop_reason == "max_tokens":
            normalized_stop = "max_tokens"
        else:
            normalized_stop = "end_turn"

        # Yield final done event
        final_response = LLMResponse(
            content=content_text if content_text else None,
            tool_calls=tool_calls,
            stop_reason=normalized_stop,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        yield StreamEvent(type="done", response=final_response)

    def _normalize_response(self, response) -> LLMResponse:
        """Convert Anthropic response to normalized format."""
        content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        # Map stop reason
        stop_reason = "end_turn"
        if response.stop_reason == "tool_use":
            stop_reason = "tool_use"
        elif response.stop_reason == "max_tokens":
            stop_reason = "max_tokens"

        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=response,
        )

    def format_tool_results(self, results: list[ToolResult]) -> list[dict]:
        """Format tool results for Anthropic's expected format."""
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
        """Format assistant message for conversation history."""
        # Reconstruct content blocks from the raw response
        if response.raw_response:
            return {
                "role": "assistant",
                "content": [self._block_to_dict(block) for block in response.raw_response.content],
            }
        # Fallback if no raw response
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

    def _block_to_dict(self, block) -> dict:
        """Convert an Anthropic content block to a dict."""
        if block.type == "text":
            return {"type": "text", "text": block.text}
        elif block.type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
        return {"type": block.type}
