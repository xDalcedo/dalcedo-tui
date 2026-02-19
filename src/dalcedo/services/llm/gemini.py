"""Google Gemini LLM provider implementation."""

from __future__ import annotations

from typing import AsyncIterator

from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

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


class GeminiLLMService(BaseLLMService):
    """Google Gemini API implementation."""

    PROVIDER = "Gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _translate_exception(self, e: Exception) -> Exception:
        """Translate Gemini/Google-specific exceptions to our custom exceptions."""
        # Check for Google API core exceptions
        if isinstance(e, google_exceptions.Unauthenticated):
            return LLMAuthenticationError(self.PROVIDER)
        if isinstance(e, google_exceptions.PermissionDenied):
            # Often indicates invalid API key
            return LLMAuthenticationError(self.PROVIDER)
        if isinstance(e, google_exceptions.ResourceExhausted):
            # Could be rate limit or quota
            error_msg = str(e).lower()
            if "quota" in error_msg or "billing" in error_msg:
                return LLMQuotaExceededError(self.PROVIDER)
            return LLMRateLimitError(self.PROVIDER)
        if isinstance(e, google_exceptions.ServiceUnavailable):
            return LLMServiceUnavailableError(self.PROVIDER)
        if isinstance(e, (google_exceptions.GoogleAPIError, ConnectionError, OSError)):
            # Network-related errors
            error_msg = str(e).lower()
            if "connect" in error_msg or "network" in error_msg or "timeout" in error_msg:
                return LLMConnectionError(self.PROVIDER)
        # Check error message for common patterns
        error_msg = str(e).lower()
        if "api key" in error_msg or "invalid" in error_msg and "key" in error_msg:
            return LLMAuthenticationError(self.PROVIDER)
        if "quota" in error_msg or "billing" in error_msg:
            return LLMQuotaExceededError(self.PROVIDER)
        return e

    async def test_connection(self) -> bool:
        """Verify API key is valid."""
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents="Hi",
                config=types.GenerateContentConfig(max_output_tokens=10),
            )
            return response is not None
        except Exception as e:
            raise self._translate_exception(e) from e

    async def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat message and get a response."""
        contents = self._format_messages(messages)
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system,
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
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
        contents = self._format_messages(messages)
        gemini_tools = self._format_tools(tools)
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system,
            tools=gemini_tools,
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
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
        contents = self._format_messages(messages)
        gemini_tools = self._format_tools(tools)
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system,
            tools=gemini_tools,
        )

        # Track state during streaming
        content_text = ""
        tool_calls: list[ToolCall] = []
        finish_reason = None

        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise self._translate_exception(e) from e

        try:
            async for chunk in stream:
                # Process each candidate
                if chunk.candidates:
                    candidate = chunk.candidates[0]

                    if candidate.finish_reason:
                        finish_reason = candidate.finish_reason

                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            # Handle text content
                            if part.text:
                                content_text += part.text
                                yield StreamEvent(type="text_delta", text=part.text)

                            # Handle function calls
                            if part.function_call:
                                fc = part.function_call
                                tool_call = ToolCall(
                                    id=fc.name,  # Gemini doesn't have separate IDs
                                    name=fc.name,
                                    arguments=dict(fc.args) if fc.args else {},
                                )
                                tool_calls.append(tool_call)
                                yield StreamEvent(type="tool_use", tool_call=tool_call)
        except Exception as e:
            raise self._translate_exception(e) from e

        # Normalize stop reason
        if tool_calls:
            stop_reason = "tool_use"
        elif finish_reason and finish_reason.name == "MAX_TOKENS":
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

    def _format_messages(self, messages: list[dict]) -> list[types.Content]:
        """Format messages for Gemini API."""
        contents: list[types.Content] = []

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
                    parts = []
                    for tool_result in content:
                        parts.append(
                            types.Part.from_function_response(
                                name=tool_result.get("tool_use_id", "unknown"),
                                response={"result": tool_result["content"]},
                            )
                        )
                    contents.append(types.Content(role="user", parts=parts))
                else:
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=content if content else "")],
                        )
                    )

            elif role == "assistant":
                # Handle assistant messages with tool calls
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if block.get("type") == "text":
                            parts.append(types.Part.from_text(text=block.get("text", "")))
                        elif block.get("type") == "tool_use":
                            parts.append(
                                types.Part.from_function_call(
                                    name=block["name"],
                                    args=block["input"],
                                )
                            )
                    if parts:
                        contents.append(types.Content(role="model", parts=parts))
                else:
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=content if content else "")],
                        )
                    )

        return contents

    def _format_tools(self, tools: list[dict]) -> list[types.Tool]:
        """Convert tools to Gemini format."""
        function_declarations = []
        for tool in tools:
            # Convert JSON Schema to Gemini schema format
            schema = tool["input_schema"]
            properties = {}
            for prop_name, prop_def in schema.get("properties", {}).items():
                prop_type = prop_def.get("type", "string").upper()
                if prop_type == "INTEGER":
                    prop_type = "NUMBER"
                properties[prop_name] = types.Schema(
                    type=prop_type,
                    description=prop_def.get("description", ""),
                )

            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=properties,
                        required=schema.get("required", []),
                    ),
                )
            )

        return [types.Tool(function_declarations=function_declarations)]

    def _normalize_response(self, response) -> LLMResponse:
        """Convert Gemini response to normalized format."""
        candidate = response.candidates[0]

        content_text = ""
        tool_calls = []

        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.text:
                    content_text += part.text
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=fc.name,
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )

        # Map finish reason
        stop_reason = "end_turn"
        if tool_calls:
            stop_reason = "tool_use"
        elif candidate.finish_reason and candidate.finish_reason.name == "MAX_TOKENS":
            stop_reason = "max_tokens"

        return LLMResponse(
            content=content_text if content_text else None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw_response=response,
        )

    def format_tool_results(self, results: list[ToolResult]) -> list[dict]:
        """Format tool results for the conversation history."""
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
