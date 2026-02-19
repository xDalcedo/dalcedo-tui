"""Tests for agent service."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from dalcedo.services.llm import AgentService
from dalcedo.services.llm.base import (
    BaseLLMService,
    LLMResponse,
    StreamEvent,
    ToolCall,
    ToolResult,
)


class MockLLMService(BaseLLMService):
    """Mock LLM service for testing."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses = responses or []
        self.response_index = 0
        self.calls: list[dict] = []

    async def test_connection(self) -> bool:
        return True

    async def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "system": system})
        return self._get_next_response()

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return self._get_next_response()

    async def chat_with_tools_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append({"messages": messages, "tools": tools, "system": system, "stream": True})
        response = self._get_next_response()

        # Simulate streaming
        if response.content:
            for char in response.content:
                yield StreamEvent(type="text_delta", text=char)

        for tool_call in response.tool_calls:
            yield StreamEvent(type="tool_use", tool_call=tool_call)

        yield StreamEvent(type="done", response=response)

    def _get_next_response(self) -> LLMResponse:
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        return LLMResponse(content="Default response", tool_calls=[], stop_reason="end_turn")

    def format_tool_results(self, results: list[ToolResult]) -> list[dict]:
        return [
            {"type": "tool_result", "tool_use_id": r.tool_call_id, "content": r.content}
            for r in results
        ]

    def format_assistant_message(self, response: LLMResponse) -> dict:
        content = []
        if response.content:
            content.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            content.append(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
            )
        return {"role": "assistant", "content": content}


class TestAgentService:
    """Tests for AgentService."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM service."""
        return MockLLMService()

    @pytest.fixture
    def agent_with_tables(self, sample_tables, mock_llm):
        """Create an agent service with sample tables."""
        return AgentService(
            llm=mock_llm,
            db=sample_tables,
            max_iterations=5,
        )

    @pytest.mark.asyncio
    async def test_simple_response(self, agent_with_tables):
        """Test agent returns simple response without tool use."""
        agent_with_tables.llm.responses = [
            LLMResponse(
                content="There are 3 users in the database.",
                tool_calls=[],
                stop_reason="end_turn",
            )
        ]

        result = await agent_with_tables.run("How many users are there?")

        assert "3 users" in result.answer
        assert len(result.sql_queries) == 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_tool_use_execute_sql(self, agent_with_tables):
        """Test agent executes SQL via tool."""
        agent_with_tables.llm.responses = [
            # First response: want to use tool
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="execute_sql",
                        arguments={"sql": "SELECT COUNT(*) as count FROM users"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Second response: final answer
            LLMResponse(
                content="There are 3 users in the database.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        result = await agent_with_tables.run("Count the users")

        assert "3" in result.answer
        assert len(result.sql_queries) == 1
        assert "SELECT COUNT" in result.sql_queries[0]

    @pytest.mark.asyncio
    async def test_tool_use_get_table_schema(self, agent_with_tables):
        """Test agent uses get_table_schema tool."""
        agent_with_tables.llm.responses = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_table_schema",
                        arguments={"table_name": "users"},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="The users table has columns: id, name, email, created_at.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        result = await agent_with_tables.run("What columns does users have?")

        assert "users" in result.answer.lower()
        # Tool was called but no SQL queries
        assert len(result.sql_queries) == 0

    @pytest.mark.asyncio
    async def test_tool_use_get_sample_data(self, agent_with_tables):
        """Test agent uses get_sample_data tool."""
        agent_with_tables.llm.responses = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_sample_data",
                        arguments={"table_name": "users", "limit": 3},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="The users table contains Alice, Bob, and Charlie.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        result = await agent_with_tables.run("Show me sample users")

        assert result.error is None

    @pytest.mark.asyncio
    async def test_sql_error_handling(self, agent_with_tables):
        """Test agent handles SQL errors."""
        agent_with_tables.llm.responses = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="execute_sql",
                        arguments={"sql": "SELECT * FROM nonexistent_table"},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="I encountered an error: the table doesn't exist.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        result = await agent_with_tables.run("Query nonexistent table")

        # Agent should handle the error gracefully
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_sql_retry_limit(self, agent_with_tables):
        """Test SQL retry limit is enforced."""
        # Simulate 3 failed attempts (exceeds MAX_SQL_RETRIES of 2)
        agent_with_tables.llm.responses = [
            # First attempt
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="execute_sql",
                        arguments={"sql": "SELECT * FROM bad_table"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Second attempt (retry 1)
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_2",
                        name="execute_sql",
                        arguments={"sql": "SELECT * FROM bad_table_v2"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Third attempt (retry 2)
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_3",
                        name="execute_sql",
                        arguments={"sql": "SELECT * FROM bad_table_v3"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Final response after retry limit
            LLMResponse(
                content="I've tried multiple times but cannot query this table.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        result = await agent_with_tables.run("Query bad table")

        # Should have recorded the SQL attempts
        assert len(result.sql_queries) >= 2

    @pytest.mark.asyncio
    async def test_max_iterations(self, sample_tables, mock_llm):
        """Test max iterations limit."""
        agent = AgentService(
            llm=mock_llm,
            db=sample_tables,
            max_iterations=2,  # Low limit for testing
        )

        # Agent keeps using tools without finishing
        mock_llm.responses = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1", name="get_table_schema", arguments={"table_name": "users"}
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_2", name="get_table_schema", arguments={"table_name": "orders"}
                    )
                ],
                stop_reason="tool_use",
            ),
            # Would continue but max iterations reached
        ]

        result = await agent.run("Analyze all tables")

        assert result.error == "max_iterations_reached"
        assert "maximum" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, agent_with_tables):
        """Test agent handles multiple tool calls in one response."""
        agent_with_tables.llm.responses = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_table_schema",
                        arguments={"table_name": "users"},
                    ),
                    ToolCall(
                        id="call_2",
                        name="get_table_schema",
                        arguments={"table_name": "orders"},
                    ),
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="Both users and orders tables are available.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        result = await agent_with_tables.run("What tables exist?")

        assert result.error is None
        assert len(result.steps) >= 4  # 2 tool calls + 2 results + final response

    @pytest.mark.asyncio
    async def test_step_callback(self, agent_with_tables):
        """Test step callback is invoked."""
        steps_received = []

        def on_step(step):
            steps_received.append(step)

        agent_with_tables.llm.responses = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="execute_sql",
                        arguments={"sql": "SELECT COUNT(*) FROM users"},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="Done",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        await agent_with_tables.run("Count users", on_step=on_step)

        assert len(steps_received) >= 2
        assert any(s.type == "tool_call" for s in steps_received)
        assert any(s.type == "tool_result" for s in steps_received)

    @pytest.mark.asyncio
    async def test_token_callback(self, agent_with_tables):
        """Test token callback is invoked during streaming."""
        tokens_received = []

        def on_token(token):
            tokens_received.append(token)

        agent_with_tables.llm.responses = [
            LLMResponse(
                content="Hello world",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        await agent_with_tables.run("Say hello", on_token=on_token)

        # MockLLMService streams character by character
        assert len(tokens_received) == len("Hello world")

    @pytest.mark.asyncio
    async def test_conversation_history(self, agent_with_tables):
        """Test conversation history is used."""
        history = [
            {"role": "user", "content": "What tables are there?"},
            {"role": "assistant", "content": "There are users and orders tables."},
        ]

        agent_with_tables.llm.responses = [
            LLMResponse(
                content="The users table has 3 rows.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        await agent_with_tables.run(
            "How many rows in users?",
            conversation_history=history,
        )

        # Check that history was passed to LLM
        call = agent_with_tables.llm.calls[0]
        messages = call["messages"]
        assert len(messages) == 3  # 2 history + 1 new question

    @pytest.mark.asyncio
    async def test_unknown_tool(self, agent_with_tables):
        """Test handling of unknown tool."""
        agent_with_tables.llm.responses = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="unknown_tool",
                        arguments={},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="I encountered an issue.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        result = await agent_with_tables.run("Use unknown tool")

        # Agent should handle gracefully
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_table_not_found_error(self, agent_with_tables):
        """Test get_table_schema with nonexistent table."""
        agent_with_tables.llm.responses = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_table_schema",
                        arguments={"table_name": "nonexistent"},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="The table does not exist.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        result = await agent_with_tables.run("Schema for nonexistent")

        # Tool result should contain available tables
        assert any(
            "not found" in str(s.content).lower() for s in result.steps if s.type == "tool_result"
        )

    @pytest.mark.asyncio
    async def test_last_query_result_tracked(self, agent_with_tables):
        """Test that last_query_result is set after successful SQL execution."""
        agent_with_tables.llm.responses = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="execute_sql",
                        arguments={"sql": "SELECT name FROM users LIMIT 1"},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="The first user is `Alice`.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        result = await agent_with_tables.run("Who is the first user?")

        assert result.last_query_result is not None
        assert "Alice" in result.last_query_result

    @pytest.mark.asyncio
    async def test_last_query_result_none_without_sql(self, agent_with_tables):
        """Test that last_query_result is None when no SQL was executed."""
        agent_with_tables.llm.responses = [
            LLMResponse(
                content="I can help you with data questions.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]

        result = await agent_with_tables.run("Hello")

        assert result.last_query_result is None


class TestValidateAnswer:
    """Tests for the _validate_answer method."""

    @pytest.fixture
    def agent(self, sample_tables):
        """Create an agent for validation testing."""
        llm = MockLLMService()
        return AgentService(llm=llm, db=sample_tables, max_iterations=5)

    def test_no_query_result(self, agent):
        """Returns empty when no query result."""
        assert agent._validate_answer("The answer is `abc123`.", None) == []

    def test_no_backtick_values(self, agent):
        """Returns empty when answer has no backtick values."""
        result_text = "Columns: name\nRows: 1\n\nAlice"
        assert agent._validate_answer("The user is Alice.", result_text) == []

    def test_short_values_ignored(self, agent):
        """Values shorter than 6 chars are not flagged."""
        result_text = "Columns: name\nRows: 1\n\nAlice"
        assert agent._validate_answer("The user is `Alice`.", result_text) == []

    def test_matching_value_not_flagged(self, agent):
        """Values that match the results are not flagged."""
        result_text = "Columns: email\nRows: 1\n\nalice@example.com"
        assert agent._validate_answer("The email is `alice@example.com`.", result_text) == []

    def test_mismatched_value_flagged(self, agent):
        """Values that don't match results are flagged."""
        result_text = "Columns: email\nRows: 1\n\nalice@example.com"
        mismatches = agent._validate_answer("The email is `wrong@example.com`.", result_text)
        assert "wrong@example.com" in mismatches

    def test_substring_match_works(self, agent):
        """Substring matching prevents false positives for partial values."""
        result_text = "Columns: id\nRows: 1\n\nproject-abc-123-def-456"
        # Cited value is a substring of a result cell
        assert agent._validate_answer("The project is `abc-123-def-456`.", result_text) == []

    def test_sql_keywords_ignored(self, agent):
        """SQL keywords in backticks are not flagged."""
        result_text = "Columns: name\nRows: 1\n\nAlice"
        assert agent._validate_answer("Use `SELECT` and `DISTINCT` for queries.", result_text) == []

    def test_empty_results(self, agent):
        """Returns empty for 'no results' query output."""
        assert agent._validate_answer("No data found.", "Query returned no results.") == []

    def test_multiple_mismatches(self, agent):
        """Multiple mismatched values are all reported."""
        result_text = "Columns: name, email\nRows: 1\n\nAlice | alice@example.com"
        answer = "The user `wrong_user_name` has email `wrong@example.com`."
        mismatches = agent._validate_answer(answer, result_text)
        assert "wrong_user_name" in mismatches
        assert "wrong@example.com" in mismatches


class TestToolOutputFormatting:
    """Tests for tool output formatting to prevent hallucinations."""

    @pytest.fixture
    def agent(self, sample_tables):
        """Create an agent for testing tool output."""
        llm = MockLLMService()
        return AgentService(llm=llm, db=sample_tables, max_iterations=5)

    def test_execute_sql_wraps_all_values_in_backticks(self, agent):
        """All values in SQL results should be wrapped in backticks."""
        result = agent._tool_execute_sql("SELECT id, name, email FROM users LIMIT 1")

        assert not result.get("is_error", False)
        content = result["content"]

        # Check that values are wrapped in backticks
        assert "`1`" in content  # id
        assert "`Alice`" in content  # name
        assert "`alice@example.com`" in content  # email

    def test_execute_sql_null_values_not_wrapped(self, agent):
        """NULL values should appear as 'NULL' without backticks."""
        # Insert a row with NULL email
        agent.db.connection.execute(
            "INSERT INTO users VALUES (99, 'NullUser', NULL, NULL)"
        )

        result = agent._tool_execute_sql(
            "SELECT id, name, email FROM users WHERE id = 99"
        )

        assert not result.get("is_error", False)
        content = result["content"]

        # NULL should not be wrapped in backticks
        assert "NULL" in content
        assert "`NULL`" not in content
        # But other values should be wrapped
        assert "`99`" in content
        assert "`NullUser`" in content

    def test_execute_sql_numeric_values_wrapped(self, agent):
        """Numeric values should also be wrapped in backticks."""
        result = agent._tool_execute_sql("SELECT id, amount FROM orders LIMIT 1")

        assert not result.get("is_error", False)
        content = result["content"]

        # Both integer and decimal values should be wrapped
        assert "`1`" in content  # id
        assert "`100" in content  # amount (may have .00 suffix)

    def test_execute_sql_date_values_wrapped(self, agent):
        """Date/timestamp values should be wrapped in backticks."""
        result = agent._tool_execute_sql("SELECT id, created_at FROM users LIMIT 1")

        assert not result.get("is_error", False)
        content = result["content"]

        # Date value should be wrapped
        assert "`2024-01-01" in content

    def test_execute_sql_status_values_wrapped(self, agent):
        """Status/label values should be wrapped in backticks."""
        result = agent._tool_execute_sql("SELECT id, status FROM orders")

        assert not result.get("is_error", False)
        content = result["content"]

        # Status values should be wrapped
        assert "`completed`" in content
        assert "`pending`" in content

    def test_get_sample_data_wraps_all_values(self, agent):
        """All values in sample data should be wrapped in backticks."""
        result = agent._tool_get_sample_data("users", limit=1)

        assert not result.get("is_error", False)
        content = result["content"]

        # Check values are wrapped
        assert "`1`" in content  # id
        assert "`Alice`" in content  # name
        assert "`alice@example.com`" in content  # email

    def test_get_sample_data_null_values_not_wrapped(self, agent):
        """NULL values in sample data should appear as 'NULL' without backticks."""
        agent.db.connection.execute(
            "INSERT INTO users VALUES (98, 'AnotherNull', NULL, NULL)"
        )

        result = agent._tool_get_sample_data("users", limit=10)

        assert not result.get("is_error", False)
        content = result["content"]

        # NULL should not be wrapped
        assert "NULL" in content
        assert "`NULL`" not in content

    def test_execute_sql_backticks_in_every_row(self, agent):
        """Every data row should have all values wrapped in backticks."""
        result = agent._tool_execute_sql("SELECT id, name FROM users")

        assert not result.get("is_error", False)
        content = result["content"]

        # Parse the data rows (skip header lines)
        lines = content.split("\n")
        data_lines = [
            line for line in lines
            if line and not line.startswith("Columns:") and not line.startswith("Rows:")
            and not line.startswith("...")
        ]

        # Each data line should have backticks
        for line in data_lines:
            if line.strip():  # Skip empty lines
                # Each cell should be wrapped in backticks (or be NULL)
                cells = line.split(" | ")
                for cell in cells:
                    cell = cell.strip()
                    if cell and cell != "NULL":
                        assert cell.startswith("`") and cell.endswith("`"), \
                            f"Cell '{cell}' is not wrapped in backticks"

    def test_execute_sql_preserves_value_content(self, agent):
        """Backticks should wrap values without modifying them."""
        result = agent._tool_execute_sql(
            "SELECT email FROM users WHERE name = 'Alice'"
        )

        assert not result.get("is_error", False)
        content = result["content"]

        # The exact value should be preserved inside backticks
        assert "`alice@example.com`" in content

    def test_execute_sql_empty_result(self, agent):
        """Empty results should not cause errors."""
        result = agent._tool_execute_sql(
            "SELECT * FROM users WHERE id = 999"
        )

        assert not result.get("is_error", False)
        assert "no results" in result["content"].lower()

    def test_get_sample_data_empty_table(self, agent):
        """Empty table sample should not cause errors."""
        agent.db.connection.execute("CREATE TABLE empty_table (id INTEGER)")

        result = agent._tool_get_sample_data("empty_table")

        assert not result.get("is_error", False)
        assert "empty" in result["content"].lower()
