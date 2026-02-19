"""Provider-agnostic agent with tool use capabilities."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Literal

from dalcedo.services.llm.base import AGENT_TOOLS, BaseLLMService, ToolResult
from dalcedo.services.tracing import get_tracer

if TYPE_CHECKING:
    from dalcedo.services.duckdb import DuckDBService


# Mode-specific settings
MODE_SETTINGS = {
    "quick": {
        "max_iterations": 10,
        "description": "Fast, concise responses",
    },
    "analytics": {
        "max_iterations": 20,
        "description": "Detailed analysis and reports",
    },
}

AGENT_SYSTEM_PROMPT_QUICK = """You are a data analyst assistant with access to a local DuckDB database.

Your goal is to answer questions QUICKLY and CONCISELY by querying the database.

CURRENT DATE: {current_date} ({current_day})

DATABASE ORGANIZATION:
- Data is organized into schemas (e.g., "analytics", "dbt_tests").
- ALWAYS use fully qualified table names: `schema_name.table_name`

RESPONSE STYLE - BE BRIEF:
- Give SHORT answers (1-2 sentences max for simple questions)
- For numbers, just state the number with minimal context
- For lists, show key items only, no commentary
- NO preambles like "Based on my analysis..." or "Let me explain..."
- Use markdown tables for tabular results
- Skip explanations unless explicitly asked

QUERY GUIDELINES:
- Use DuckDB SQL syntax
- Use LIMIT to keep results small
- Always use schema-qualified table names

DATA ACCURACY - CRITICAL:
- ALL values in query results are wrapped in backticks (`). Copy them EXACTLY as shown.
- NEVER invent, guess, or modify values. This includes IDs, names, statuses, dates, labels.
- If you need a value you don't have, run a query to get it first.
- If asked about something not in query results, say "I don't have that information."

ERROR HANDLING:
- Max 2 retry attempts per query
- If "RETRY LIMIT REACHED", explain briefly and stop

Available schema:
{schema}"""

AGENT_SYSTEM_PROMPT_ANALYTICS = """You are a senior data analyst assistant with access to a local DuckDB database containing data from multiple sources.

Your goal is to provide THOROUGH, INSIGHTFUL analysis by querying the database. Take your time to explore the data comprehensively.

CURRENT DATE: {current_date} ({current_day})

DATABASE ORGANIZATION:
- The database contains data from multiple sources, organized into schemas (namespaces).
- Each data source plugin has its own schema (e.g., "analytics", "dbt_tests").
- ALWAYS use fully qualified table names: `schema_name.table_name`
- Example: SELECT * FROM analytics.users JOIN dbt_tests.results ON ...
- You can query across schemas using JOINs with fully qualified names.

RESPONSE STYLE - BE THOROUGH:
- Provide detailed, comprehensive analysis
- Include context, trends, and insights
- Explain what the data means and why it matters
- Highlight anomalies, patterns, or notable findings
- Make recommendations when appropriate
- Use markdown formatting: headers, bullet points, tables
- Structure long responses with clear sections

ANALYSIS APPROACH:
- Start with exploratory queries to understand the data
- Run multiple queries to build a complete picture
- Cross-reference data from different tables/schemas
- Calculate relevant metrics and comparisons
- Look for trends over time if date fields exist
- Identify outliers and explain their significance

QUERY GUIDELINES:
- Think step by step about what data you need
- Use multiple queries to fully answer complex questions
- Always verify your assumptions about the data
- Use DuckDB SQL syntax (PostgreSQL-compatible)
- For aggregations, always include GROUP BY clauses
- Use window functions for advanced analysis
- Always use schema-qualified table names (schema.table)

DATA ACCURACY - CRITICAL:
- ALL values in query results are wrapped in backticks (`). Copy them EXACTLY as shown.
- NEVER invent, guess, or modify values. This includes IDs, names, statuses, dates, labels.
- If you need a value you don't have, run a query to get it first.
- If asked about something not in query results, say "I don't have that information."

ERROR HANDLING:
- If a SQL query fails, analyze the error and try to fix it
- You have a maximum of 2 retry attempts per query
- If you see "RETRY LIMIT REACHED", stop retrying and explain the issue to the user

Available database schema:
{schema}"""


@dataclass
class AgentStep:
    """A single step in the agent's execution."""

    type: str  # "thinking", "tool_call", "tool_result", "response"
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    sql: str | None = None  # Extracted SQL for display
    duration_ms: float = 0.0  # Time taken for this step


@dataclass
class AgentTiming:
    """Timing breakdown for agent execution."""

    total_ms: float = 0.0
    llm_ms: float = 0.0  # Time spent waiting for LLM API
    tool_ms: float = 0.0  # Time spent executing tools (SQL, etc.)
    iterations: int = 0
    input_tokens: int = 0  # Total input tokens across all LLM calls
    output_tokens: int = 0  # Total output tokens across all LLM calls

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def summary(self) -> str:
        """Get a human-readable summary of timing."""
        parts = []
        parts.append(f"Total: {self._format_time(self.total_ms)}")
        parts.append(f"LLM: {self._format_time(self.llm_ms)} ({self._pct(self.llm_ms)}%)")
        parts.append(f"Tools: {self._format_time(self.tool_ms)} ({self._pct(self.tool_ms)}%)")
        parts.append(f"Tokens: {self.input_tokens:,}→{self.output_tokens:,}")
        parts.append(f"Iterations: {self.iterations}")
        return " | ".join(parts)

    def _format_time(self, ms: float) -> str:
        """Format time in ms or seconds."""
        if ms >= 1000:
            return f"{ms/1000:.1f}s"
        return f"{ms:.0f}ms"

    def _pct(self, value: float) -> int:
        """Calculate percentage of total."""
        if self.total_ms == 0:
            return 0
        return int((value / self.total_ms) * 100)


@dataclass
class AgentResult:
    """Final result from agent execution."""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    sql_queries: list[str] = field(default_factory=list)  # All SQL executed
    last_query_result: str | None = None  # Raw text of last successful query result
    timing: AgentTiming = field(default_factory=AgentTiming)
    error: str | None = None


class AgentService:
    """Manages agentic query execution with tools."""

    MAX_SQL_RETRIES = 2  # Maximum retry attempts for failed SQL queries

    def __init__(
        self,
        llm: BaseLLMService,
        db: "DuckDBService",
        max_iterations: int = 10,
        custom_context: str | None = None,
    ):
        self.llm = llm
        self.db = db
        self.max_iterations = max_iterations
        self.mode: Literal["quick", "analytics"] = "quick"
        self.custom_context = custom_context

    def set_custom_context(self, context: str | None) -> None:
        """Set custom context for the agent."""
        self.custom_context = context

    def get_custom_context(self) -> str | None:
        """Get the current custom context."""
        return self.custom_context

    def set_mode(self, mode: Literal["quick", "analytics"]) -> None:
        """Set the response mode."""
        self.mode = mode

    def get_mode(self) -> Literal["quick", "analytics"]:
        """Get the current response mode."""
        return self.mode

    def get_mode_description(self) -> str:
        """Get a description of the current mode."""
        return MODE_SETTINGS[self.mode]["description"]

    async def run(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        on_step: Callable[[AgentStep], None] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> AgentResult:
        """
        Run the agent to answer a question.

        Args:
            question: The user's question
            conversation_history: Previous conversation messages
            on_step: Optional callback for each step (for streaming UI updates)
            on_token: Optional callback for each token during final response streaming
        """
        steps: list[AgentStep] = []
        sql_queries: list[str] = []

        # Timing tracking
        timing = AgentTiming()
        total_start = time.perf_counter()

        # Start tracing
        tracer = get_tracer()
        tracer.start_trace(question)

        # Track last successful query result for validation
        last_query_result_text: str | None = None

        # Track SQL failures for retry limiting
        sql_failure_count = 0
        last_failed_sql: str | None = None

        # Build system prompt with schema based on mode
        schema = self.db.get_schema_description()
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_day = now.strftime("%A")  # Day name like "Monday"

        if self.mode == "analytics":
            system = AGENT_SYSTEM_PROMPT_ANALYTICS.format(
                schema=schema, current_date=current_date, current_day=current_day
            )
        else:
            system = AGENT_SYSTEM_PROMPT_QUICK.format(
                schema=schema, current_date=current_date, current_day=current_day
            )

        # Append custom context if provided
        if self.custom_context:
            system += f"\n\nADDITIONAL CONTEXT (provided by user):\n{self.custom_context}"

        # Get effective max iterations (respect constructor limit as ceiling)
        mode_max_iterations = MODE_SETTINGS[self.mode]["max_iterations"]
        effective_max_iterations = min(self.max_iterations, mode_max_iterations)

        # Initialize messages
        messages = list(conversation_history) if conversation_history else []
        messages.append({"role": "user", "content": question})

        # Accumulate all streamed text across iterations for the final answer
        accumulated_answer = ""

        for iteration in range(effective_max_iterations):
            timing.iterations = iteration + 1
            tracer.log_iteration_start(iteration + 1, self.mode)

            # Use streaming to get the response
            response = None
            streamed_text = ""  # Text from this iteration only

            llm_start = time.perf_counter()
            async for event in self.llm.chat_with_tools_stream(
                messages=messages,
                tools=AGENT_TOOLS,
                system=system,
            ):
                if event.type == "text_delta":
                    streamed_text += event.text
                    if on_token:
                        on_token(event.text)
                elif event.type == "done":
                    response = event.response
                    # Accumulate token usage
                    if response:
                        timing.input_tokens += response.input_tokens
                        timing.output_tokens += response.output_tokens
            llm_duration = (time.perf_counter() - llm_start) * 1000
            timing.llm_ms += llm_duration

            # Log LLM call
            if response:
                tracer.log_llm_call(
                    iteration=iteration + 1,
                    duration_ms=llm_duration,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    stop_reason=response.stop_reason,
                    tool_calls=[tc.name for tc in response.tool_calls] if response.tool_calls else None,
                    has_content=bool(streamed_text),
                )

            # Accumulate text from this iteration
            if streamed_text:
                accumulated_answer += streamed_text

            if response is None:
                # Should not happen, but handle gracefully
                timing.total_ms = (time.perf_counter() - total_start) * 1000
                tracer.log_error("stream_error", "Stream ended unexpectedly", iteration + 1)
                tracer.log_trace_end(
                    total_ms=timing.total_ms,
                    llm_ms=timing.llm_ms,
                    tool_ms=timing.tool_ms,
                    iterations=timing.iterations,
                    input_tokens=timing.input_tokens,
                    output_tokens=timing.output_tokens,
                    sql_queries=len(sql_queries),
                    error="stream_error",
                )
                return AgentResult(
                    answer="Stream ended unexpectedly.",
                    steps=steps,
                    sql_queries=sql_queries,
                    timing=timing,
                    error="stream_error",
                )

            # Check if the agent wants to use tools
            if response.stop_reason == "tool_use" and response.tool_calls:
                # Process each tool call
                tool_results: list[ToolResult] = []

                for tool_call in response.tool_calls:
                    # Record the step
                    step = AgentStep(
                        type="tool_call",
                        content=f"Using {tool_call.name}",
                        tool_name=tool_call.name,
                        tool_args=tool_call.arguments,
                        sql=tool_call.arguments.get("sql")
                        if tool_call.name == "execute_sql"
                        else None,
                    )
                    steps.append(step)
                    if on_step:
                        on_step(step)

                    # Execute the tool with retry tracking for SQL
                    tool_start = time.perf_counter()
                    if tool_call.name == "execute_sql":
                        current_sql = tool_call.arguments["sql"]
                        sql_queries.append(current_sql)

                        # Check if this is a retry of a similar failed query
                        if last_failed_sql is not None:
                            sql_failure_count += 1

                        result = self._execute_tool(tool_call.name, tool_call.arguments)

                        if result.get("is_error", False):
                            last_failed_sql = current_sql

                            # Check retry limit
                            if sql_failure_count >= self.MAX_SQL_RETRIES:
                                result["content"] = (
                                    f"RETRY LIMIT REACHED: The query has failed {sql_failure_count + 1} times. "
                                    f"Please stop retrying and explain to the user what went wrong.\n\n"
                                    f"Last error: {result['content']}"
                                )
                        else:
                            # Reset failure tracking on success
                            sql_failure_count = 0
                            last_failed_sql = None
                            last_query_result_text = result["content"]
                    else:
                        result = self._execute_tool(tool_call.name, tool_call.arguments)
                    tool_duration_ms = (time.perf_counter() - tool_start) * 1000
                    timing.tool_ms += tool_duration_ms
                    step.duration_ms = tool_duration_ms

                    # Extract row count from result if it's a successful SQL query
                    row_count = None
                    if tool_call.name == "execute_sql" and not result.get("is_error", False):
                        # Parse "Rows: N" from result content
                        import re as re_module
                        match = re_module.search(r"Rows: (\d+)", result["content"])
                        if match:
                            row_count = int(match.group(1))

                    # Log tool call
                    tracer.log_tool_call(
                        iteration=iteration + 1,
                        tool_name=tool_call.name,
                        tool_args=tool_call.arguments,
                        duration_ms=tool_duration_ms,
                        is_error=result.get("is_error", False),
                        row_count=row_count,
                        error_message=result["content"][:200] if result.get("is_error", False) else None,
                    )

                    tool_results.append(
                        ToolResult(
                            tool_call_id=tool_call.id,
                            content=result["content"],
                            is_error=result.get("is_error", False),
                        )
                    )

                    # Record result step
                    result_step = AgentStep(
                        type="tool_result",
                        content=result["content"][:500] + "..."
                        if len(result["content"]) > 500
                        else result["content"],
                        tool_name=tool_call.name,
                    )
                    steps.append(result_step)
                    if on_step:
                        on_step(result_step)

                # Add assistant message and tool results to conversation
                messages.append(self.llm.format_assistant_message(response))
                messages.append(
                    {"role": "user", "content": self.llm.format_tool_results(tool_results)}
                )

            else:
                # Agent is done - return the final answer
                # Use accumulated_answer which contains all streamed text across iterations
                answer = accumulated_answer or "I was unable to generate a response."

                # Note: Validation/correction disabled - was causing false positives with
                # markdown formatting (backticks) and regenerating reports unnecessarily.
                # TODO: Re-enable with smarter validation that doesn't trigger on formatting.

                step = AgentStep(type="response", content=answer)
                steps.append(step)
                if on_step:
                    on_step(step)

                timing.total_ms = (time.perf_counter() - total_start) * 1000
                tracer.log_trace_end(
                    total_ms=timing.total_ms,
                    llm_ms=timing.llm_ms,
                    tool_ms=timing.tool_ms,
                    iterations=timing.iterations,
                    input_tokens=timing.input_tokens,
                    output_tokens=timing.output_tokens,
                    sql_queries=len(sql_queries),
                    answer_length=len(answer),
                )
                return AgentResult(
                    answer=answer,
                    steps=steps,
                    sql_queries=sql_queries,
                    last_query_result=last_query_result_text,
                    timing=timing,
                )

        # Max iterations reached - return what we have so far
        timing.total_ms = (time.perf_counter() - total_start) * 1000

        if not accumulated_answer:
            if self.mode == "quick":
                fallback = (
                    "I reached the maximum number of steps without a complete answer. "
                    "Try `/mode analytics` for complex questions that require more exploration."
                )
            else:
                fallback = (
                    "I reached the maximum number of steps without a complete answer. "
                    "Try breaking your question into smaller, more specific queries."
                )
            final_answer = fallback
        else:
            final_answer = accumulated_answer

        tracer.log_trace_end(
            total_ms=timing.total_ms,
            llm_ms=timing.llm_ms,
            tool_ms=timing.tool_ms,
            iterations=timing.iterations,
            input_tokens=timing.input_tokens,
            output_tokens=timing.output_tokens,
            sql_queries=len(sql_queries),
            error="max_iterations_reached",
            answer_length=len(final_answer),
        )
        return AgentResult(
            answer=final_answer,
            steps=steps,
            sql_queries=sql_queries,
            last_query_result=last_query_result_text,
            timing=timing,
            error="max_iterations_reached",
        )

    def _validate_answer(self, answer: str, query_result_text: str | None) -> list[str]:
        """Extract cited values from answer and check against query results.

        Returns list of values that appear in the answer but not in results.
        Only flags values >= 6 chars to avoid false positives on common words.
        """
        if not query_result_text or query_result_text == "Query returned no results.":
            return []

        # Extract backtick-quoted values from the answer
        cited_values = re.findall(r"`([^`]+)`", answer)
        if not cited_values:
            return []

        # Filter to values that look like data (>= 6 chars, not SQL keywords or markdown)
        sql_keywords = {
            "select", "from", "where", "join", "group", "order", "limit",
            "insert", "update", "delete", "create", "table", "having",
            "count", "null", "true", "false", "between", "distinct",
        }
        filtered_values = [
            v for v in cited_values
            if len(v) >= 6 and v.lower() not in sql_keywords
        ]
        if not filtered_values:
            return []

        # Build set of all cell values from the query result text
        # Result format: "Columns: ...\nRows: N\n\nval1 | val2 | ...\n..."
        result_values: set[str] = set()
        for line in query_result_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("Columns:") or line.startswith("Rows:") or line.startswith("..."):
                continue
            for cell in line.split(" | "):
                cell = cell.strip()
                if cell:
                    result_values.add(cell)

        # Check each cited value against result values (substring match)
        mismatches = []
        for value in filtered_values:
            # Check if the cited value appears as a substring in any result cell
            found = any(value in cell or cell in value for cell in result_values)
            if not found:
                mismatches.append(value)

        return mismatches

    def _execute_tool(self, name: str, args: dict) -> dict:
        """Execute a tool and return the result."""
        try:
            if name == "execute_sql":
                return self._tool_execute_sql(args["sql"])
            elif name == "get_table_schema":
                return self._tool_get_table_schema(args["table_name"])
            elif name == "get_sample_data":
                limit = min(args.get("limit", 5), 20)
                return self._tool_get_sample_data(args["table_name"], limit)
            else:
                return {"content": f"Unknown tool: {name}", "is_error": True}
        except Exception as e:
            return {"content": f"Error: {e}", "is_error": True}

    def _tool_execute_sql(self, sql: str) -> dict:
        """Execute SQL and format results."""
        try:
            result = self.db.execute(sql)

            if result.row_count == 0:
                return {"content": "Query returned no results."}

            # Format as text table for the LLM
            lines = [f"Columns: {', '.join(result.columns)}"]
            lines.append(f"Rows: {result.row_count}")
            lines.append("")

            # Show up to 50 rows, wrapping ALL values in backticks to prevent hallucination
            max_rows = 50
            for row in result.rows[:max_rows]:
                formatted_cells = []
                for v in row:
                    if v is None:
                        formatted_cells.append("NULL")
                    else:
                        # Wrap all values in backticks to signal "copy exactly"
                        formatted_cells.append(f"`{v}`")
                lines.append(" | ".join(formatted_cells))

            if result.row_count > max_rows:
                lines.append(f"... and {result.row_count - max_rows} more rows")

            return {"content": "\n".join(lines)}

        except Exception as e:
            return {"content": f"SQL Error: {e}", "is_error": True}

    def _tool_get_table_schema(self, table_name: str) -> dict:
        """Get schema for a specific table. Supports schema.table format."""
        try:
            # Parse schema.table format
            if "." in table_name:
                schema_name, bare_table = table_name.split(".", 1)
            else:
                schema_name = None
                bare_table = table_name

            # Build list of available tables for error messages
            tables_by_schema = self.db.get_tables_by_schema()
            all_qualified = [f"{s}.{t}" for s, tables in tables_by_schema.items() for t in tables]

            # Check if table exists
            found_schema = None
            found_table = None
            if schema_name:
                # Explicit schema provided
                if schema_name in tables_by_schema and bare_table in tables_by_schema[schema_name]:
                    found_schema = schema_name
                    found_table = bare_table
            else:
                # Search all schemas for the table
                for s, tables in tables_by_schema.items():
                    if bare_table in tables:
                        found_schema = s
                        found_table = bare_table
                        break

            if not found_schema or not found_table:
                return {
                    "content": f"Table '{table_name}' not found. Available tables: {', '.join(all_qualified)}",
                    "is_error": True,
                }

            # Get column info
            result = self.db.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = '{found_schema}' AND table_name = '{found_table}'
                ORDER BY ordinal_position
            """)

            qualified_name = f"{found_schema}.{found_table}"
            lines = [f"Schema for table '{qualified_name}':", ""]
            for col_name, data_type, nullable in result.rows:
                null_str = " (nullable)" if nullable == "YES" else " (not null)"
                lines.append(f"  {col_name}: {data_type}{null_str}")

            # Get row count
            count_result = self.db.execute(f'SELECT COUNT(*) FROM "{found_schema}"."{found_table}"')
            lines.append(f"\nTotal rows: {count_result.rows[0][0]}")

            return {"content": "\n".join(lines)}

        except Exception as e:
            return {"content": f"Error: {e}", "is_error": True}

    def _tool_get_sample_data(self, table_name: str, limit: int = 5) -> dict:
        """Get sample rows from a table. Supports schema.table format."""
        try:
            # Parse schema.table format
            if "." in table_name:
                schema_name, bare_table = table_name.split(".", 1)
            else:
                schema_name = None
                bare_table = table_name

            # Build list of available tables for error messages
            tables_by_schema = self.db.get_tables_by_schema()
            all_qualified = [f"{s}.{t}" for s, tables in tables_by_schema.items() for t in tables]

            # Check if table exists
            found_schema = None
            found_table = None
            if schema_name:
                # Explicit schema provided
                if schema_name in tables_by_schema and bare_table in tables_by_schema[schema_name]:
                    found_schema = schema_name
                    found_table = bare_table
            else:
                # Search all schemas for the table
                for s, tables in tables_by_schema.items():
                    if bare_table in tables:
                        found_schema = s
                        found_table = bare_table
                        break

            if not found_schema or not found_table:
                return {
                    "content": f"Table '{table_name}' not found. Available tables: {', '.join(all_qualified)}",
                    "is_error": True,
                }

            qualified_name = f"{found_schema}.{found_table}"
            result = self.db.execute(
                f'SELECT * FROM "{found_schema}"."{found_table}" LIMIT {limit}'
            )

            if result.row_count == 0:
                return {"content": f"Table '{qualified_name}' is empty."}

            lines = [f"Sample data from '{qualified_name}' ({result.row_count} rows):", ""]
            lines.append("Columns: " + " | ".join(result.columns))
            lines.append("-" * 40)

            # Wrap ALL values in backticks to prevent hallucination
            for row in result.rows:
                formatted_cells = []
                for v in row:
                    if v is None:
                        formatted_cells.append("NULL")
                    else:
                        # Wrap all values in backticks to signal "copy exactly"
                        cell_str = str(v)[:50]
                        formatted_cells.append(f"`{cell_str}`")
                lines.append(" | ".join(formatted_cells))

            return {"content": "\n".join(lines)}

        except Exception as e:
            return {"content": f"Error: {e}", "is_error": True}
