"""Main chat screen."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header
from textual.worker import Worker, get_current_worker

from dalcedo.services.llm import AgentStep
from dalcedo.services.llm.base import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMInternalError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
)
from dalcedo.widgets.command_input import CommandInput, QuerySubmitted
from dalcedo.widgets.message_list import ChatMessage, MessageList

if TYPE_CHECKING:
    from dalcedo.app import DalcedoApp


class ChatScreen(Screen):
    """Main chat interface screen."""

    BINDINGS = [
        ("ctrl+l", "login", "Login"),
        ("ctrl+s", "sync", "Sync"),
        ("ctrl+r", "reset", "Reset"),
        ("ctrl+m", "toggle_mode", "Mode"),
        ("ctrl+u", "usage", "Usage"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield MessageList(id="messages")
        yield CommandInput(
            placeholder="Ask a question or use /login, /sync, /reset, /help...",
            id="input",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Show welcome message."""
        app: DalcedoApp = self.app  # type: ignore
        messages = self.query_one("#messages", MessageList)

        if not app.config_storage.is_configured():
            messages.add_system_message(
                "Welcome to Dalcedo! Please run /login to configure your data source."
            )
        elif not app.db_service or not app.db_service.get_tables():
            plugin_count = len(app.connectors) if app.connectors else 0
            if plugin_count == 0:
                messages.add_system_message(
                    "Welcome back! Run /sync to sync your data sources, then ask questions."
                )
            else:
                plugin_names = ", ".join(app.connectors.keys())
                messages.add_system_message(
                    f"Welcome back! Run /sync to sync your data sources ({plugin_names}), then ask questions."
                )
        else:
            tables_by_schema = app.db_service.get_tables_by_schema()
            total_tables = sum(len(t) for t in tables_by_schema.values())
            schema_count = len(tables_by_schema)
            messages.add_system_message(
                f"Ready! {total_tables} tables in {schema_count} schema(s). Ask me anything about your data."
            )

            # Check if we should suggest a sync (data might be stale)
            if app.config_storage.should_suggest_sync():
                sync_age = app.config_storage.get_sync_age_description()
                if sync_age:
                    messages.add_system_message(
                        f"Note: Your data was last synced {sync_age}. "
                        "Run /sync to get the latest data."
                    )
                app.config_storage.record_sync_suggestion_time()

        # Focus the input
        self.query_one("#input", CommandInput).focus()

    def on_query_submitted(self, event: QuerySubmitted) -> None:
        """Handle query submission."""
        query_text = event.query

        # Check for commands
        if query_text.startswith("/"):
            self._handle_command(query_text)
            return

        # Process natural language query
        self._start_query(query_text)

    def _handle_command(self, command: str) -> None:
        """Route slash commands."""
        cmd = command.lower().strip()
        messages = self.query_one("#messages", MessageList)

        if cmd == "/login":
            self.app.push_screen("login")
        elif cmd == "/sync":
            app: DalcedoApp = self.app  # type: ignore
            if not app.config:
                self.notify("Please run /login first", severity="warning")
                messages.add_system_message("Please configure your connection with /login first.")
            else:
                self.app.push_screen("sync")
        elif cmd == "/reset":
            self._reset_conversation()
        elif cmd == "/help":
            self._show_help()
        elif cmd == "/tables":
            self._show_tables()
        elif cmd == "/schema":
            self._show_schema()
        elif cmd == "/mode":
            self._show_mode()
        elif cmd == "/mode quick":
            self._set_mode("quick")
        elif cmd == "/mode analytics":
            self._set_mode("analytics")
        elif cmd == "/context":
            self._show_context()
        elif cmd == "/context clear":
            self._clear_context()
        elif cmd.startswith("/context "):
            # Extract the context text (preserve original case)
            context_text = command.strip()[9:]  # len("/context ") = 9
            self._set_context(context_text)
        elif cmd == "/traces":
            self._show_traces()
        elif cmd.startswith("/traces "):
            # /traces N - show last N traces
            try:
                count = int(cmd.split()[1])
                self._show_traces(count)
            except (ValueError, IndexError):
                messages.add_system_message("Usage: /traces [N] - Show last N traces (default 5)")
        elif cmd == "/usage":
            self._show_usage()
        else:
            self.notify(f"Unknown command: {cmd}", severity="warning")
            messages.add_system_message(
                f"Unknown command: {cmd}. Use /help to see available commands."
            )

    def _start_query(self, query: str) -> None:
        """Start processing a natural language query."""
        app: DalcedoApp = self.app  # type: ignore
        messages = self.query_one("#messages", MessageList)

        # Check if agent service is available
        if not app.agent_service:
            messages.add_system_message("Please run /login to configure your connection first.")
            return

        # Check if schema is synced
        tables = app.db_service.get_tables()
        if not tables:
            messages.add_system_message("No tables synced. Please run /sync first.")
            return

        # Check token limits
        limit_exceeded = self._check_token_limits()
        if limit_exceeded:
            messages.add_system_message(limit_exceeded)
            return

        # Show user message immediately
        messages.add_user_message(query)
        app.conversation.add_user_message(query)

        # Show thinking indicator immediately
        messages.add_assistant_message("Thinking...", is_loading=True)

        # Disable input while processing
        self.query_one("#input", CommandInput).disabled = True

        # Run agent in background thread
        self.run_worker(
            lambda: self._process_query_worker(query),
            thread=True,
            exclusive=True,
        )

    def _process_query_worker(self, query: str) -> None:
        """Process query using the agent in background thread."""
        import asyncio

        app: DalcedoApp = self.app  # type: ignore
        worker = get_current_worker()

        # Track streaming state
        streaming_text = ""

        def on_step(step: AgentStep) -> None:
            """Handle agent step updates."""
            nonlocal streaming_text

            if worker.is_cancelled:
                return

            if step.type == "tool_call":
                # Show tool execution status below any streamed text
                if step.sql:
                    status = "Executing query..."
                elif step.tool_name == "get_table_schema":
                    status = "Inspecting table schema..."
                elif step.tool_name == "get_sample_data":
                    status = "Fetching sample data..."
                else:
                    status = f"Running {step.tool_name}..."
                self.app.call_from_thread(self._update_with_status, streaming_text, status)

        def on_token(token: str) -> None:
            """Handle streaming tokens."""
            nonlocal streaming_text

            if worker.is_cancelled:
                return

            streaming_text += token

            # Update UI with current streamed text (removes any status line)
            self.app.call_from_thread(self._update_streaming_response, streaming_text)

        try:
            # Run agent (async in thread)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    app.agent_service.run(
                        question=query,
                        conversation_history=app.conversation.to_messages()[:-1],
                        on_step=on_step,
                        on_token=on_token,
                    )
                )
            finally:
                loop.close()

            if worker.is_cancelled:
                return

            # Get the last SQL query executed (if any) for display
            last_sql = result.sql_queries[-1] if result.sql_queries else None

            # Use accumulated streaming_text if result.answer is incomplete
            # (agent's answer only captures last iteration, but streaming_text has everything)
            final_answer = streaming_text.strip() if streaming_text else (result.answer or "").strip()

            # Ensure we never show an empty message
            if not final_answer:
                if result.error:
                    final_answer = f"Agent stopped: {result.error}"
                elif result.sql_queries:
                    final_answer = f"Executed {len(result.sql_queries)} queries but no response was generated."
                else:
                    final_answer = "No response was generated. Please try rephrasing your question."

                # Debug notification for empty responses
                self.app.call_from_thread(
                    self.notify,
                    f"Debug: streaming={len(streaming_text)}, answer={len(result.answer) if result.answer else 0}, iters={result.timing.iterations if result.timing else '?'}",
                    severity="warning",
                )

            # Show final response (with SQL and raw results collapsible)
            self.app.call_from_thread(
                self._update_response,
                final_answer,
                last_sql,
                False,
                result.last_query_result,
                result.timing.summary() if result.timing else None,
            )

            # Add to conversation history
            app.conversation.add_assistant_message(
                content=final_answer,
                sql=last_sql,
                result_summary=f"Agent executed {len(result.sql_queries)} queries"
                if result.sql_queries
                else None,
            )

            # Record token usage
            if result.timing:
                app.config_storage.record_token_usage(
                    result.timing.input_tokens,
                    result.timing.output_tokens,
                )

        except LLMAuthenticationError as e:
            self.app.call_from_thread(
                self._show_error,
                f"{e}\n\nPlease run /login to update your API key.",
            )
        except LLMQuotaExceededError as e:
            self.app.call_from_thread(
                self._show_error,
                str(e),
            )
        except LLMRateLimitError as e:
            retry_msg = ""
            if e.retry_after:
                retry_msg = f" Try again in {e.retry_after} seconds."
            self.app.call_from_thread(
                self._show_error,
                f"{e}{retry_msg}",
            )
        except LLMServiceUnavailableError as e:
            self.app.call_from_thread(
                self._show_error,
                str(e),
            )
        except LLMInternalError as e:
            self.app.call_from_thread(
                self._show_error,
                str(e),
            )
        except LLMConnectionError as e:
            self.app.call_from_thread(
                self._show_error,
                str(e),
            )
        except LLMError as e:
            self.app.call_from_thread(
                self._show_error,
                str(e),
            )
        except Exception as e:
            self.app.call_from_thread(self._show_error, f"Error: {e}")

    def _update_response(
        self,
        content: str,
        sql: str | None,
        is_loading: bool,
        query_result: str | None = None,
        timing: str | None = None,
    ) -> None:
        """Update the assistant response (called from main thread)."""
        try:
            messages = self.query_one("#messages", MessageList)
            messages.update_last_message(
                ChatMessage(
                    role="assistant",
                    content=content,
                    sql=sql,
                    query_result=query_result,
                    timing=timing,
                    is_loading=is_loading,
                )
            )
        except Exception:
            pass  # Screen may have been dismissed

    def _update_streaming_response(self, content: str) -> None:
        """Update the assistant response during streaming (called from main thread)."""
        try:
            messages = self.query_one("#messages", MessageList)
            messages.update_last_message(
                ChatMessage(
                    role="assistant",
                    content=content,
                    sql=None,  # No SQL during streaming
                    is_loading=False,  # Show as regular message during streaming
                )
            )
        except Exception:
            pass  # Screen may have been dismissed

    def _update_with_status(self, content: str, status: str) -> None:
        """Update the assistant message with content and a status line below (called from main thread)."""
        try:
            messages = self.query_one("#messages", MessageList)
            messages.update_last_message(
                ChatMessage(
                    role="assistant",
                    content=content,
                    status=status,
                    is_loading=True,
                )
            )
        except Exception:
            pass  # Screen may have been dismissed

    def _show_error(self, error_message: str) -> None:
        """Show an error message (called from main thread)."""
        try:
            messages = self.query_one("#messages", MessageList)
            messages.update_last_message(
                ChatMessage(
                    role="assistant",
                    content=error_message,
                    is_error=True,
                )
            )
        except Exception:
            pass  # Screen may have been dismissed

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Re-enable input when worker completes."""
        if event.state.name in ("SUCCESS", "CANCELLED", "ERROR"):
            try:
                self.query_one("#input", CommandInput).disabled = False
                self.query_one("#input", CommandInput).focus()
            except Exception:
                pass  # Screen may have been dismissed

    def _reset_conversation(self) -> None:
        """Clear conversation history."""
        app: DalcedoApp = self.app  # type: ignore
        app.conversation.clear()
        messages = self.query_one("#messages", MessageList)
        messages.clear()
        messages.add_system_message("Conversation reset. Ask me anything!")
        self.notify("Conversation reset", severity="information")

    def _show_help(self) -> None:
        """Show help message."""
        messages = self.query_one("#messages", MessageList)
        help_text = """Available commands:
- /login - Configure data source and API key
- /sync - Sync data to local storage
- /reset - Clear conversation history
- /tables - List synced tables
- /schema - Show database schema
- /mode - Show current response mode
- /mode quick - Fast, concise responses
- /mode analytics - Detailed analysis and reports
- /context - Show custom context
- /context <text> - Set custom context (domain knowledge, terminology)
- /context clear - Clear custom context
- /traces - Show recent traces (last 5)
- /traces N - Show last N traces
- /usage - Show token usage (daily/weekly)
- /help - Show this help message

Keyboard shortcuts:
- Ctrl+L - Open login screen
- Ctrl+S - Open sync screen
- Ctrl+R - Reset conversation
- Ctrl+M - Toggle response mode
- Ctrl+U - Show token usage
- Up/Down - Navigate input history

Copy text:
- Hold Shift + click and drag to select, then Ctrl+C to copy"""
        messages.add_system_message(help_text)

    def _show_mode(self) -> None:
        """Show current LLM response mode."""
        app: DalcedoApp = self.app  # type: ignore
        messages = self.query_one("#messages", MessageList)

        if not app.agent_service:
            messages.add_system_message("Please run /login first.")
            return

        mode = app.agent_service.get_mode()
        description = app.agent_service.get_mode_description()
        messages.add_system_message(
            f"Current mode: **{mode}** - {description}\n\n"
            "Use `/mode quick` or `/mode analytics` to change."
        )

    def _set_mode(self, mode: str) -> None:
        """Set the LLM response mode."""
        app: DalcedoApp = self.app  # type: ignore
        messages = self.query_one("#messages", MessageList)

        if not app.agent_service:
            messages.add_system_message("Please run /login first.")
            return

        if mode not in ("quick", "analytics"):
            messages.add_system_message(f"Unknown mode: {mode}. Use 'quick' or 'analytics'.")
            return

        app.agent_service.set_mode(mode)  # type: ignore
        description = app.agent_service.get_mode_description()
        messages.add_system_message(f"Mode set to **{mode}** - {description}")
        self.notify(f"Mode: {mode}", severity="information")

    def _show_context(self) -> None:
        """Show current custom context."""
        app: DalcedoApp = self.app  # type: ignore
        messages = self.query_one("#messages", MessageList)

        if not app.agent_service:
            messages.add_system_message("Please run /login first.")
            return

        context = app.agent_service.get_custom_context()
        if context:
            messages.add_system_message(
                f"**Custom context:**\n\n{context}\n\n"
                "Use `/context clear` to remove or `/context <text>` to replace."
            )
        else:
            messages.add_system_message(
                "No custom context set.\n\n"
                "Use `/context <text>` to add domain knowledge, terminology, or other context.\n\n"
                "Example: `/context A DBT model is a SQL transformation that defines how raw data "
                "should be transformed. Models are typically organized by source and purpose.`"
            )

    def _set_context(self, context: str) -> None:
        """Set custom context for the agent."""
        app: DalcedoApp = self.app  # type: ignore
        messages = self.query_one("#messages", MessageList)

        if not app.agent_service:
            messages.add_system_message("Please run /login first.")
            return

        if not app.config:
            messages.add_system_message("Please run /login first.")
            return

        # Update agent
        app.agent_service.set_custom_context(context)

        # Save to config
        app.config.custom_context = context
        app.config_storage.save_config(app.config)

        messages.add_system_message(f"Custom context updated:\n\n{context}")
        self.notify("Context saved", severity="information")

    def _clear_context(self) -> None:
        """Clear custom context."""
        app: DalcedoApp = self.app  # type: ignore
        messages = self.query_one("#messages", MessageList)

        if not app.agent_service:
            messages.add_system_message("Please run /login first.")
            return

        if not app.config:
            messages.add_system_message("Please run /login first.")
            return

        # Clear from agent
        app.agent_service.set_custom_context(None)

        # Save to config
        app.config.custom_context = None
        app.config_storage.save_config(app.config)

        messages.add_system_message("Custom context cleared.")
        self.notify("Context cleared", severity="information")

    def _show_usage(self) -> None:
        """Show token usage statistics."""
        app: DalcedoApp = self.app  # type: ignore
        messages = self.query_one("#messages", MessageList)

        summary = app.config_storage.get_token_usage_summary()
        daily = summary["daily"]
        weekly = summary["weekly"]

        # Get limits from config
        daily_limit = app.config.daily_token_limit if app.config else None
        weekly_limit = app.config.weekly_token_limit if app.config else None

        lines = ["**Token Usage**\n"]

        # Daily usage
        daily_str = f"**Today**: {daily['total']:,} tokens ({daily['input']:,} in / {daily['output']:,} out)"
        if daily_limit:
            pct = (daily['total'] / daily_limit) * 100
            daily_str += f" — {pct:.1f}% of {daily_limit:,} limit"
        lines.append(daily_str)

        # Weekly usage
        weekly_str = f"**This week**: {weekly['total']:,} tokens ({weekly['input']:,} in / {weekly['output']:,} out)"
        if weekly_limit:
            pct = (weekly['total'] / weekly_limit) * 100
            weekly_str += f" — {pct:.1f}% of {weekly_limit:,} limit"
        lines.append(weekly_str)

        # Show limit status
        if not daily_limit and not weekly_limit:
            lines.append("\n_No token limits configured. Set limits in /login._")

        messages.add_system_message("\n".join(lines))

    def _check_token_limits(self) -> str | None:
        """Check if token limits are exceeded. Returns error message or None if OK."""
        app: DalcedoApp = self.app  # type: ignore

        if not app.config:
            return None

        summary = app.config_storage.get_token_usage_summary()

        # Check daily limit
        if app.config.daily_token_limit:
            if summary["daily"]["total"] >= app.config.daily_token_limit:
                return (
                    f"**Daily token limit reached** ({summary['daily']['total']:,} / {app.config.daily_token_limit:,})\n\n"
                    "Your daily token limit has been exceeded. Please wait until tomorrow or "
                    "increase your limit in /login."
                )

        # Check weekly limit
        if app.config.weekly_token_limit:
            if summary["weekly"]["total"] >= app.config.weekly_token_limit:
                return (
                    f"**Weekly token limit reached** ({summary['weekly']['total']:,} / {app.config.weekly_token_limit:,})\n\n"
                    "Your weekly token limit has been exceeded. Please wait until next week or "
                    "increase your limit in /login."
                )

        return None

    def _show_traces(self, count: int = 5) -> None:
        """Show recent traces for debugging."""
        import json
        from dalcedo.services.tracing import get_tracer

        messages = self.query_one("#messages", MessageList)
        tracer = get_tracer()

        if not tracer.trace_file.exists():
            messages.add_system_message("No traces found. Run a query first.")
            return

        # Read last N trace summaries
        traces: dict[str, dict] = {}
        try:
            with open(tracer.trace_file) as f:
                for line in f:
                    event = json.loads(line.strip())
                    trace_id = event.get("trace_id")
                    if not trace_id:
                        continue

                    if event["event_type"] == "trace_start":
                        traces[trace_id] = {
                            "question": event.get("question", "")[:60],
                            "timestamp": event.get("timestamp", ""),
                        }
                    elif event["event_type"] == "trace_end":
                        if trace_id in traces:
                            traces[trace_id].update({
                                "total_ms": event.get("total_ms", 0),
                                "iterations": event.get("iterations", 0),
                                "tokens": f"{event.get('input_tokens', 0)}→{event.get('output_tokens', 0)}",
                                "error": event.get("error"),
                            })
        except Exception as e:
            messages.add_system_message(f"Error reading traces: {e}")
            return

        # Get last N complete traces
        complete_traces = [
            (tid, t) for tid, t in traces.items()
            if "total_ms" in t
        ][-count:]

        if not complete_traces:
            messages.add_system_message("No complete traces found.")
            return

        lines = [f"**Last {len(complete_traces)} traces:**\n"]
        for trace_id, t in reversed(complete_traces):
            error_str = f" ❌ {t['error']}" if t.get("error") else ""
            lines.append(
                f"- `{trace_id}` | {t['total_ms']:.0f}ms | "
                f"{t['iterations']} iters | {t['tokens']} tokens{error_str}\n"
                f"  {t['question']}..."
            )

        lines.append(f"\nTrace file: `{tracer.trace_file}`")
        messages.add_system_message("\n".join(lines))

    def _show_tables(self) -> None:
        """Show list of synced tables organized by schema."""
        app: DalcedoApp = self.app  # type: ignore
        messages = self.query_one("#messages", MessageList)

        if not app.db_service:
            messages.add_system_message("Please run /login first.")
            return

        tables_by_schema = app.db_service.get_tables_by_schema()
        if tables_by_schema:
            total_tables = sum(len(t) for t in tables_by_schema.values())
            lines = [f"Synced tables ({total_tables}):"]
            for schema, tables in sorted(tables_by_schema.items()):
                lines.append(f"\n=== {schema} ===")
                for table in tables:
                    lines.append(f"- {schema}.{table}")
            messages.add_system_message("\n".join(lines))
        else:
            messages.add_system_message("No tables synced. Run /sync to sync your data sources.")

    def _show_schema(self) -> None:
        """Show database schema."""
        app: DalcedoApp = self.app  # type: ignore
        messages = self.query_one("#messages", MessageList)

        if not app.db_service:
            messages.add_system_message("Please run /login first.")
            return

        schema = app.db_service.get_schema_description()
        messages.add_system_message(f"Database Schema:\n\n{schema}")

    def action_login(self) -> None:
        """Open login screen."""
        self.app.push_screen("login")

    def action_sync(self) -> None:
        """Open sync screen."""
        app: DalcedoApp = self.app  # type: ignore
        if app.config:
            self.app.push_screen("sync")
        else:
            self.notify("Please run /login first", severity="warning")

    def action_reset(self) -> None:
        """Reset conversation."""
        self._reset_conversation()

    def action_toggle_mode(self) -> None:
        """Toggle between quick and analytics mode."""
        app: DalcedoApp = self.app  # type: ignore
        if not app.agent_service:
            self.notify("Please run /login first", severity="warning")
            return

        current = app.agent_service.get_mode()
        new_mode = "analytics" if current == "quick" else "quick"
        self._set_mode(new_mode)

    def action_usage(self) -> None:
        """Show token usage."""
        self._show_usage()
