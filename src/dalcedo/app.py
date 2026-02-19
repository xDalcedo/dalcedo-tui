"""Main Textual application for Dalcedo."""

from __future__ import annotations

from textual.app import App
from textual.binding import Binding

from dalcedo.config import AppConfig, ConfigStorage
from dalcedo.connectors import BaseConnector, get_connector, initialize_connectors
from dalcedo.models.conversation import Conversation
from dalcedo.screens.chat import ChatScreen
from dalcedo.screens.login import LoginScreen
from dalcedo.screens.sync import SyncScreen
from dalcedo.services.duckdb import DuckDBService
from dalcedo.services.llm import (
    AgentService,
    AnthropicLLMService,
    BaseLLMService,
    GeminiLLMService,
    OpenAILLMService,
)


class DalcedoApp(App):
    """BigQuery Natural Language Query Interface."""

    CSS_PATH = "styles/app.tcss"
    TITLE = "Dalcedo TUI - Chat with your data"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    SCREENS = {
        "chat": ChatScreen,
        "login": LoginScreen,
        "sync": SyncScreen,
    }

    def __init__(self) -> None:
        super().__init__()
        self.config_storage = ConfigStorage()
        self.config: AppConfig | None = None
        self.llm_service: BaseLLMService | None = None
        self.agent_service: AgentService | None = None
        self.db_service: DuckDBService | None = None
        # Multiple connectors, keyed by plugin name
        self.connectors: dict[str, BaseConnector] = {}
        self.conversation = Conversation()

        # Initialize connector registry
        initialize_connectors()

    def on_mount(self) -> None:
        """Initialize application state."""
        if self.config_storage.is_configured():
            self.config = self.config_storage.load_config()
            self.initialize_services()

        # Always start with chat screen
        self.push_screen("chat")

    def initialize_services(self) -> None:
        """Initialize service instances."""
        if not self.config or not self.config.credentials:
            return

        # Initialize LLM service based on provider
        model = self.config.get_llm_model()
        if self.config.llm_provider == "openai":
            self.llm_service = OpenAILLMService(
                api_key=self.config.credentials.llm_api_key,
                model=model,
            )
        elif self.config.llm_provider == "gemini":
            self.llm_service = GeminiLLMService(
                api_key=self.config.credentials.llm_api_key,
                model=model,
            )
        else:
            self.llm_service = AnthropicLLMService(
                api_key=self.config.credentials.llm_api_key,
                model=model,
            )

        # Initialize DuckDB
        self.db_service = DuckDBService(db_path=self.config_storage.duckdb_path)

        # Initialize connectors for all enabled plugins
        self.connectors = {}
        for plugin in self.config.get_enabled_plugins():
            connection = self.config.get_plugin_connection(plugin)
            if not connection:
                continue
            connector_cls = get_connector(connection.type)
            if connector_cls:
                # Pass both connection config and source config
                self.connectors[plugin.name] = connector_cls(
                    connection_config=connection.config,
                    source_config=plugin.source_config,
                )
                # Create schema for this plugin
                self.db_service.create_schema(plugin.schema_name)

        # Initialize agent service with LLM and DB
        self.agent_service = AgentService(
            llm=self.llm_service,
            db=self.db_service,
            max_iterations=10,
            custom_context=self.config.custom_context if self.config else None,
        )

    def action_quit(self) -> None:
        """Quit the application."""
        if self.db_service:
            self.db_service.close()
        self.exit()
