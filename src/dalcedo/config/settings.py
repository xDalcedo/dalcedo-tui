"""Configuration dataclasses for Dalcedo."""

import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Literal

# Supported LLM providers
LLMProvider = Literal["anthropic", "openai", "gemini"]

# LLM response modes
LLMMode = Literal["quick", "analytics"]

# Default models for each provider
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
}


def _generate_id() -> str:
    """Generate a short random ID."""
    return secrets.token_hex(3)  # 6 characters


def _sanitize_schema_name(name: str) -> str:
    """Sanitize a name to be a valid DuckDB schema name."""
    # Replace non-alphanumeric chars with underscores, lowercase
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())
    # Ensure it starts with a letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "plugin"


@dataclass(frozen=True)
class Credentials:
    """Sensitive credentials."""

    llm_api_key: str  # API key for the selected LLM provider


@dataclass
class Connection:
    """A connection to a data warehouse or storage system.

    Connections hold the credentials and connection details for a warehouse.
    Multiple plugins can share the same connection.
    """

    id: str  # Auto-generated, e.g., "abc123"
    name: str  # User-friendly name, e.g., "Production BigQuery"
    type: str  # Connector type: "bigquery", "snowflake", etc.
    config: dict[str, Any] = field(default_factory=dict)  # Connection-level config

    @classmethod
    def create(
        cls,
        name: str,
        connection_type: str,
        config: dict[str, Any] | None = None,
    ) -> "Connection":
        """Create a new connection with auto-generated ID."""
        return cls(
            id=_generate_id(),
            name=name,
            type=connection_type,
            config=config or {},
        )


@dataclass
class Plugin:
    """A data source plugin configuration.

    Plugins represent a specific data location within a connection.
    For example, a BigQuery dataset within a BigQuery project.
    """

    id: str  # Auto-generated, e.g., "abc123"
    name: str  # User-friendly name, becomes DuckDB schema name
    connection_id: str  # References a Connection
    source_config: dict[str, Any] = field(default_factory=dict)  # Source-specific config
    enabled: bool = True

    @property
    def schema_name(self) -> str:
        """Get the DuckDB schema name (sanitized version of name)."""
        return _sanitize_schema_name(self.name)

    @classmethod
    def create(
        cls,
        name: str,
        connection_id: str,
        source_config: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> "Plugin":
        """Create a new plugin with auto-generated ID."""
        return cls(
            id=_generate_id(),
            name=name,
            connection_id=connection_id,
            source_config=source_config or {},
            enabled=enabled,
        )


@dataclass
class AppConfig:
    """Complete application configuration."""

    # Connections (data warehouses/storage systems)
    connections: list[Connection] = field(default_factory=list)

    # Plugins (data sources within connections)
    plugins: list[Plugin] = field(default_factory=list)

    # LLM settings
    credentials: Credentials | None = None
    llm_provider: LLMProvider = "anthropic"
    llm_model: str | None = None  # None = use default for provider
    max_context_messages: int = 20

    # Custom context for the LLM (domain knowledge, terminology, etc.)
    custom_context: str | None = None

    # Token usage limits (None = unlimited)
    daily_token_limit: int | None = None
    weekly_token_limit: int | None = None

    def get_llm_model(self) -> str:
        """Get the LLM model, using provider default if not set."""
        return self.llm_model or DEFAULT_MODELS.get(self.llm_provider, "claude-sonnet-4-5-20250929")

    def get_enabled_plugins(self) -> list[Plugin]:
        """Get list of enabled plugins."""
        return [p for p in self.plugins if p.enabled]

    def get_plugin_by_id(self, plugin_id: str) -> Plugin | None:
        """Get a plugin by its ID."""
        for p in self.plugins:
            if p.id == plugin_id:
                return p
        return None

    def get_plugin_by_name(self, name: str) -> Plugin | None:
        """Get a plugin by its name."""
        for p in self.plugins:
            if p.name == name:
                return p
        return None

    def get_connection_by_id(self, connection_id: str) -> Connection | None:
        """Get a connection by its ID."""
        for c in self.connections:
            if c.id == connection_id:
                return c
        return None

    def get_connection_by_name(self, name: str) -> Connection | None:
        """Get a connection by its name."""
        for c in self.connections:
            if c.name == name:
                return c
        return None

    def get_plugin_connection(self, plugin: Plugin) -> Connection | None:
        """Get the connection for a plugin."""
        return self.get_connection_by_id(plugin.connection_id)
