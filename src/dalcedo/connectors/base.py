"""Base connector interface for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from dalcedo.services.duckdb import DuckDBService


@dataclass
class ConnectorConfig:
    """Base configuration for connectors."""

    pass


@dataclass
class ConnectorField:
    """Definition of a configuration field for UI rendering."""

    name: str
    label: str
    field_type: str = "text"  # "text", "password", "select", "file"
    placeholder: str = ""
    required: bool = True
    default: str = ""
    options: list[tuple[str, str]] = field(default_factory=list)  # For select fields


@dataclass
class SyncUpdate:
    """Progress update during sync."""

    message: str
    progress: float  # 0.0 to 1.0
    table: str | None = None
    schema_name: str | None = None  # The schema being synced to
    is_error: bool = False


class BaseConnector(ABC):
    """Abstract base class for data source connectors.

    Connectors have two types of configuration:
    1. Connection config - credentials and warehouse-level settings (shared across plugins)
    2. Source config - data location within the warehouse (per plugin)

    Connectors are responsible for:
    1. Defining their configuration fields (both connection and source)
    2. Validating and testing connections
    3. Syncing data to the local DuckDB database
    4. Providing schema context for the LLM agent
    """

    # Connector metadata - override in subclasses
    name: str = "base"
    display_name: str = "Base Connector"
    description: str = "Base connector interface"

    def __init__(self, connection_config: dict[str, Any], source_config: dict[str, Any]):
        """Initialize connector with configuration.

        Args:
            connection_config: Warehouse-level config (credentials, project, etc.)
            source_config: Source-specific config (dataset, index, etc.)
        """
        self.connection_config = connection_config
        self.source_config = source_config

    @classmethod
    @abstractmethod
    def get_connection_fields(cls) -> list[ConnectorField]:
        """Return the connection-level configuration fields.

        These are shared across all plugins using the same connection.
        Examples: project_id, credentials_path, account, warehouse
        """
        pass

    @classmethod
    @abstractmethod
    def get_source_fields(cls) -> list[ConnectorField]:
        """Return the source-specific configuration fields.

        These are unique to each plugin/data source.
        Examples: dataset_id, schema, index_pattern, dbt_schema_paths
        """
        pass

    @classmethod
    def get_config_fields(cls) -> list[ConnectorField]:
        """Return all configuration fields (for backwards compatibility).

        Returns connection fields followed by source fields.
        """
        return cls.get_connection_fields() + cls.get_source_fields()

    @classmethod
    def validate_connection_config(cls, config: dict[str, Any]) -> list[str]:
        """Validate connection configuration and return list of errors."""
        errors = []
        for cfg_field in cls.get_connection_fields():
            if cfg_field.required and not config.get(cfg_field.name):
                errors.append(f"{cfg_field.label} is required")
        return errors

    @classmethod
    def validate_source_config(cls, config: dict[str, Any]) -> list[str]:
        """Validate source configuration and return list of errors."""
        errors = []
        for cfg_field in cls.get_source_fields():
            if cfg_field.required and not config.get(cfg_field.name):
                errors.append(f"{cfg_field.label} is required")
        return errors

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[str]:
        """Validate full configuration (for backwards compatibility)."""
        return cls.validate_connection_config(config) + cls.validate_source_config(config)

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test the connection to the data source.

        Raises:
            Exception: If connection fails
        """
        pass

    @abstractmethod
    def sync(self, db: "DuckDBService", schema_name: str = "main") -> Iterator[SyncUpdate]:
        """Sync data from source to local DuckDB.

        Args:
            db: The DuckDB service instance
            schema_name: The schema to sync tables into

        Yields SyncUpdate objects to report progress.
        """
        pass

    @abstractmethod
    def get_schema_context(self, db: "DuckDBService") -> str:
        """Return schema description for LLM context.

        This should describe what data is available and how to query it.
        """
        pass

    def get_agent_tools(self) -> list[dict] | None:
        """Return additional tools specific to this connector.

        Override to provide connector-specific tools for the agent.
        Returns None to use only default tools.
        """
        return None
