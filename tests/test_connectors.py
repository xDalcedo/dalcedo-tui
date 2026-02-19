"""Tests for connector system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dalcedo.connectors.base import BaseConnector, ConnectorField, SyncUpdate
from dalcedo.connectors.registry import (
    _CONNECTORS,
    get_all_connectors,
    get_connector,
    initialize_connectors,
    list_connector_names,
    register_connector,
)
from dalcedo.services.duckdb import DuckDBService

if TYPE_CHECKING:
    pass


# Inline MockConnector since conftest fixtures can't be imported directly
class MockConnector(BaseConnector):
    """Mock connector for testing with split config."""

    name = "mock"
    display_name = "Mock Connector"
    description = "A mock connector for testing"

    @classmethod
    def get_connection_fields(cls) -> list[ConnectorField]:
        """Connection-level fields (shared)."""
        return [
            ConnectorField(name="api_key", label="API Key", field_type="password", required=True),
            ConnectorField(name="project", label="Project", required=True),
        ]

    @classmethod
    def get_source_fields(cls) -> list[ConnectorField]:
        """Source-level fields (per plugin)."""
        return [
            ConnectorField(name="dataset", label="Dataset", required=True),
            ConnectorField(name="optional_field", label="Optional", required=False),
        ]

    async def test_connection(self) -> bool:
        if self.connection_config.get("api_key") == "invalid":
            raise ValueError("Invalid API key")
        return True

    def sync(self, db: DuckDBService, schema_name: str = "main"):
        yield SyncUpdate("Starting sync...", 0.0, schema_name=schema_name)
        yield SyncUpdate("Syncing table1...", 0.5, table="table1", schema_name=schema_name)
        yield SyncUpdate("Sync complete!", 1.0, schema_name=schema_name)

    def get_schema_context(self, db: DuckDBService) -> str:
        return "Mock schema context"


class TestConnectorField:
    """Tests for ConnectorField dataclass."""

    def test_basic_field(self):
        """Test basic field creation."""
        field = ConnectorField(name="test", label="Test Field")
        assert field.name == "test"
        assert field.label == "Test Field"
        assert field.field_type == "text"
        assert field.required is True
        assert field.default == ""

    def test_password_field(self):
        """Test password field type."""
        field = ConnectorField(
            name="api_key",
            label="API Key",
            field_type="password",
            required=True,
        )
        assert field.field_type == "password"

    def test_optional_field(self):
        """Test optional field with default."""
        field = ConnectorField(
            name="region",
            label="Region",
            required=False,
            default="us-central1",
        )
        assert field.required is False
        assert field.default == "us-central1"

    def test_select_field(self):
        """Test select field with options."""
        field = ConnectorField(
            name="provider",
            label="Provider",
            field_type="select",
            options=[("aws", "AWS"), ("gcp", "GCP")],
        )
        assert field.field_type == "select"
        assert len(field.options) == 2


class TestSyncUpdate:
    """Tests for SyncUpdate dataclass."""

    def test_basic_update(self):
        """Test basic sync update."""
        update = SyncUpdate(message="Syncing...", progress=0.5)
        assert update.message == "Syncing..."
        assert update.progress == 0.5
        assert update.table is None
        assert update.schema_name is None
        assert update.is_error is False

    def test_table_update(self):
        """Test sync update with table name."""
        update = SyncUpdate(message="Syncing users", progress=0.3, table="users")
        assert update.table == "users"

    def test_schema_update(self):
        """Test sync update with schema name."""
        update = SyncUpdate(message="Syncing to analytics", progress=0.3, schema_name="analytics")
        assert update.schema_name == "analytics"

    def test_error_update(self):
        """Test error sync update."""
        update = SyncUpdate(message="Failed to sync", progress=0.5, is_error=True)
        assert update.is_error is True


class TestMockConnector:
    """Tests for MockConnector implementation."""

    def test_connector_metadata(self):
        """Test connector has required metadata."""
        assert MockConnector.name == "mock"
        assert MockConnector.display_name == "Mock Connector"
        assert MockConnector.description is not None

    def test_get_config_fields(self):
        """Test config fields are defined (combined connection + source)."""
        fields = MockConnector.get_config_fields()
        assert len(fields) >= 3
        field_names = [f.name for f in fields]
        assert "api_key" in field_names
        assert "project" in field_names
        assert "dataset" in field_names

    def test_get_connection_fields(self):
        """Test connection-level fields."""
        fields = MockConnector.get_connection_fields()
        field_names = [f.name for f in fields]
        assert "api_key" in field_names
        assert "project" in field_names
        assert "dataset" not in field_names

    def test_get_source_fields(self):
        """Test source-level fields."""
        fields = MockConnector.get_source_fields()
        field_names = [f.name for f in fields]
        assert "dataset" in field_names
        assert "api_key" not in field_names

    def test_validate_connection_config_valid(self):
        """Test validation with valid connection config."""
        config = {"api_key": "test", "project": "test-project"}
        errors = MockConnector.validate_connection_config(config)
        assert errors == []

    def test_validate_connection_config_missing_required(self):
        """Test validation with missing required field."""
        config = {"api_key": "test"}  # Missing project
        errors = MockConnector.validate_connection_config(config)
        assert len(errors) > 0
        assert any("Project" in e for e in errors)

    def test_validate_source_config_valid(self):
        """Test validation with valid source config."""
        config = {"dataset": "my_dataset"}
        errors = MockConnector.validate_source_config(config)
        assert errors == []

    def test_validate_source_config_missing_required(self):
        """Test validation with missing required source field."""
        config = {}  # Missing dataset
        errors = MockConnector.validate_source_config(config)
        assert len(errors) > 0
        assert any("Dataset" in e for e in errors)

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector for this test class."""
        return MockConnector(
            connection_config={"api_key": "test_key", "project": "test_project"},
            source_config={"dataset": "test_dataset"},
        )

    @pytest.mark.asyncio
    async def test_test_connection_success(self, mock_connector):
        """Test successful connection."""
        result = await mock_connector.test_connection()
        assert result is True

    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """Test connection failure with invalid key."""
        connector = MockConnector(
            connection_config={"api_key": "invalid", "project": "test"},
            source_config={"dataset": "test"},
        )
        with pytest.raises(ValueError, match="Invalid API key"):
            await connector.test_connection()

    def test_sync_yields_updates(self, mock_connector, temp_duckdb):
        """Test sync yields progress updates."""
        updates = list(mock_connector.sync(temp_duckdb))
        assert len(updates) >= 2
        # Check progress values
        assert updates[0].progress == 0.0
        assert updates[-1].progress == 1.0

    def test_get_schema_context(self, mock_connector, temp_duckdb):
        """Test schema context generation."""
        context = mock_connector.get_schema_context(temp_duckdb)
        assert isinstance(context, str)
        assert len(context) > 0


class TestConnectorRegistry:
    """Tests for connector registry."""

    def test_register_connector(self):
        """Test registering a connector."""
        # Clear registry for test isolation
        _CONNECTORS.clear()

        @register_connector
        class TestConnector(BaseConnector):
            name = "test_connector"
            display_name = "Test"
            description = "Test connector"

            @classmethod
            def get_connection_fields(cls):
                return []

            @classmethod
            def get_source_fields(cls):
                return []

            async def test_connection(self):
                return True

            def sync(self, db, schema_name: str = "main"):
                yield SyncUpdate("Done", 1.0, schema_name=schema_name)

            def get_schema_context(self, db):
                return ""

        assert "test_connector" in _CONNECTORS
        assert _CONNECTORS["test_connector"] is TestConnector

    def test_get_connector(self):
        """Test getting connector by name."""
        _CONNECTORS.clear()
        _CONNECTORS["mock"] = MockConnector

        result = get_connector("mock")
        assert result is MockConnector

        result = get_connector("nonexistent")
        assert result is None

    def test_get_all_connectors(self):
        """Test getting all connectors."""
        _CONNECTORS.clear()
        _CONNECTORS["mock"] = MockConnector

        result = get_all_connectors()
        assert "mock" in result
        # Verify it's a copy
        result["new"] = None
        assert "new" not in _CONNECTORS

    def test_list_connector_names(self):
        """Test listing connector names."""
        _CONNECTORS.clear()
        _CONNECTORS["mock"] = MockConnector
        _CONNECTORS["other"] = MockConnector

        names = list_connector_names()
        assert "mock" in names
        assert "other" in names

    def test_initialize_connectors(self):
        """Test initializing connectors loads builtins."""
        _CONNECTORS.clear()
        initialize_connectors()
        # Should have at least bigquery connector
        assert "bigquery" in _CONNECTORS
