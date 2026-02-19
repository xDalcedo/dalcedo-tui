"""Tests for configuration and storage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dalcedo.config.settings import AppConfig, Connection, Credentials, DEFAULT_MODELS, Plugin


class TestCredentials:
    """Tests for Credentials dataclass."""

    def test_create_credentials(self):
        """Test creating credentials."""
        creds = Credentials(llm_api_key="sk-test-123")
        assert creds.llm_api_key == "sk-test-123"

    def test_credentials_frozen(self):
        """Test credentials are immutable."""
        creds = Credentials(llm_api_key="sk-test-123")
        with pytest.raises(Exception):  # FrozenInstanceError
            creds.llm_api_key = "new-key"


class TestConnection:
    """Tests for Connection dataclass."""

    def test_create_connection(self):
        """Test creating a connection directly."""
        conn = Connection(
            id="conn123",
            name="Production BQ",
            type="bigquery",
            config={"project_id": "my-project", "location": "US"},
        )
        assert conn.id == "conn123"
        assert conn.name == "Production BQ"
        assert conn.type == "bigquery"
        assert conn.config["project_id"] == "my-project"

    def test_create_connection_factory(self):
        """Test creating a connection with auto-generated ID."""
        conn = Connection.create(
            name="Production BQ",
            connection_type="bigquery",
            config={"project_id": "my-project"},
        )
        assert len(conn.id) == 6  # hex string from secrets.token_hex(3)
        assert conn.name == "Production BQ"
        assert conn.type == "bigquery"


class TestPlugin:
    """Tests for Plugin dataclass."""

    def test_create_plugin(self):
        """Test creating a plugin directly."""
        plugin = Plugin(
            id="abc123",
            name="analytics",
            connection_id="conn123",
            source_config={"dataset_id": "my-dataset"},
            enabled=True,
        )
        assert plugin.id == "abc123"
        assert plugin.name == "analytics"
        assert plugin.connection_id == "conn123"
        assert plugin.source_config["dataset_id"] == "my-dataset"
        assert plugin.enabled is True

    def test_create_plugin_factory(self):
        """Test creating a plugin with auto-generated ID."""
        plugin = Plugin.create(
            name="analytics",
            connection_id="conn123",
            source_config={"dataset_id": "my-dataset"},
        )
        assert len(plugin.id) == 6  # hex string from secrets.token_hex(3)
        assert plugin.name == "analytics"
        assert plugin.enabled is True

    def test_plugin_schema_name(self):
        """Test schema name sanitization."""
        plugin = Plugin(id="abc", name="My Analytics", connection_id="conn")
        assert plugin.schema_name == "my_analytics"

    def test_plugin_schema_name_numeric_start(self):
        """Test schema name with numeric start gets underscore prefix."""
        plugin = Plugin(id="abc", name="123data", connection_id="conn")
        assert plugin.schema_name == "_123data"


class TestAppConfig:
    """Tests for AppConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AppConfig()
        assert config.connections == []
        assert config.plugins == []
        assert config.credentials is None
        assert config.llm_provider == "anthropic"
        assert config.llm_model is None
        assert config.max_context_messages == 20

    def test_custom_values(self):
        """Test custom configuration values."""
        conn = Connection.create(name="BQ", connection_type="bigquery")
        plugin = Plugin.create(name="analytics", connection_id=conn.id)
        config = AppConfig(
            connections=[conn],
            plugins=[plugin],
            credentials=Credentials(llm_api_key="test-key"),
            llm_provider="openai",
            llm_model="gpt-4-turbo",
            max_context_messages=10,
        )
        assert len(config.connections) == 1
        assert len(config.plugins) == 1
        assert config.plugins[0].name == "analytics"
        assert config.credentials.llm_api_key == "test-key"
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4-turbo"
        assert config.max_context_messages == 10

    def test_get_llm_model_custom(self):
        """Test getting custom LLM model."""
        config = AppConfig(llm_model="claude-opus-4-20250514")
        assert config.get_llm_model() == "claude-opus-4-20250514"

    def test_get_llm_model_default_anthropic(self):
        """Test default model for Anthropic."""
        config = AppConfig(llm_provider="anthropic")
        assert config.get_llm_model() == DEFAULT_MODELS["anthropic"]

    def test_get_llm_model_default_openai(self):
        """Test default model for OpenAI."""
        config = AppConfig(llm_provider="openai")
        assert config.get_llm_model() == DEFAULT_MODELS["openai"]

    def test_get_llm_model_default_gemini(self):
        """Test default model for Gemini."""
        config = AppConfig(llm_provider="gemini")
        assert config.get_llm_model() == DEFAULT_MODELS["gemini"]

    def test_get_enabled_plugins(self):
        """Test filtering to only enabled plugins."""
        p1 = Plugin(id="1", name="enabled", connection_id="c", enabled=True)
        p2 = Plugin(id="2", name="disabled", connection_id="c", enabled=False)
        p3 = Plugin(id="3", name="also_enabled", connection_id="c", enabled=True)
        config = AppConfig(plugins=[p1, p2, p3])

        enabled = config.get_enabled_plugins()
        assert len(enabled) == 2
        assert enabled[0].name == "enabled"
        assert enabled[1].name == "also_enabled"

    def test_get_plugin_by_id(self):
        """Test finding plugin by ID."""
        p1 = Plugin(id="abc123", name="plugin1", connection_id="c")
        p2 = Plugin(id="def456", name="plugin2", connection_id="c")
        config = AppConfig(plugins=[p1, p2])

        found = config.get_plugin_by_id("def456")
        assert found is not None
        assert found.name == "plugin2"

        not_found = config.get_plugin_by_id("xyz999")
        assert not_found is None

    def test_get_plugin_by_name(self):
        """Test finding plugin by name."""
        p1 = Plugin(id="abc123", name="analytics", connection_id="c")
        p2 = Plugin(id="def456", name="dbt_tests", connection_id="c")
        config = AppConfig(plugins=[p1, p2])

        found = config.get_plugin_by_name("analytics")
        assert found is not None
        assert found.id == "abc123"

        not_found = config.get_plugin_by_name("nonexistent")
        assert not_found is None

    def test_get_connection_by_id(self):
        """Test finding connection by ID."""
        c1 = Connection(id="conn1", name="BQ1", type="bigquery")
        c2 = Connection(id="conn2", name="BQ2", type="bigquery")
        config = AppConfig(connections=[c1, c2])

        found = config.get_connection_by_id("conn2")
        assert found is not None
        assert found.name == "BQ2"

        not_found = config.get_connection_by_id("xyz")
        assert not_found is None

    def test_get_plugin_connection(self):
        """Test getting connection for a plugin."""
        conn = Connection(id="conn1", name="BQ", type="bigquery")
        plugin = Plugin(id="p1", name="analytics", connection_id="conn1")
        config = AppConfig(connections=[conn], plugins=[plugin])

        found_conn = config.get_plugin_connection(plugin)
        assert found_conn is not None
        assert found_conn.id == "conn1"


class TempConfigStorage:
    """ConfigStorage with custom directories for testing."""

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        from dalcedo.config.storage import ConfigStorage

        self._storage = ConfigStorage()

    @property
    def config_dir(self) -> Path:
        return self._base_dir / "config"

    @property
    def data_dir(self) -> Path:
        return self._base_dir / "data"

    @property
    def cache_dir(self) -> Path:
        return self._base_dir / "cache"

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.toml"

    @property
    def credentials_file(self) -> Path:
        return self.data_dir / "credentials.json"

    @property
    def duckdb_path(self) -> Path:
        return self.cache_dir / "local.duckdb"

    @property
    def _metadata_file(self) -> Path:
        return self.cache_dir / "metadata.json"

    def _ensure_dirs(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.data_dir, 0o700)

    def _load_metadata(self) -> dict:
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata: dict) -> None:
        self._ensure_dirs()
        with open(self._metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_config(self) -> AppConfig:
        from dalcedo.config.storage import ConfigStorage

        # Monkey-patch the storage instance to use our temp dirs
        storage = ConfigStorage()
        storage.__class__.config_dir = property(lambda s: self.config_dir)
        storage.__class__.data_dir = property(lambda s: self.data_dir)
        storage.__class__.cache_dir = property(lambda s: self.cache_dir)
        return storage.load_config()

    def save_config(self, config: AppConfig) -> None:
        from dalcedo.config.storage import ConfigStorage

        storage = ConfigStorage()
        storage.__class__.config_dir = property(lambda s: self.config_dir)
        storage.__class__.data_dir = property(lambda s: self.data_dir)
        storage.__class__.cache_dir = property(lambda s: self.cache_dir)
        storage.save_config(config)

    def is_configured(self) -> bool:
        return self.config_file.exists() and self.credentials_file.exists()

    def clear_cache(self) -> None:
        if self.duckdb_path.exists():
            self.duckdb_path.unlink()
        wal_path = self.cache_dir / "local.duckdb.wal"
        if wal_path.exists():
            wal_path.unlink()

    def record_sync_time(self) -> None:
        from datetime import datetime

        metadata = self._load_metadata()
        metadata["last_sync"] = datetime.now().isoformat()
        self._save_metadata(metadata)

    def record_plugin_sync_time(self, plugin_id: str) -> None:
        from datetime import datetime

        metadata = self._load_metadata()
        if "plugin_syncs" not in metadata:
            metadata["plugin_syncs"] = {}
        metadata["plugin_syncs"][plugin_id] = datetime.now().isoformat()
        metadata["last_sync"] = datetime.now().isoformat()
        self._save_metadata(metadata)

    def get_plugin_sync_time(self, plugin_id: str):
        from datetime import datetime

        metadata = self._load_metadata()
        plugin_syncs = metadata.get("plugin_syncs", {})
        sync_time = plugin_syncs.get(plugin_id)
        if sync_time:
            return datetime.fromisoformat(sync_time)
        return None

    def record_sync_suggestion_time(self) -> None:
        from datetime import datetime

        metadata = self._load_metadata()
        metadata["last_sync_suggestion"] = datetime.now().isoformat()
        self._save_metadata(metadata)

    def get_last_sync_time(self):
        from datetime import datetime

        metadata = self._load_metadata()
        last_sync = metadata.get("last_sync")
        if last_sync:
            return datetime.fromisoformat(last_sync)
        return None

    def should_suggest_sync(
        self, sync_threshold_hours: int = 24, suggestion_threshold_hours: int = 12
    ) -> bool:
        from datetime import datetime, timedelta

        metadata = self._load_metadata()
        now = datetime.now()

        last_sync_str = metadata.get("last_sync")
        if last_sync_str:
            last_sync = datetime.fromisoformat(last_sync_str)
            if now - last_sync < timedelta(hours=sync_threshold_hours):
                return False

        last_suggestion_str = metadata.get("last_sync_suggestion")
        if last_suggestion_str:
            last_suggestion = datetime.fromisoformat(last_suggestion_str)
            if now - last_suggestion < timedelta(hours=suggestion_threshold_hours):
                return False

        return True

    def get_sync_age_description(self) -> str | None:
        from datetime import datetime

        last_sync = self.get_last_sync_time()
        if not last_sync:
            return None

        age = datetime.now() - last_sync
        if age.days > 0:
            return f"{age.days} day{'s' if age.days > 1 else ''} ago"
        hours = age.seconds // 3600
        if hours > 0:
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        minutes = age.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"


class TestConfigStorage:
    """Tests for ConfigStorage."""

    @pytest.fixture
    def temp_storage(self, temp_dir):
        """Create a ConfigStorage with temp directories."""
        return TempConfigStorage(temp_dir)

    def test_directory_paths(self, temp_storage):
        """Test directory path properties."""
        assert "config" in str(temp_storage.config_dir)
        assert "data" in str(temp_storage.data_dir)
        assert "cache" in str(temp_storage.cache_dir)

    def test_file_paths(self, temp_storage):
        """Test file path properties."""
        assert temp_storage.config_file.name == "config.toml"
        assert temp_storage.credentials_file.name == "credentials.json"
        assert temp_storage.duckdb_path.name == "local.duckdb"

    def test_is_configured_false(self, temp_storage):
        """Test is_configured returns False when files don't exist."""
        assert temp_storage.is_configured() is False

    def test_save_and_load_config(self, temp_storage):
        """Test saving and loading configuration with connections and plugins."""
        conn = Connection(
            id="conn123",
            name="Production BQ",
            type="bigquery",
            config={"project_id": "my-project", "location": "US"},
        )
        plugin = Plugin(
            id="abc123",
            name="analytics",
            connection_id="conn123",
            source_config={"dataset_id": "my-dataset"},
            enabled=True,
        )
        config = AppConfig(
            connections=[conn],
            plugins=[plugin],
            credentials=Credentials(llm_api_key="sk-test-123"),
            llm_provider="anthropic",
            max_context_messages=15,
        )

        temp_storage.save_config(config)

        # Check files were created
        assert temp_storage.config_file.exists()
        assert temp_storage.credentials_file.exists()

        # Load and verify
        loaded = temp_storage.load_config()
        assert len(loaded.connections) == 1
        assert loaded.connections[0].id == "conn123"
        assert loaded.connections[0].config["project_id"] == "my-project"
        assert len(loaded.plugins) == 1
        assert loaded.plugins[0].id == "abc123"
        assert loaded.plugins[0].name == "analytics"
        assert loaded.plugins[0].connection_id == "conn123"
        assert loaded.plugins[0].source_config["dataset_id"] == "my-dataset"
        assert loaded.credentials.llm_api_key == "sk-test-123"
        assert loaded.llm_provider == "anthropic"
        assert loaded.max_context_messages == 15

    def test_save_config_permissions(self, temp_storage):
        """Test credentials file has secure permissions."""
        config = AppConfig(
            credentials=Credentials(llm_api_key="secret"),
        )
        temp_storage.save_config(config)

        # Check permissions (should be 600)
        mode = os.stat(temp_storage.credentials_file).st_mode & 0o777
        assert mode == 0o600

    def test_save_config_filters_sensitive(self, temp_storage):
        """Test sensitive fields are not stored in config.toml."""
        conn = Connection(
            id="conn123",
            name="BQ",
            type="bigquery",
            config={
                "project_id": "my-project",
                "credentials_path": "/secret/path",
            },
        )
        config = AppConfig(
            connections=[conn],
            credentials=Credentials(llm_api_key="llm-key"),
        )
        temp_storage.save_config(config)

        # Read config.toml directly
        import tomllib

        with open(temp_storage.config_file, "rb") as f:
            data = tomllib.load(f)

        conn_config = data["connections"][0]["config"]
        assert "project_id" in conn_config
        assert "credentials_path" not in conn_config

    def test_is_configured_true(self, temp_storage):
        """Test is_configured returns True after saving."""
        config = AppConfig(credentials=Credentials(llm_api_key="test"))
        temp_storage.save_config(config)
        assert temp_storage.is_configured() is True

    def test_clear_cache(self, temp_storage):
        """Test clearing cache removes DuckDB files."""
        # Create dummy cache files
        temp_storage.cache_dir.mkdir(parents=True, exist_ok=True)
        temp_storage.duckdb_path.touch()
        wal_path = temp_storage.cache_dir / "local.duckdb.wal"
        wal_path.touch()

        assert temp_storage.duckdb_path.exists()
        assert wal_path.exists()

        temp_storage.clear_cache()

        assert not temp_storage.duckdb_path.exists()
        assert not wal_path.exists()

    def test_load_empty_config(self, temp_storage):
        """Test loading when no config exists returns defaults."""
        config = temp_storage.load_config()
        assert config.connections == []
        assert config.plugins == []
        assert config.credentials is None
        assert config.llm_provider == "anthropic"

    def test_backwards_compatibility_connector_format(self, temp_storage):
        """Test loading old connector config format (single connector)."""
        # Create old format config
        temp_storage.config_dir.mkdir(parents=True, exist_ok=True)
        temp_storage.data_dir.mkdir(parents=True, exist_ok=True)

        old_config = """
[connector]
type = "bigquery"
[connector.config]
project_id = "old-project"
dataset_id = "old-dataset"

[llm]
provider = "openai"
"""
        with open(temp_storage.config_file, "w") as f:
            f.write(old_config)

        old_creds = {
            "llm_api_key": "old-api-key",
        }
        with open(temp_storage.credentials_file, "w") as f:
            json.dump(old_creds, f)

        # Load and verify migration
        config = temp_storage.load_config()
        # Should have migrated to connection + plugin
        assert len(config.connections) == 1
        assert config.connections[0].type == "bigquery"
        assert config.connections[0].config["project_id"] == "old-project"
        assert len(config.plugins) == 1
        assert config.plugins[0].name == "default"
        assert config.plugins[0].source_config["dataset_id"] == "old-dataset"
        assert config.credentials.llm_api_key == "old-api-key"
        assert config.llm_provider == "openai"

    def test_record_and_get_sync_time(self, temp_storage):
        """Test recording and retrieving sync time."""
        # Initially no sync time
        assert temp_storage.get_last_sync_time() is None

        # Record sync time
        temp_storage.record_sync_time()

        # Should have a sync time now
        sync_time = temp_storage.get_last_sync_time()
        assert sync_time is not None
        # Should be very recent (within last minute)
        from datetime import datetime, timedelta

        assert datetime.now() - sync_time < timedelta(minutes=1)

    def test_record_and_get_plugin_sync_time(self, temp_storage):
        """Test recording and retrieving per-plugin sync time."""
        plugin_id = "abc123"

        # Initially no sync time
        assert temp_storage.get_plugin_sync_time(plugin_id) is None

        # Record sync time
        temp_storage.record_plugin_sync_time(plugin_id)

        # Should have a sync time now
        sync_time = temp_storage.get_plugin_sync_time(plugin_id)
        assert sync_time is not None
        from datetime import datetime, timedelta

        assert datetime.now() - sync_time < timedelta(minutes=1)

        # Global sync time should also be set
        assert temp_storage.get_last_sync_time() is not None

    def test_should_suggest_sync_never_synced(self, temp_storage):
        """Test sync suggestion when never synced."""
        # Never synced, should suggest
        assert temp_storage.should_suggest_sync() is True

    def test_should_suggest_sync_recently_synced(self, temp_storage):
        """Test sync suggestion when recently synced."""
        # Record a sync
        temp_storage.record_sync_time()

        # Should not suggest (synced just now)
        assert temp_storage.should_suggest_sync() is False

    def test_should_suggest_sync_recently_suggested(self, temp_storage):
        """Test sync suggestion when recently suggested."""
        # Never synced but recently suggested
        temp_storage.record_sync_suggestion_time()

        # Should not suggest (suggested recently)
        assert temp_storage.should_suggest_sync() is False

    def test_get_sync_age_description_never_synced(self, temp_storage):
        """Test sync age description when never synced."""
        assert temp_storage.get_sync_age_description() is None

    def test_get_sync_age_description_recent(self, temp_storage):
        """Test sync age description for recent sync."""
        temp_storage.record_sync_time()
        age_desc = temp_storage.get_sync_age_description()
        assert age_desc is not None
        assert "minute" in age_desc or "0" in age_desc
