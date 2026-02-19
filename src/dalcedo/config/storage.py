"""XDG-compliant configuration storage for Dalcedo."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import tomli_w

from dalcedo.config.settings import AppConfig, Connection, Credentials, Plugin

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]

from xdg_base_dirs import xdg_cache_home, xdg_config_home, xdg_data_home

APP_NAME = "dalcedo"


class ConfigStorage:
    """XDG-compliant configuration storage."""

    @property
    def config_dir(self) -> Path:
        """Config directory for non-sensitive settings."""
        return xdg_config_home() / APP_NAME

    @property
    def data_dir(self) -> Path:
        """Data directory for sensitive credentials."""
        return xdg_data_home() / APP_NAME

    @property
    def cache_dir(self) -> Path:
        """Cache directory for DuckDB database."""
        return xdg_cache_home() / APP_NAME

    @property
    def config_file(self) -> Path:
        """Path to config.toml."""
        return self.config_dir / "config.toml"

    @property
    def credentials_file(self) -> Path:
        """Path to credentials.json."""
        return self.data_dir / "credentials.json"

    @property
    def duckdb_path(self) -> Path:
        """Path to DuckDB database file."""
        return self.cache_dir / "local.duckdb"

    def _ensure_dirs(self) -> None:
        """Create necessary directories with proper permissions."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Secure the data directory
        os.chmod(self.data_dir, 0o700)

    def load_config(self) -> AppConfig:
        """Load configuration from disk."""
        config = AppConfig()

        # Load config.toml
        if self.config_file.exists():
            with open(self.config_file, "rb") as f:
                data = tomllib.load(f)

            llm_data = data.get("llm", {})
            connections_data = data.get("connections", [])
            plugins_data = data.get("plugins", [])

            # Handle backwards compatibility with old formats
            connections_data, plugins_data = self._migrate_old_formats(
                data, connections_data, plugins_data
            )

            # Parse connections
            connections = []
            for c in connections_data:
                connections.append(
                    Connection(
                        id=c.get("id", ""),
                        name=c.get("name", ""),
                        type=c.get("type", "bigquery"),
                        config=c.get("config", {}),
                    )
                )

            # Parse plugins
            plugins = []
            for p in plugins_data:
                plugins.append(
                    Plugin(
                        id=p.get("id", ""),
                        name=p.get("name", ""),
                        connection_id=p.get("connection_id", ""),
                        source_config=p.get("source", {}),
                        enabled=p.get("enabled", True),
                    )
                )

            # Load token limits
            limits_data = data.get("limits", {})

            config = AppConfig(
                connections=connections,
                plugins=plugins,
                llm_provider=llm_data.get("provider", "anthropic"),
                llm_model=llm_data.get("model"),
                max_context_messages=llm_data.get("max_context_messages", 20),
                custom_context=data.get("context", {}).get("custom"),
                daily_token_limit=limits_data.get("daily_tokens"),
                weekly_token_limit=limits_data.get("weekly_tokens"),
            )

        # Load credentials.json
        if self.credentials_file.exists():
            with open(self.credentials_file) as f:
                creds_data = json.load(f)

            # Support old field names for backwards compatibility
            api_key = creds_data.get("llm_api_key") or creds_data.get("anthropic_api_key", "")

            # Load per-connection sensitive credentials
            connection_creds = creds_data.get("connection_credentials", {})
            for connection in config.connections:
                if connection.id in connection_creds:
                    connection.config.update(connection_creds[connection.id])

            # Legacy: migrate old bigquery_credentials_path or plugin_credentials
            bq_creds_path = creds_data.get("bigquery_credentials_path")
            if bq_creds_path and config.connections:
                first_conn = config.connections[0]
                if "credentials_path" not in first_conn.config:
                    first_conn.config["credentials_path"] = bq_creds_path

            # Legacy: migrate old plugin_credentials to connection_credentials
            plugin_creds = creds_data.get("plugin_credentials", {})
            if plugin_creds and not connection_creds:
                # Old format had credentials per plugin, move to first connection
                if config.connections:
                    for cred_values in plugin_creds.values():
                        config.connections[0].config.update(cred_values)
                        break  # Only need first one

            config = AppConfig(
                connections=config.connections,
                plugins=config.plugins,
                credentials=Credentials(llm_api_key=api_key),
                llm_provider=config.llm_provider,
                llm_model=config.llm_model,
                max_context_messages=config.max_context_messages,
                custom_context=config.custom_context,
                daily_token_limit=config.daily_token_limit,
                weekly_token_limit=config.weekly_token_limit,
            )

        return config

    def _migrate_old_formats(
        self, data: dict, connections_data: list, plugins_data: list
    ) -> tuple[list, list]:
        """Migrate old config formats to new connection+plugin format."""
        if connections_data and plugins_data:
            return connections_data, plugins_data

        # Handle old plugins format (plugins had connector_type + connector_config)
        old_plugins = data.get("plugins", [])
        if old_plugins and not connections_data:
            # Check if this is old format (has connector_type instead of connection_id)
            first_plugin = old_plugins[0] if old_plugins else {}
            if "connector_type" in first_plugin or "config" in first_plugin:
                # Old format - migrate to new format
                # Create a connection from the first plugin's connector info
                connector_type = first_plugin.get("connector_type", "bigquery")
                connector_config = first_plugin.get("config", {})

                # Extract connection-level config (project_id, location, credentials_path)
                connection_config = {}
                source_config = {}
                connection_fields = ["project_id", "location", "credentials_path"]

                for key, value in connector_config.items():
                    if key in connection_fields:
                        connection_config[key] = value
                    else:
                        source_config[key] = value

                connection_id = "migrated_conn"
                connections_data = [
                    {
                        "id": connection_id,
                        "name": f"Migrated {connector_type.title()}",
                        "type": connector_type,
                        "config": connection_config,
                    }
                ]

                # Convert old plugins to new format
                plugins_data = []
                for p in old_plugins:
                    old_config = p.get("config", {})
                    new_source = {
                        k: v for k, v in old_config.items() if k not in connection_fields
                    }
                    plugins_data.append(
                        {
                            "id": p.get("id", ""),
                            "name": p.get("name", ""),
                            "connection_id": connection_id,
                            "source": new_source,
                            "enabled": p.get("enabled", True),
                        }
                    )

                return connections_data, plugins_data

        # Handle old connector format
        if "connector" in data and not plugins_data:
            old_connector = data["connector"]
            connector_type = old_connector.get("type", "bigquery")
            connector_config = old_connector.get("config", {})

            connection_config = {}
            source_config = {}
            connection_fields = ["project_id", "location", "credentials_path"]

            for key, value in connector_config.items():
                if key in connection_fields:
                    connection_config[key] = value
                else:
                    source_config[key] = value

            connection_id = "migrated_conn"
            connections_data = [
                {
                    "id": connection_id,
                    "name": f"Migrated {connector_type.title()}",
                    "type": connector_type,
                    "config": connection_config,
                }
            ]
            plugins_data = [
                {
                    "id": "migrated",
                    "name": "default",
                    "connection_id": connection_id,
                    "source": source_config,
                    "enabled": True,
                }
            ]
            return connections_data, plugins_data

        # Handle very old bigquery-specific format
        if "bigquery" in data and not plugins_data:
            old_bq = data["bigquery"]
            dbt_paths = data.get("dbt", {}).get("schema_paths", [])

            connection_config = {
                "project_id": old_bq.get("project_id", ""),
                "location": old_bq.get("location", "US"),
            }
            source_config = {
                "dataset_id": old_bq.get("dataset_id", ""),
            }
            if dbt_paths:
                source_config["dbt_schema_paths"] = ", ".join(dbt_paths)

            connection_id = "migrated_conn"
            connections_data = [
                {
                    "id": connection_id,
                    "name": "Migrated BigQuery",
                    "type": "bigquery",
                    "config": connection_config,
                }
            ]
            plugins_data = [
                {
                    "id": "migrated",
                    "name": "default",
                    "connection_id": connection_id,
                    "source": source_config,
                    "enabled": True,
                }
            ]
            return connections_data, plugins_data

        return connections_data, plugins_data

    def save_config(self, config: AppConfig) -> None:
        """Save configuration to disk with proper permissions."""
        self._ensure_dirs()

        # Build connections array for TOML
        connections_data = []
        for conn in config.connections:
            connections_data.append(
                {
                    "id": conn.id,
                    "name": conn.name,
                    "type": conn.type,
                    "config": self._filter_sensitive_config(conn.config),
                }
            )

        # Build plugins array for TOML
        plugins_data = []
        for plugin in config.plugins:
            plugins_data.append(
                {
                    "id": plugin.id,
                    "name": plugin.name,
                    "connection_id": plugin.connection_id,
                    "enabled": plugin.enabled,
                    "source": plugin.source_config,
                }
            )

        # Save config.toml (non-sensitive)
        config_data: dict = {
            "llm": {
                "provider": config.llm_provider,
                "max_context_messages": config.max_context_messages,
            },
            "connections": connections_data,
            "plugins": plugins_data,
        }

        if config.llm_model:
            config_data["llm"]["model"] = config.llm_model

        if config.custom_context:
            config_data["context"] = {"custom": config.custom_context}

        # Save token limits
        if config.daily_token_limit or config.weekly_token_limit:
            limits: dict = {}
            if config.daily_token_limit:
                limits["daily_tokens"] = config.daily_token_limit
            if config.weekly_token_limit:
                limits["weekly_tokens"] = config.weekly_token_limit
            config_data["limits"] = limits

        with open(self.config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        # Save credentials.json (sensitive connection config + LLM key)
        if config.credentials:
            creds_data: dict = {
                "llm_api_key": config.credentials.llm_api_key,
            }

            # Store per-connection sensitive credentials
            connection_creds = {}
            sensitive_keys = ["credentials_path", "api_key", "token", "secret", "password"]
            for conn in config.connections:
                conn_sensitive = {}
                for key in sensitive_keys:
                    if key in conn.config:
                        conn_sensitive[key] = conn.config[key]
                if conn_sensitive:
                    connection_creds[conn.id] = conn_sensitive

            if connection_creds:
                creds_data["connection_credentials"] = connection_creds

            with open(self.credentials_file, "w") as f:
                json.dump(creds_data, f, indent=2)

            # Secure credentials file
            os.chmod(self.credentials_file, 0o600)

    def _filter_sensitive_config(self, config: dict) -> dict:
        """Remove sensitive fields from config for storage in config.toml."""
        sensitive_keys = ["credentials_path", "api_key", "token", "secret", "password"]
        return {k: v for k, v in config.items() if k not in sensitive_keys}

    def is_configured(self) -> bool:
        """Check if login has been completed."""
        return self.config_file.exists() and self.credentials_file.exists()

    def clear_cache(self) -> None:
        """Remove cached DuckDB database."""
        if self.duckdb_path.exists():
            self.duckdb_path.unlink()
        wal_path = self.cache_dir / "local.duckdb.wal"
        if wal_path.exists():
            wal_path.unlink()

    @property
    def _metadata_file(self) -> Path:
        """Path to cache metadata file."""
        return self.cache_dir / "metadata.json"

    def _load_metadata(self) -> dict:
        """Load cache metadata."""
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata: dict) -> None:
        """Save cache metadata."""
        self._ensure_dirs()
        with open(self._metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def record_sync_time(self) -> None:
        """Record the current time as the last sync time (global)."""
        metadata = self._load_metadata()
        metadata["last_sync"] = datetime.now().isoformat()
        self._save_metadata(metadata)

    def record_plugin_sync_time(self, plugin_id: str) -> None:
        """Record the current time as the last sync time for a specific plugin."""
        metadata = self._load_metadata()
        if "plugin_syncs" not in metadata:
            metadata["plugin_syncs"] = {}
        metadata["plugin_syncs"][plugin_id] = datetime.now().isoformat()
        # Also update global sync time
        metadata["last_sync"] = datetime.now().isoformat()
        self._save_metadata(metadata)

    def get_plugin_sync_time(self, plugin_id: str) -> datetime | None:
        """Get the last sync time for a specific plugin."""
        metadata = self._load_metadata()
        plugin_syncs = metadata.get("plugin_syncs", {})
        sync_time = plugin_syncs.get(plugin_id)
        if sync_time:
            return datetime.fromisoformat(sync_time)
        return None

    def get_plugin_sync_age_description(self, plugin_id: str) -> str | None:
        """Get a human-readable description of how old a plugin's sync is."""
        last_sync = self.get_plugin_sync_time(plugin_id)
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

    def record_sync_suggestion_time(self) -> None:
        """Record the current time as the last sync suggestion time."""
        metadata = self._load_metadata()
        metadata["last_sync_suggestion"] = datetime.now().isoformat()
        self._save_metadata(metadata)

    def get_last_sync_time(self) -> datetime | None:
        """Get the last sync time, or None if never synced."""
        metadata = self._load_metadata()
        last_sync = metadata.get("last_sync")
        if last_sync:
            return datetime.fromisoformat(last_sync)
        return None

    def should_suggest_sync(
        self,
        sync_threshold_hours: int = 24,
        suggestion_threshold_hours: int = 12,
    ) -> bool:
        """Check if we should suggest a sync to the user.

        Returns True if:
        - Last sync was more than sync_threshold_hours ago (or never synced)
        - AND last suggestion was more than suggestion_threshold_hours ago (or never suggested)
        """
        metadata = self._load_metadata()
        now = datetime.now()

        # Check last sync time
        last_sync_str = metadata.get("last_sync")
        if last_sync_str:
            last_sync = datetime.fromisoformat(last_sync_str)
            if now - last_sync < timedelta(hours=sync_threshold_hours):
                return False  # Synced recently, no need to suggest
        # If never synced, we should suggest (but check suggestion threshold)

        # Check last suggestion time
        last_suggestion_str = metadata.get("last_sync_suggestion")
        if last_suggestion_str:
            last_suggestion = datetime.fromisoformat(last_suggestion_str)
            if now - last_suggestion < timedelta(hours=suggestion_threshold_hours):
                return False  # Suggested recently, don't nag

        return True

    def get_sync_age_description(self) -> str | None:
        """Get a human-readable description of how old the sync is.

        Returns None if never synced.
        """
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

    # Token usage tracking

    def record_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for today."""
        metadata = self._load_metadata()
        today = datetime.now().strftime("%Y-%m-%d")

        if "token_usage" not in metadata:
            metadata["token_usage"] = {}

        if today not in metadata["token_usage"]:
            metadata["token_usage"][today] = {"input": 0, "output": 0}

        metadata["token_usage"][today]["input"] += input_tokens
        metadata["token_usage"][today]["output"] += output_tokens

        # Clean up old entries (keep last 14 days)
        cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        metadata["token_usage"] = {
            k: v for k, v in metadata["token_usage"].items() if k >= cutoff
        }

        self._save_metadata(metadata)

    def get_daily_token_usage(self) -> tuple[int, int]:
        """Get today's token usage (input, output)."""
        metadata = self._load_metadata()
        today = datetime.now().strftime("%Y-%m-%d")
        usage = metadata.get("token_usage", {}).get(today, {"input": 0, "output": 0})
        return usage["input"], usage["output"]

    def get_weekly_token_usage(self) -> tuple[int, int]:
        """Get this week's token usage (input, output). Week starts on Monday."""
        metadata = self._load_metadata()
        token_usage = metadata.get("token_usage", {})

        # Get start of current week (Monday)
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_start_str = week_start.strftime("%Y-%m-%d")

        total_input = 0
        total_output = 0
        for date_str, usage in token_usage.items():
            if date_str >= week_start_str:
                total_input += usage.get("input", 0)
                total_output += usage.get("output", 0)

        return total_input, total_output

    def get_token_usage_summary(self) -> dict:
        """Get a summary of token usage for display."""
        daily_in, daily_out = self.get_daily_token_usage()
        weekly_in, weekly_out = self.get_weekly_token_usage()

        return {
            "daily": {"input": daily_in, "output": daily_out, "total": daily_in + daily_out},
            "weekly": {"input": weekly_in, "output": weekly_out, "total": weekly_in + weekly_out},
        }
