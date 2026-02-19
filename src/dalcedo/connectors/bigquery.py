"""BigQuery connector for data infrastructure metadata."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from google.cloud import bigquery
from google.oauth2 import service_account

from dalcedo.connectors.base import BaseConnector, ConnectorField, SyncUpdate
from dalcedo.connectors.registry import register_connector

if TYPE_CHECKING:
    import pyarrow as pa

    from dalcedo.services.duckdb import DuckDBService


@register_connector
class BigQueryConnector(BaseConnector):
    """BigQuery metadata connector.

    Syncs table metadata and data from a BigQuery dataset to local DuckDB
    for natural language querying.

    Connection config (shared): project_id, location, credentials_path
    Source config (per plugin): dataset_id, dbt_schema_paths
    """

    name = "bigquery"
    display_name = "BigQuery"
    description = "Query BigQuery dataset metadata and tables"

    def __init__(self, connection_config: dict[str, Any], source_config: dict[str, Any]):
        super().__init__(connection_config, source_config)
        self._client: bigquery.Client | None = None

    @classmethod
    def get_connection_fields(cls) -> list[ConnectorField]:
        """Connection-level fields (shared across plugins)."""
        return [
            ConnectorField(
                name="project_id",
                label="BigQuery Project ID",
                placeholder="my-gcp-project",
                required=True,
            ),
            ConnectorField(
                name="location",
                label="BigQuery Region",
                placeholder="US",
                default="US",
                required=False,
            ),
            ConnectorField(
                name="credentials_path",
                label="Service Account JSON",
                field_type="file",
                placeholder="/path/to/service-account.json",
                required=False,
            ),
        ]

    @classmethod
    def get_source_fields(cls) -> list[ConnectorField]:
        """Source-level fields (unique per plugin)."""
        return [
            ConnectorField(
                name="dataset_id",
                label="BigQuery Dataset",
                placeholder="my_dataset",
                required=True,
            ),
            ConnectorField(
                name="dbt_schema_paths",
                label="DBT Schema Files",
                placeholder="/path/to/schema.yml, /path/to/other.yml",
                required=False,
            ),
        ]

    @property
    def project_id(self) -> str:
        return self.connection_config["project_id"]

    @property
    def location(self) -> str:
        return self.connection_config.get("location", "US")

    @property
    def credentials_path(self) -> Path | None:
        path = self.connection_config.get("credentials_path")
        return Path(path) if path else None

    @property
    def dataset_id(self) -> str:
        return self.source_config["dataset_id"]

    @property
    def dbt_schema_paths(self) -> list[Path] | None:
        paths_str = self.source_config.get("dbt_schema_paths", "")
        if not paths_str:
            return None
        return [Path(p.strip()) for p in paths_str.split(",") if p.strip()]

    @property
    def client(self) -> bigquery.Client:
        """Get or create BigQuery client."""
        if self._client is None:
            if self.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    str(self.credentials_path)
                )
                self._client = bigquery.Client(
                    project=self.project_id,
                    credentials=credentials,
                    location=self.location,
                )
            else:
                self._client = bigquery.Client(
                    project=self.project_id,
                    location=self.location,
                )
        return self._client

    @property
    def dataset_ref(self) -> str:
        """Full dataset reference."""
        return f"{self.project_id}.{self.dataset_id}"

    async def test_connection(self) -> bool:
        """Test BigQuery connection by getting dataset info."""
        try:
            dataset = self.client.get_dataset(self.dataset_ref)
            return dataset is not None
        except Exception:
            raise

    def list_tables(self) -> list[bigquery.Table]:
        """List all tables in the dataset."""
        tables = self.client.list_tables(self.dataset_ref)
        return list(tables)

    def get_full_table(self, table_id: str) -> "pa.Table":
        """Get all rows from a table as Arrow table."""
        query = f"SELECT * FROM `{self.dataset_ref}.{table_id}`"
        query_job = self.client.query(query)
        return query_job.to_arrow()

    def sync(self, db: "DuckDBService", schema_name: str = "main") -> Iterator[SyncUpdate]:
        """Sync BigQuery tables to local DuckDB."""
        yield SyncUpdate("Connecting to BigQuery...", 0.0, schema_name=schema_name)

        # Create the schema in DuckDB
        db.create_schema(schema_name)

        try:
            tables = self.list_tables()
        except Exception as e:
            yield SyncUpdate(
                f"Failed to list tables: {e}", 0.0, schema_name=schema_name, is_error=True
            )
            return

        if not tables:
            yield SyncUpdate("No tables found in dataset", 1.0, schema_name=schema_name)
            return

        total = len(tables)
        yield SyncUpdate(f"Found {total} tables to sync", 0.05, schema_name=schema_name)

        for i, table in enumerate(tables):
            table_id = table.table_id
            base_progress = 0.1 + (0.85 * i / total)

            yield SyncUpdate(
                f"Downloading {table_id}...", base_progress, table_id, schema_name=schema_name
            )

            try:
                arrow_table = self.get_full_table(table_id)
                db.create_table_from_arrow(table_id, arrow_table, schema_name=schema_name)

                yield SyncUpdate(
                    f"Completed {table_id} ({arrow_table.num_rows} rows)",
                    base_progress + (0.85 / total),
                    table_id,
                    schema_name=schema_name,
                )

            except Exception as e:
                yield SyncUpdate(
                    f"Error syncing {table_id}: {e}",
                    base_progress + (0.85 / total),
                    table_id,
                    schema_name=schema_name,
                    is_error=True,
                )

        # Store schema metadata
        yield SyncUpdate("Storing schema metadata...", 0.95, schema_name=schema_name)
        db.store_schema_metadata(schema_name=schema_name)

        # Load DBT metadata if configured
        if self.dbt_schema_paths:
            yield SyncUpdate("Loading DBT metadata...", 0.97, schema_name=schema_name)
            try:
                from dalcedo.services.dbt import DBTService

                dbt_service = DBTService(self.dbt_schema_paths)
                dbt_metadata = dbt_service.load_metadata()
                db.store_dbt_metadata(dbt_metadata, schema_name=schema_name)
                yield SyncUpdate(
                    f"Loaded descriptions for {len(dbt_metadata.tables)} tables",
                    0.99,
                    schema_name=schema_name,
                )
            except Exception as e:
                yield SyncUpdate(
                    f"Warning: Could not load DBT metadata: {e}",
                    0.99,
                    schema_name=schema_name,
                    is_error=True,
                )

        yield SyncUpdate("Sync complete!", 1.0, schema_name=schema_name)

    def get_schema_context(self, db: "DuckDBService") -> str:
        """Return schema description for LLM context."""
        return db.get_schema_description()
