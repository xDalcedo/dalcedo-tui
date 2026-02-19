"""Sync service for BigQuery to DuckDB synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from dalcedo.services.bigquery import BigQueryService
    from dalcedo.services.duckdb import DuckDBService


@dataclass
class SyncUpdate:
    """Progress update during sync."""

    message: str
    progress: float  # 0.0 to 1.0
    table: str | None = None
    is_error: bool = False


class SyncService:
    """Handles BigQuery to DuckDB synchronization."""

    def __init__(
        self,
        bigquery: "BigQueryService",
        duckdb: "DuckDBService",
        dbt_schema_paths: list[Path] | None = None,
    ):
        self.bq = bigquery
        self.db = duckdb
        self.dbt_schema_paths = dbt_schema_paths

    def sync(self) -> Iterator[SyncUpdate]:
        """Execute full sync and yield progress updates."""
        yield SyncUpdate("Connecting to BigQuery...", 0.0)

        try:
            tables = self.bq.list_tables()
        except Exception as e:
            yield SyncUpdate(f"Failed to list tables: {e}", 0.0, is_error=True)
            return

        if not tables:
            yield SyncUpdate("No tables found in dataset", 1.0)
            return

        total = len(tables)
        yield SyncUpdate(f"Found {total} tables to sync", 0.05)

        for i, table in enumerate(tables):
            table_id = table.table_id
            base_progress = 0.1 + (0.85 * i / total)

            yield SyncUpdate(f"Downloading {table_id}...", base_progress, table_id)

            try:
                # Get full table data as Arrow and create DuckDB table
                arrow_table = self.bq.get_full_table(table_id)
                self.db.create_table_from_arrow(table_id, arrow_table)

                yield SyncUpdate(
                    f"Completed {table_id} ({arrow_table.num_rows} rows)",
                    base_progress + (0.85 / total),
                    table_id,
                )

            except Exception as e:
                yield SyncUpdate(
                    f"Error syncing {table_id}: {e}",
                    base_progress + (0.85 / total),
                    table_id,
                    is_error=True,
                )

        # Store schema metadata
        yield SyncUpdate("Storing schema metadata...", 0.95)
        self.db.store_schema_metadata()

        # Load and store DBT metadata if configured
        if self.dbt_schema_paths:
            yield SyncUpdate("Loading DBT metadata...", 0.97)
            try:
                from dalcedo.services.dbt import DBTService

                dbt_service = DBTService(self.dbt_schema_paths)
                dbt_metadata = dbt_service.load_metadata()
                self.db.store_dbt_metadata(dbt_metadata)
                yield SyncUpdate(f"Loaded descriptions for {len(dbt_metadata.tables)} tables", 0.99)
            except Exception as e:
                yield SyncUpdate(f"Warning: Could not load DBT metadata: {e}", 0.99, is_error=True)

        yield SyncUpdate("Sync complete!", 1.0)
