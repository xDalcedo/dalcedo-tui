"""BigQuery service for data access."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from google.cloud import bigquery
from google.oauth2 import service_account

if TYPE_CHECKING:
    import pyarrow as pa


class BigQueryService:
    """Handles BigQuery connection and data retrieval."""

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        credentials_path: Path | None = None,
        location: str = "US",
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.location = location
        self._client: bigquery.Client | None = None
        self._credentials_path = credentials_path

    @property
    def client(self) -> bigquery.Client:
        """Get or create BigQuery client."""
        if self._client is None:
            if self._credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    str(self._credentials_path)
                )
                self._client = bigquery.Client(
                    project=self.project_id,
                    credentials=credentials,
                    location=self.location,
                )
            else:
                # Use Application Default Credentials
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
        """Test BigQuery connection by listing tables."""
        try:
            dataset = self.client.get_dataset(self.dataset_ref)
            return dataset is not None
        except Exception:
            raise

    def list_tables(self) -> list[bigquery.Table]:
        """List all tables in the dataset."""
        tables = self.client.list_tables(self.dataset_ref)
        return list(tables)

    def get_table_schema(self, table_id: str) -> list[dict]:
        """Get schema for a specific table."""
        table_ref = f"{self.dataset_ref}.{table_id}"
        table = self.client.get_table(table_ref)

        schema = []
        for field in table.schema:
            schema.append(
                {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description,
                }
            )
        return schema

    def get_table_info(self, table_id: str) -> dict:
        """Get table metadata."""
        table_ref = f"{self.dataset_ref}.{table_id}"
        table = self.client.get_table(table_ref)

        return {
            "table_id": table.table_id,
            "num_rows": table.num_rows,
            "num_bytes": table.num_bytes,
            "created": table.created,
            "modified": table.modified,
            "description": table.description,
        }

    def query_to_arrow(self, query: str) -> "pa.Table":
        """Execute query and return results as Arrow table."""
        query_job = self.client.query(query)
        return query_job.to_arrow()

    def get_sample_data(self, table_id: str, limit: int = 1000) -> "pa.Table":
        """Get sample rows from a table as Arrow table."""
        query = f"""
            SELECT *
            FROM `{self.dataset_ref}.{table_id}`
            LIMIT {limit}
        """
        return self.query_to_arrow(query)

    def get_full_table(self, table_id: str) -> "pa.Table":
        """Get all rows from a table as Arrow table."""
        query = f"SELECT * FROM `{self.dataset_ref}.{table_id}`"
        return self.query_to_arrow(query)

    def estimate_table_size_mb(self, table_id: str) -> float:
        """Estimate table size in MB."""
        info = self.get_table_info(table_id)
        return (info["num_bytes"] or 0) / (1024 * 1024)
