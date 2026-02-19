"""DuckDB service for local query execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    import pyarrow as pa

    from dalcedo.services.dbt import DBTMetadata


@dataclass
class QueryResult:
    """Result of a DuckDB query."""

    columns: list[str]
    rows: list[tuple]
    row_count: int

    def summary(self) -> str:
        """Generate a brief summary of the results."""
        if self.row_count == 0:
            return "No rows returned"
        elif self.row_count == 1:
            return f"1 row returned with columns: {', '.join(self.columns)}"
        else:
            return f"{self.row_count} rows returned with columns: {', '.join(self.columns)}"


class DuckDBService:
    """Handles DuckDB connection and queries."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._connection: duckdb.DuckDBPyConnection | None = None

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = duckdb.connect(str(self.db_path))
        return self._connection

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def execute(self, sql: str) -> QueryResult:
        """Execute a SQL query and return results."""
        result = self.connection.execute(sql)
        columns = [desc[0] for desc in result.description] if result.description else []
        rows = result.fetchall()
        return QueryResult(columns=columns, rows=rows, row_count=len(rows))

    def execute_arrow(self, sql: str) -> "pa.Table":
        """Execute a SQL query and return as Arrow table."""
        return self.connection.execute(sql).arrow()

    # Schema management methods
    def create_schema(self, schema_name: str) -> None:
        """Create a schema if it doesn't exist."""
        self.connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')

    def drop_schema(self, schema_name: str, cascade: bool = True) -> None:
        """Drop a schema."""
        cascade_sql = "CASCADE" if cascade else ""
        self.connection.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" {cascade_sql}')

    def get_schemas(self) -> list[str]:
        """List all user schemas (excluding system schemas)."""
        result = self.connection.execute("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
            ORDER BY schema_name
        """)
        return [row[0] for row in result.fetchall()]

    def insert_arrow(
        self, table_name: str, arrow_table: "pa.Table", schema_name: str = "main"
    ) -> None:
        """Insert Arrow table data into a DuckDB table."""
        qualified_name = f'"{schema_name}"."{table_name}"'
        # Register Arrow table as a temporary view and insert from it
        self.connection.register("_temp_arrow_table", arrow_table)
        try:
            self.connection.execute(f"INSERT INTO {qualified_name} SELECT * FROM _temp_arrow_table")
        finally:
            self.connection.unregister("_temp_arrow_table")

    def create_table_from_arrow(
        self, table_name: str, arrow_table: "pa.Table", schema_name: str = "main"
    ) -> None:
        """Create a table directly from an Arrow table (lets DuckDB infer types)."""
        qualified_name = f'"{schema_name}"."{table_name}"'
        self.connection.execute(f"DROP TABLE IF EXISTS {qualified_name}")
        self.connection.register("_temp_arrow_table", arrow_table)
        try:
            self.connection.execute(
                f"CREATE TABLE {qualified_name} AS SELECT * FROM _temp_arrow_table"
            )
        finally:
            self.connection.unregister("_temp_arrow_table")

    def get_tables(self, schema_name: str | None = None) -> list[str]:
        """List all user tables in the database, optionally filtered by schema.

        Args:
            schema_name: If provided, only return tables from this schema.
                        If None, return tables from all schemas (unqualified names).
        """
        if schema_name:
            result = self.connection.execute(f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = '{schema_name}'
                  AND table_name NOT LIKE '_dalcedo_%'
                ORDER BY table_name
            """)
            return [row[0] for row in result.fetchall()]
        else:
            # Return all tables from non-system schemas
            result = self.connection.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                  AND table_name NOT LIKE '_dalcedo_%'
                ORDER BY table_name
            """)
            return [row[0] for row in result.fetchall()]

    def get_tables_by_schema(self) -> dict[str, list[str]]:
        """Get tables organized by schema.

        Returns:
            Dict mapping schema names to lists of table names.
        """
        result = self.connection.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
              AND table_name NOT LIKE '_dalcedo_%'
            ORDER BY table_schema, table_name
        """)

        tables_by_schema: dict[str, list[str]] = {}
        for schema, table in result.fetchall():
            if schema not in tables_by_schema:
                tables_by_schema[schema] = []
            tables_by_schema[schema].append(table)

        return tables_by_schema

    def get_schema_description(self) -> str:
        """Get a human-readable schema description for LLM context, including DBT metadata."""
        tables_by_schema = self.get_tables_by_schema()
        if not tables_by_schema:
            return "No tables synced yet. Run /sync to sync your data."

        # Load DBT metadata if available (from all schemas)
        table_descriptions: dict[str, dict[str, str]] = {}  # schema -> table -> desc
        column_descriptions: dict[
            str, dict[str, dict[str, str]]
        ] = {}  # schema -> table -> col -> desc

        # Check for metadata tables in each schema
        for schema in tables_by_schema:
            dbt_tables_name = f'"{schema}"._dalcedo_dbt_tables'
            dbt_columns_name = f'"{schema}"._dalcedo_dbt_columns'

            if self.table_exists("_dalcedo_dbt_tables", schema):
                result = self.connection.execute(
                    f"SELECT table_name, description FROM {dbt_tables_name}"
                )
                if schema not in table_descriptions:
                    table_descriptions[schema] = {}
                for row in result.fetchall():
                    if row[1]:  # Only if description exists
                        table_descriptions[schema][row[0]] = row[1]

            if self.table_exists("_dalcedo_dbt_columns", schema):
                result = self.connection.execute(
                    f"SELECT table_name, column_name, description FROM {dbt_columns_name}"
                )
                if schema not in column_descriptions:
                    column_descriptions[schema] = {}
                for row in result.fetchall():
                    if row[2]:  # Only if description exists
                        if row[0] not in column_descriptions[schema]:
                            column_descriptions[schema][row[0]] = {}
                        column_descriptions[schema][row[0]][row[1]] = row[2]

        # Also check main schema for legacy metadata
        if self.table_exists("_dalcedo_dbt_tables"):
            result = self.connection.execute(
                "SELECT table_name, description FROM _dalcedo_dbt_tables"
            )
            if "main" not in table_descriptions:
                table_descriptions["main"] = {}
            for row in result.fetchall():
                if row[1]:
                    table_descriptions["main"][row[0]] = row[1]

        if self.table_exists("_dalcedo_dbt_columns"):
            result = self.connection.execute(
                "SELECT table_name, column_name, description FROM _dalcedo_dbt_columns"
            )
            if "main" not in column_descriptions:
                column_descriptions["main"] = {}
            for row in result.fetchall():
                if row[2]:
                    if row[0] not in column_descriptions["main"]:
                        column_descriptions["main"][row[0]] = {}
                    column_descriptions["main"][row[0]][row[1]] = row[2]

        schema_parts = []
        for schema, tables in sorted(tables_by_schema.items()):
            schema_parts.append(f"=== Schema: {schema} ===")

            for table in tables:
                qualified_name = f"{schema}.{table}"
                # Get table description from DBT if available
                schema_table_descs = table_descriptions.get(schema, {})
                table_desc = schema_table_descs.get(table, "")
                table_header = f"Table: {qualified_name}"
                if table_desc:
                    table_header += f"\n  Description: {table_desc}"

                result = self.connection.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = '{schema}' AND table_name = '{table}'
                    ORDER BY ordinal_position
                """)
                columns = result.fetchall()

                col_descs = []
                schema_col_descs = column_descriptions.get(schema, {})
                table_col_descs = schema_col_descs.get(table, {})
                for col_name, data_type, nullable in columns:
                    null_str = "" if nullable == "YES" else " NOT NULL"
                    col_desc = table_col_descs.get(col_name, "")
                    if col_desc:
                        col_descs.append(f"  - {col_name}: {data_type}{null_str} -- {col_desc}")
                    else:
                        col_descs.append(f"  - {col_name}: {data_type}{null_str}")

                schema_parts.append(table_header + "\n" + "\n".join(col_descs))

        return "\n\n".join(schema_parts)

    def create_table_from_schema(
        self,
        table_name: str,
        columns: list[dict],
        drop_existing: bool = True,
        schema_name: str = "main",
    ) -> None:
        """Create a table from a schema definition."""
        qualified_name = f'"{schema_name}"."{table_name}"'
        if drop_existing:
            self.connection.execute(f"DROP TABLE IF EXISTS {qualified_name}")

        # Map BigQuery types to DuckDB types
        type_mapping = {
            "STRING": "VARCHAR",
            "INT64": "BIGINT",
            "INTEGER": "BIGINT",
            "FLOAT64": "DOUBLE",
            "FLOAT": "DOUBLE",
            "NUMERIC": "DECIMAL(38, 9)",
            "BIGNUMERIC": "DECIMAL(76, 38)",
            "BOOLEAN": "BOOLEAN",
            "BOOL": "BOOLEAN",
            "TIMESTAMP": "TIMESTAMP",
            "DATE": "DATE",
            "TIME": "TIME",
            "DATETIME": "TIMESTAMP",
            "BYTES": "BLOB",
            "JSON": "JSON",
            "GEOGRAPHY": "VARCHAR",  # Store as WKT
        }

        col_defs = []
        for col in columns:
            duckdb_type = type_mapping.get(col["type"].upper(), "VARCHAR")
            nullable = "" if col.get("mode") == "REQUIRED" else ""
            col_defs.append(f'"{col["name"]}" {duckdb_type}{nullable}')

        create_sql = f"CREATE TABLE {qualified_name} ({', '.join(col_defs)})"
        self.connection.execute(create_sql)

    def store_schema_metadata(self, schema_name: str = "main") -> None:
        """Store schema metadata for reference."""
        qualified_name = f'"{schema_name}"._dalcedo_schema_metadata'
        self.connection.execute(f"""
            CREATE OR REPLACE TABLE {qualified_name} AS
            SELECT
                table_name,
                column_name,
                data_type,
                is_nullable = 'YES' as is_nullable
            FROM information_schema.columns
            WHERE table_schema = '{schema_name}'
              AND table_name NOT LIKE '_dalcedo_%'
            ORDER BY table_name, ordinal_position
        """)

    def store_dbt_metadata(self, metadata: "DBTMetadata", schema_name: str = "main") -> None:
        """Store DBT metadata (table and column descriptions)."""
        tables_qualified = f'"{schema_name}"._dalcedo_dbt_tables'
        columns_qualified = f'"{schema_name}"._dalcedo_dbt_columns'

        # Create tables for DBT metadata
        self.connection.execute(f"""
            CREATE OR REPLACE TABLE {tables_qualified} (
                table_name VARCHAR PRIMARY KEY,
                description VARCHAR
            )
        """)
        self.connection.execute(f"""
            CREATE OR REPLACE TABLE {columns_qualified} (
                table_name VARCHAR,
                column_name VARCHAR,
                description VARCHAR,
                PRIMARY KEY (table_name, column_name)
            )
        """)

        # Insert table metadata
        for table_name, table_meta in metadata.tables.items():
            self.connection.execute(
                f"INSERT OR REPLACE INTO {tables_qualified} VALUES (?, ?)",
                [table_name, table_meta.description],
            )
            # Insert column metadata
            for col_name, col_meta in table_meta.columns.items():
                self.connection.execute(
                    f"INSERT OR REPLACE INTO {columns_qualified} VALUES (?, ?, ?)",
                    [table_name, col_name, col_meta.description],
                )

    def table_exists(self, table_name: str, schema_name: str = "main") -> bool:
        """Check if a table exists."""
        result = self.connection.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
        """)
        return result.fetchone()[0] > 0
