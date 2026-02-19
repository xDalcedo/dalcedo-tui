"""DBT metadata parsing service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ColumnMeta:
    """Column metadata from DBT schema."""

    name: str
    description: str = ""
    data_type: str | None = None


@dataclass
class TableMeta:
    """Table metadata from DBT schema."""

    name: str
    description: str = ""
    columns: dict[str, ColumnMeta] = field(default_factory=dict)


@dataclass
class DBTMetadata:
    """Parsed DBT metadata."""

    tables: dict[str, TableMeta] = field(default_factory=dict)

    def get_table_description(self, table_name: str) -> str | None:
        """Get description for a table."""
        table = self.tables.get(table_name)
        return table.description if table else None

    def get_column_description(self, table_name: str, column_name: str) -> str | None:
        """Get description for a column."""
        table = self.tables.get(table_name)
        if table and column_name in table.columns:
            return table.columns[column_name].description
        return None


class DBTService:
    """Parses DBT schema.yml files for table/column metadata."""

    def __init__(self, schema_paths: list[Path]):
        """Initialize with explicit paths to schema.yml files."""
        self.schema_paths = schema_paths

    def load_metadata(self) -> DBTMetadata:
        """Load metadata from the configured schema.yml files."""
        metadata = DBTMetadata()

        for schema_file in self.schema_paths:
            if schema_file.exists():
                try:
                    self._parse_schema_file(schema_file, metadata)
                except Exception:
                    # Skip files that can't be parsed
                    continue

        return metadata

    def _parse_schema_file(self, file_path: Path, metadata: DBTMetadata) -> None:
        """Parse a single schema.yml file."""
        with open(file_path) as f:
            content = yaml.safe_load(f)

        if not content:
            return

        # Handle models section
        models = content.get("models", [])
        for model in models:
            self._parse_model(model, metadata)

        # Handle sources section
        sources = content.get("sources", [])
        for source in sources:
            source_name = source.get("name", "")
            for table in source.get("tables", []):
                self._parse_model(table, metadata, source_prefix=source_name)

    def _parse_model(
        self, model: dict, metadata: DBTMetadata, source_prefix: str | None = None
    ) -> None:
        """Parse a model/table definition."""
        name = model.get("name", "")
        if not name:
            return

        table_meta = TableMeta(
            name=name,
            description=model.get("description", ""),
        )

        # Parse columns
        for column in model.get("columns", []):
            col_name = column.get("name", "")
            if col_name:
                table_meta.columns[col_name] = ColumnMeta(
                    name=col_name,
                    description=column.get("description", ""),
                    data_type=column.get("data_type"),
                )

        # Store by table name (without source prefix for easier matching)
        metadata.tables[name] = table_meta
