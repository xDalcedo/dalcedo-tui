"""Tests for DuckDB service."""

from __future__ import annotations

import pytest
import pyarrow as pa

from dalcedo.services.duckdb import DuckDBService, QueryResult


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_empty_result(self):
        """Test empty result summary."""
        result = QueryResult(columns=[], rows=[], row_count=0)
        assert "No rows" in result.summary()

    def test_single_row_result(self):
        """Test single row summary."""
        result = QueryResult(
            columns=["id", "name"],
            rows=[(1, "Alice")],
            row_count=1,
        )
        assert "1 row" in result.summary()
        assert "id" in result.summary()

    def test_multiple_rows_result(self):
        """Test multiple rows summary."""
        result = QueryResult(
            columns=["id"],
            rows=[(1,), (2,), (3,)],
            row_count=3,
        )
        assert "3 rows" in result.summary()


class TestDuckDBService:
    """Tests for DuckDBService."""

    def test_connection_created_lazily(self, temp_dir):
        """Test connection is created lazily."""
        db_path = temp_dir / "test.duckdb"
        service = DuckDBService(db_path=db_path)

        # Connection should not exist yet
        assert service._connection is None

        # Access connection property
        _ = service.connection

        # Now it should exist
        assert service._connection is not None
        assert db_path.exists()

        service.close()

    def test_execute_simple_query(self, temp_duckdb):
        """Test executing a simple query."""
        result = temp_duckdb.execute("SELECT 1 as num, 'hello' as greeting")
        assert result.columns == ["num", "greeting"]
        assert result.rows == [(1, "hello")]
        assert result.row_count == 1

    def test_execute_with_sample_tables(self, sample_tables):
        """Test executing query on sample tables."""
        result = sample_tables.execute("SELECT * FROM users ORDER BY id")
        assert result.row_count == 3
        assert "name" in result.columns
        assert result.rows[0][1] == "Alice"

    def test_execute_aggregation(self, sample_tables):
        """Test executing aggregation query."""
        result = sample_tables.execute("""
            SELECT user_id, SUM(amount) as total
            FROM orders
            GROUP BY user_id
            ORDER BY user_id
        """)
        assert result.row_count == 2

    def test_get_tables(self, sample_tables):
        """Test listing tables."""
        tables = sample_tables.get_tables()
        assert "users" in tables
        assert "orders" in tables

    def test_get_tables_excludes_internal(self, sample_tables):
        """Test internal tables are excluded."""
        # Create an internal table
        sample_tables.execute("CREATE TABLE _dalcedo_internal (id INTEGER)")
        tables = sample_tables.get_tables()
        assert "_dalcedo_internal" not in tables

    def test_get_schema_description(self, sample_tables):
        """Test schema description generation."""
        schema = sample_tables.get_schema_description()
        assert "users" in schema
        assert "orders" in schema
        assert "id" in schema
        assert "VARCHAR" in schema or "INTEGER" in schema

    def test_table_exists(self, sample_tables):
        """Test checking if table exists."""
        assert sample_tables.table_exists("users") is True
        assert sample_tables.table_exists("nonexistent") is False

    def test_execute_arrow(self, sample_tables):
        """Test executing query returning Arrow table."""
        result = sample_tables.execute_arrow("SELECT * FROM users")
        # Result may be a RecordBatchReader, convert to table
        arrow_table = result.read_all() if hasattr(result, "read_all") else result
        assert arrow_table.num_rows == 3
        assert "name" in arrow_table.column_names

    def test_create_table_from_arrow(self, temp_duckdb):
        """Test creating table from Arrow table."""
        # Create an Arrow table
        arrow_table = pa.table(
            {
                "id": [1, 2, 3],
                "value": ["a", "b", "c"],
            }
        )

        temp_duckdb.create_table_from_arrow("test_arrow", arrow_table)

        # Verify table was created
        assert temp_duckdb.table_exists("test_arrow")
        result = temp_duckdb.execute("SELECT * FROM test_arrow ORDER BY id")
        assert result.row_count == 3

    def test_insert_arrow(self, temp_duckdb):
        """Test inserting Arrow data into existing table."""
        # Create table first
        temp_duckdb.execute("""
            CREATE TABLE test_insert (
                id INTEGER,
                value VARCHAR
            )
        """)
        temp_duckdb.execute("INSERT INTO test_insert VALUES (1, 'existing')")

        # Insert Arrow data
        arrow_table = pa.table(
            {
                "id": [2, 3],
                "value": ["new1", "new2"],
            }
        )
        temp_duckdb.insert_arrow("test_insert", arrow_table)

        result = temp_duckdb.execute("SELECT COUNT(*) FROM test_insert")
        assert result.rows[0][0] == 3

    def test_create_table_from_schema(self, temp_duckdb):
        """Test creating table from schema definition."""
        columns = [
            {"name": "id", "type": "INT64", "mode": "REQUIRED"},
            {"name": "name", "type": "STRING"},
            {"name": "created_at", "type": "TIMESTAMP"},
            {"name": "is_active", "type": "BOOLEAN"},
        ]

        temp_duckdb.create_table_from_schema("schema_test", columns)

        assert temp_duckdb.table_exists("schema_test")

        # Verify column types
        result = temp_duckdb.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'schema_test'
            ORDER BY ordinal_position
        """)
        rows = result.rows
        assert rows[0][0] == "id"
        assert "BIGINT" in rows[0][1]
        assert rows[1][0] == "name"
        assert "VARCHAR" in rows[1][1]

    def test_store_schema_metadata(self, sample_tables):
        """Test storing schema metadata."""
        sample_tables.store_schema_metadata()

        # Check metadata table was created
        assert sample_tables.table_exists("_dalcedo_schema_metadata")

        result = sample_tables.execute("""
            SELECT table_name, column_name
            FROM _dalcedo_schema_metadata
            WHERE table_name = 'users'
            ORDER BY column_name
        """)
        assert result.row_count >= 3  # users has at least 3 columns

    def test_close_connection(self, temp_dir):
        """Test closing connection."""
        db_path = temp_dir / "test.duckdb"
        service = DuckDBService(db_path=db_path)

        # Force connection creation
        _ = service.connection

        service.close()
        assert service._connection is None

        # Connection should be recreated on next access
        _ = service.connection
        assert service._connection is not None

        service.close()

    def test_execute_error(self, temp_duckdb):
        """Test executing invalid SQL raises error."""
        with pytest.raises(Exception):
            temp_duckdb.execute("SELECT * FROM nonexistent_table")

    def test_get_schema_description_with_dbt_metadata(self, sample_tables):
        """Test schema description includes DBT metadata when available."""
        # Store some DBT metadata
        sample_tables.execute("""
            CREATE TABLE _dalcedo_dbt_tables (
                table_name VARCHAR PRIMARY KEY,
                description VARCHAR
            )
        """)
        sample_tables.execute("""
            INSERT INTO _dalcedo_dbt_tables VALUES
            ('users', 'Contains user account information')
        """)

        sample_tables.execute("""
            CREATE TABLE _dalcedo_dbt_columns (
                table_name VARCHAR,
                column_name VARCHAR,
                description VARCHAR,
                PRIMARY KEY (table_name, column_name)
            )
        """)
        sample_tables.execute("""
            INSERT INTO _dalcedo_dbt_columns VALUES
            ('users', 'email', 'User email address for notifications')
        """)

        schema = sample_tables.get_schema_description()
        assert "Contains user account information" in schema
        assert "User email address" in schema

    def test_create_schema(self, temp_duckdb):
        """Test creating a new schema."""
        temp_duckdb.create_schema("analytics")
        schemas = temp_duckdb.get_schemas()
        assert "analytics" in schemas

    def test_drop_schema(self, temp_duckdb):
        """Test dropping a schema."""
        temp_duckdb.create_schema("temp_schema")
        assert "temp_schema" in temp_duckdb.get_schemas()

        temp_duckdb.drop_schema("temp_schema")
        assert "temp_schema" not in temp_duckdb.get_schemas()

    def test_get_schemas(self, temp_duckdb):
        """Test listing schemas."""
        temp_duckdb.create_schema("schema1")
        temp_duckdb.create_schema("schema2")

        schemas = temp_duckdb.get_schemas()
        assert "schema1" in schemas
        assert "schema2" in schemas
        # System schemas should be excluded
        assert "information_schema" not in schemas
        assert "pg_catalog" not in schemas

    def test_create_table_from_arrow_with_schema(self, temp_duckdb):
        """Test creating table in a specific schema."""
        temp_duckdb.create_schema("myschema")

        arrow_table = pa.table({"id": [1, 2], "value": ["a", "b"]})
        temp_duckdb.create_table_from_arrow("mytable", arrow_table, schema_name="myschema")

        # Table should exist in the schema
        assert temp_duckdb.table_exists("mytable", schema_name="myschema")

        # Query should work with qualified name
        result = temp_duckdb.execute("SELECT * FROM myschema.mytable ORDER BY id")
        assert result.row_count == 2

    def test_get_tables_with_schema(self, temp_duckdb):
        """Test listing tables filtered by schema."""
        temp_duckdb.create_schema("schema_a")
        temp_duckdb.create_schema("schema_b")

        temp_duckdb.execute("CREATE TABLE schema_a.table1 (id INTEGER)")
        temp_duckdb.execute("CREATE TABLE schema_a.table2 (id INTEGER)")
        temp_duckdb.execute("CREATE TABLE schema_b.table3 (id INTEGER)")

        tables_a = temp_duckdb.get_tables(schema_name="schema_a")
        assert "table1" in tables_a
        assert "table2" in tables_a
        assert "table3" not in tables_a

        tables_b = temp_duckdb.get_tables(schema_name="schema_b")
        assert "table3" in tables_b
        assert "table1" not in tables_b

    def test_get_tables_by_schema(self, temp_duckdb):
        """Test getting tables organized by schema."""
        temp_duckdb.create_schema("analytics")
        temp_duckdb.create_schema("dbt_tests")

        temp_duckdb.execute("CREATE TABLE analytics.users (id INTEGER)")
        temp_duckdb.execute("CREATE TABLE analytics.events (id INTEGER)")
        temp_duckdb.execute("CREATE TABLE dbt_tests.results (id INTEGER)")

        tables_by_schema = temp_duckdb.get_tables_by_schema()

        assert "analytics" in tables_by_schema
        assert "dbt_tests" in tables_by_schema
        assert "users" in tables_by_schema["analytics"]
        assert "events" in tables_by_schema["analytics"]
        assert "results" in tables_by_schema["dbt_tests"]

    def test_get_schema_description_multi_schema(self, temp_duckdb):
        """Test schema description shows tables from multiple schemas."""
        temp_duckdb.create_schema("analytics")
        temp_duckdb.create_schema("dbt_tests")

        temp_duckdb.execute("CREATE TABLE analytics.users (id INTEGER, name VARCHAR)")
        temp_duckdb.execute("CREATE TABLE dbt_tests.results (test_id VARCHAR, status VARCHAR)")

        schema_desc = temp_duckdb.get_schema_description()

        # Should show schema headers
        assert "=== Schema: analytics ===" in schema_desc
        assert "=== Schema: dbt_tests ===" in schema_desc

        # Should show qualified table names
        assert "analytics.users" in schema_desc
        assert "dbt_tests.results" in schema_desc
