"""Shared test fixtures for Dalcedo tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from dalcedo.services.duckdb import DuckDBService


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_duckdb(temp_dir: Path) -> DuckDBService:
    """Create a temporary DuckDB instance for testing."""
    db_path = temp_dir / "test.duckdb"
    db = DuckDBService(db_path=db_path)
    yield db
    db.close()


@pytest.fixture
def sample_tables(temp_duckdb: DuckDBService) -> DuckDBService:
    """Create sample tables in the test database."""
    # Create a sample table
    temp_duckdb.connection.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL,
            email VARCHAR,
            created_at TIMESTAMP
        )
    """)
    temp_duckdb.connection.execute("""
        INSERT INTO users VALUES
        (1, 'Alice', 'alice@example.com', '2024-01-01 00:00:00'),
        (2, 'Bob', 'bob@example.com', '2024-01-02 00:00:00'),
        (3, 'Charlie', 'charlie@example.com', '2024-01-03 00:00:00')
    """)

    # Create another table
    temp_duckdb.connection.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            amount DECIMAL(10, 2),
            status VARCHAR
        )
    """)
    temp_duckdb.connection.execute("""
        INSERT INTO orders VALUES
        (1, 1, 100.00, 'completed'),
        (2, 1, 50.00, 'pending'),
        (3, 2, 75.00, 'completed')
    """)

    return temp_duckdb
