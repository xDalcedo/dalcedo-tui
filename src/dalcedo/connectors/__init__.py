"""Connectors for data sources."""

from dalcedo.connectors.base import BaseConnector, ConnectorField, SyncUpdate
from dalcedo.connectors.registry import (
    get_all_connectors,
    get_connector,
    initialize_connectors,
    list_connector_names,
    register_connector,
)

__all__ = [
    "BaseConnector",
    "ConnectorField",
    "SyncUpdate",
    "get_all_connectors",
    "get_connector",
    "initialize_connectors",
    "list_connector_names",
    "register_connector",
]
