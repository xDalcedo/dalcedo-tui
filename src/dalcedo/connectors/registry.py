"""Connector registry for plugin discovery and management."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dalcedo.connectors.base import BaseConnector

# Global registry of available connectors
_CONNECTORS: dict[str, type["BaseConnector"]] = {}


def register_connector(connector_cls: type["BaseConnector"]) -> type["BaseConnector"]:
    """Register a connector class.

    Can be used as a decorator:

        @register_connector
        class MyConnector(BaseConnector):
            name = "my_connector"
            ...
    """
    _CONNECTORS[connector_cls.name] = connector_cls
    return connector_cls


def get_connector(name: str) -> type["BaseConnector"] | None:
    """Get a connector class by name."""
    return _CONNECTORS.get(name)


def get_all_connectors() -> dict[str, type["BaseConnector"]]:
    """Get all registered connectors."""
    return _CONNECTORS.copy()


def list_connector_names() -> list[str]:
    """List all registered connector names."""
    return list(_CONNECTORS.keys())


def load_builtin_connectors() -> None:
    """Load built-in connectors (free tier)."""
    # Import to trigger registration via @register_connector decorator
    from dalcedo.connectors import bigquery  # noqa: F401


def load_plugins() -> None:
    """Discover and load paid connector plugins.

    Looks for the dalcedo_pro package and loads its connectors.
    """
    try:
        import dalcedo_pro

        if hasattr(dalcedo_pro, "register_connectors"):
            dalcedo_pro.register_connectors()
    except ImportError:
        # dalcedo_pro not installed - free tier only
        pass


def initialize_connectors() -> None:
    """Initialize all connectors (builtin + plugins)."""
    load_builtin_connectors()
    load_plugins()
