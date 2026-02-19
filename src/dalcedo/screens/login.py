"""Login screen for configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Checkbox, Footer, Header, Input, Label, Select, Static

from dalcedo.config.settings import Connection, Plugin
from dalcedo.connectors import get_all_connectors, get_connector

if TYPE_CHECKING:
    from dalcedo.app import DalcedoApp


# =============================================================================
# Connection Components
# =============================================================================


class ConnectionListItem(Vertical):
    """A single connection row in the list."""

    class EditRequested(Message):
        def __init__(self, connection_id: str) -> None:
            self.connection_id = connection_id
            super().__init__()

    class RemoveRequested(Message):
        def __init__(self, connection_id: str) -> None:
            self.connection_id = connection_id
            super().__init__()

    def __init__(self, connection: Connection, plugin_count: int = 0) -> None:
        super().__init__()
        self.connection = connection
        self.plugin_count = plugin_count

    def compose(self) -> ComposeResult:
        connector_cls = get_connector(self.connection.type)
        connector_name = connector_cls.display_name if connector_cls else self.connection.type

        with Horizontal(classes="connection-row"):
            yield Static(f"{self.connection.name}", classes="connection-name")
            yield Static(f"({connector_name})", classes="connection-type")
            yield Static(f"[{self.plugin_count} plugins]", classes="connection-plugins")
            yield Static("", classes="row-spacer")
            yield Button("Edit", id=f"edit-conn-{self.connection.id}", classes="item-btn")
            yield Button(
                "Remove",
                id=f"remove-conn-{self.connection.id}",
                variant="error",
                classes="item-btn",
            )

        # Show config summary
        config = self.connection.config
        if self.connection.type == "bigquery":
            project = config.get("project_id", "")
            location = config.get("location", "US")
            if project:
                yield Static(f"    {project} ({location})", classes="item-desc")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("edit-conn-"):
            self.post_message(self.EditRequested(self.connection.id))
        elif event.button.id and event.button.id.startswith("remove-conn-"):
            self.post_message(self.RemoveRequested(self.connection.id))


class ConnectionEditorModal(ModalScreen[Connection | None]):
    """Modal screen for adding/editing a connection."""

    CSS = """
    ConnectionEditorModal {
        align: center middle;
    }

    #connection-editor {
        width: 70;
        max-height: 90%;
        border: round $primary;
        padding: 2;
        background: $surface;
    }

    #connection-editor #title {
        text-align: center;
        text-style: bold;
        margin-bottom: 2;
    }

    #connection-editor Label {
        margin-top: 1;
        margin-bottom: 0;
    }

    #connection-editor Input, #connection-editor Select {
        margin-bottom: 1;
    }

    #connection-editor .buttons {
        layout: horizontal;
        margin-top: 2;
        width: 100%;
        align: center middle;
    }

    #connection-editor .buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, connection: Connection | None = None) -> None:
        super().__init__()
        self.connection = connection
        self.is_new = connection is None
        self._render_id = 0
        self._rendering = False

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="connection-editor"):
            title = "Add Connection" if self.is_new else f"Edit: {self.connection.name}"
            yield Static(title, id="title")

            yield Label("Connection Name:")
            yield Input(
                placeholder="Production BigQuery",
                value=self.connection.name if self.connection else "",
                id="connection-name",
            )

            yield Label("Connector Type:")
            connectors = get_all_connectors()
            connector_options = [(cls.display_name, name) for name, cls in connectors.items()]
            current_type = self.connection.type if self.connection else "bigquery"
            yield Select(
                connector_options,
                value=current_type if current_type in connectors else connector_options[0][1],
                id="connector-type",
            )

            yield Vertical(id="connection-fields-container")

            with Horizontal(classes="buttons"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn")

    async def on_mount(self) -> None:
        await self._render_connection_fields()

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "connector-type":
            await self._render_connection_fields()

    async def _render_connection_fields(self) -> None:
        if self._rendering:
            return
        self._rendering = True

        try:
            self._render_id += 1
            render_id = self._render_id

            connector_type = self.query_one("#connector-type", Select).value
            connector_cls = get_connector(str(connector_type))

            container = self.query_one("#connection-fields-container", Vertical)
            await container.remove_children()

            if not connector_cls:
                return

            current_config = {}
            if self.connection and self.connection.type == connector_type:
                current_config = self.connection.config

            # Only show connection-level fields
            fields = connector_cls.get_connection_fields()
            widgets_to_mount = []

            for field in fields:
                label_text = f"{field.label}:" if field.required else f"{field.label} (optional):"
                widgets_to_mount.append(Label(label_text))

                value = current_config.get(field.name, field.default)
                widget_id = f"conn-{field.name}-{render_id}"
                if field.field_type == "password":
                    inp = Input(
                        placeholder=field.placeholder,
                        password=True,
                        value=str(value) if value else "",
                        id=widget_id,
                    )
                else:
                    inp = Input(
                        placeholder=field.placeholder,
                        value=str(value) if value else "",
                        id=widget_id,
                    )
                widgets_to_mount.append(inp)

            await container.mount_all(widgets_to_mount)
        finally:
            self._rendering = False

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            await self._save_connection()
        elif event.button.id == "cancel-btn":
            self.dismiss(None)

    async def _save_connection(self) -> None:
        name = self.query_one("#connection-name", Input).value.strip()
        if not name:
            self.notify("Connection name is required", severity="error")
            return

        connector_type = str(self.query_one("#connector-type", Select).value)
        connector_cls = get_connector(connector_type)

        if not connector_cls:
            self.notify(f"Unknown connector: {connector_type}", severity="error")
            return

        # Build connection config from fields
        connection_config = {}
        for field in connector_cls.get_connection_fields():
            try:
                inp = self.query_one(f"#conn-{field.name}-{self._render_id}", Input)
                connection_config[field.name] = inp.value.strip()
            except Exception:
                pass

        # Validate connection config
        errors = connector_cls.validate_connection_config(connection_config)
        if errors:
            self.notify(errors[0], severity="error")
            return

        if self.connection:
            connection = Connection(
                id=self.connection.id,
                name=name,
                type=connector_type,
                config=connection_config,
            )
        else:
            connection = Connection.create(
                name=name,
                connection_type=connector_type,
                config=connection_config,
            )

        self.dismiss(connection)


# =============================================================================
# Plugin Components
# =============================================================================


class PluginListItem(Vertical):
    """A single plugin row in the list."""

    class EditRequested(Message):
        def __init__(self, plugin_id: str) -> None:
            self.plugin_id = plugin_id
            super().__init__()

    class RemoveRequested(Message):
        def __init__(self, plugin_id: str) -> None:
            self.plugin_id = plugin_id
            super().__init__()

    class EnableToggled(Message):
        def __init__(self, plugin_id: str, enabled: bool) -> None:
            self.plugin_id = plugin_id
            self.enabled = enabled
            super().__init__()

    def __init__(self, plugin: Plugin, connection: Connection | None = None) -> None:
        super().__init__()
        self.plugin = plugin
        self.connection = connection

    def compose(self) -> ComposeResult:
        conn_name = self.connection.name if self.connection else "No connection"

        with Horizontal(classes="plugin-row"):
            yield Checkbox(
                self.plugin.name,
                value=self.plugin.enabled,
                id=f"plugin-check-{self.plugin.id}",
            )
            yield Static("", classes="row-spacer")
            yield Button("Edit", id=f"edit-{self.plugin.id}", classes="item-btn")
            yield Button(
                "Remove", id=f"remove-{self.plugin.id}", variant="error", classes="item-btn"
            )

        # Show connection and source config
        yield Static(f"    Connection: {conn_name}", classes="item-desc")
        config = self.plugin.source_config
        if config.get("dataset_id"):
            yield Static(f"    Dataset: {config['dataset_id']}", classes="item-desc")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        self.post_message(self.EnableToggled(self.plugin.id, event.value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("edit-"):
            self.post_message(self.EditRequested(self.plugin.id))
        elif event.button.id and event.button.id.startswith("remove-"):
            self.post_message(self.RemoveRequested(self.plugin.id))


class PluginEditorModal(ModalScreen[Plugin | None]):
    """Modal screen for adding/editing a plugin."""

    CSS = """
    PluginEditorModal {
        align: center middle;
    }

    #plugin-editor {
        width: 70;
        max-height: 90%;
        border: round $primary;
        padding: 2;
        background: $surface;
    }

    #plugin-editor #title {
        text-align: center;
        text-style: bold;
        margin-bottom: 2;
    }

    #plugin-editor Label {
        margin-top: 1;
        margin-bottom: 0;
    }

    #plugin-editor Input, #plugin-editor Select {
        margin-bottom: 1;
    }

    #plugin-editor .buttons {
        layout: horizontal;
        margin-top: 2;
        width: 100%;
        align: center middle;
    }

    #plugin-editor .buttons Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        connections: list[Connection],
        plugin: Plugin | None = None,
    ) -> None:
        super().__init__()
        self.connections = connections
        self.plugin = plugin
        self.is_new = plugin is None
        self._render_id = 0
        self._rendering = False

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="plugin-editor"):
            title = "Add Data Source" if self.is_new else f"Edit: {self.plugin.name}"
            yield Static(title, id="title")

            yield Label("Data Source Name:")
            yield Input(
                placeholder="analytics (becomes DuckDB schema)",
                value=self.plugin.name if self.plugin else "",
                id="plugin-name",
            )

            yield Label("Connection:")
            if not self.connections:
                yield Static("No connections available. Add a connection first.", id="no-conns")
            else:
                connection_options = [(c.name, c.id) for c in self.connections]
                current_conn = self.plugin.connection_id if self.plugin else self.connections[0].id
                yield Select(
                    connection_options,
                    value=current_conn,
                    id="connection-select",
                )

            yield Vertical(id="source-fields-container")

            with Horizontal(classes="buttons"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn")

    async def on_mount(self) -> None:
        if self.connections:
            await self._render_source_fields()

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "connection-select":
            await self._render_source_fields()

    def _get_selected_connection(self) -> Connection | None:
        try:
            conn_id = str(self.query_one("#connection-select", Select).value)
            for c in self.connections:
                if c.id == conn_id:
                    return c
        except Exception:
            pass
        return None

    async def _render_source_fields(self) -> None:
        if self._rendering:
            return
        self._rendering = True

        try:
            self._render_id += 1
            render_id = self._render_id

            connection = self._get_selected_connection()
            if not connection:
                return

            connector_cls = get_connector(connection.type)
            container = self.query_one("#source-fields-container", Vertical)
            await container.remove_children()

            if not connector_cls:
                return

            current_config = {}
            if self.plugin and self.plugin.connection_id == connection.id:
                current_config = self.plugin.source_config

            # Only show source-level fields
            fields = connector_cls.get_source_fields()
            widgets_to_mount = []

            for field in fields:
                label_text = f"{field.label}:" if field.required else f"{field.label} (optional):"
                widgets_to_mount.append(Label(label_text))

                value = current_config.get(field.name, field.default)
                widget_id = f"source-{field.name}-{render_id}"
                if field.field_type == "password":
                    inp = Input(
                        placeholder=field.placeholder,
                        password=True,
                        value=str(value) if value else "",
                        id=widget_id,
                    )
                else:
                    inp = Input(
                        placeholder=field.placeholder,
                        value=str(value) if value else "",
                        id=widget_id,
                    )
                widgets_to_mount.append(inp)

            await container.mount_all(widgets_to_mount)
        finally:
            self._rendering = False

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            await self._save_plugin()
        elif event.button.id == "cancel-btn":
            self.dismiss(None)

    async def _save_plugin(self) -> None:
        name = self.query_one("#plugin-name", Input).value.strip()
        if not name:
            self.notify("Data source name is required", severity="error")
            return

        connection = self._get_selected_connection()
        if not connection:
            self.notify("Please select a connection", severity="error")
            return

        connector_cls = get_connector(connection.type)
        if not connector_cls:
            self.notify(f"Unknown connector: {connection.type}", severity="error")
            return

        # Build source config from fields
        source_config = {}
        for field in connector_cls.get_source_fields():
            try:
                inp = self.query_one(f"#source-{field.name}-{self._render_id}", Input)
                source_config[field.name] = inp.value.strip()
            except Exception:
                pass

        # Validate source config
        errors = connector_cls.validate_source_config(source_config)
        if errors:
            self.notify(errors[0], severity="error")
            return

        # Test connection with both configs
        try:
            self.notify(f"Testing connection to {name}...", severity="information")
            connector = connector_cls(
                connection_config=connection.config,
                source_config=source_config,
            )
            await connector.test_connection()
        except Exception as e:
            self.notify(f"Connection failed: {e}", severity="error")
            return

        if self.plugin:
            plugin = Plugin(
                id=self.plugin.id,
                name=name,
                connection_id=connection.id,
                source_config=source_config,
                enabled=self.plugin.enabled,
            )
        else:
            plugin = Plugin.create(
                name=name,
                connection_id=connection.id,
                source_config=source_config,
            )

        self.dismiss(plugin)


# =============================================================================
# Login Screen
# =============================================================================


class LoginScreen(Screen):
    """Configuration and login screen."""

    CSS = """
    LoginScreen {
        height: 100%;
        width: 100%;
    }

    #login-container {
        height: auto;
        min-height: 100%;
    }

    .config-section {
        margin-top: 1;
        margin-bottom: 1;
        padding: 1;
        border: round $primary-darken-2;
        width: 100%;
        height: auto;
    }

    .config-section .section-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .config-section .section-list {
        width: 100%;
        height: auto;
    }

    ConnectionListItem, PluginListItem {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    .connection-row, .plugin-row {
        width: 100%;
        height: auto;
        min-height: 3;
        align: left middle;
    }

    .connection-name {
        width: auto;
        text-style: bold;
        margin-right: 1;
    }

    .connection-type, .plugin-connection {
        color: $text-muted;
        width: auto;
        margin-right: 1;
    }

    .connection-plugins {
        color: $primary;
        width: auto;
        margin-right: 1;
    }

    .plugin-row Checkbox {
        width: auto;
        margin-right: 1;
    }

    .row-spacer {
        width: 1fr;
    }

    .item-btn {
        min-width: 10;
        margin-left: 2;
    }

    .item-desc {
        color: $text-muted;
        margin-left: 2;
    }

    .add-btn {
        margin-top: 1;
    }

    .no-items {
        color: $text-muted;
        text-style: italic;
        margin: 1 0;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="login-container"):
            yield Static("Configure Dalcedo", id="title")

            # LLM Provider
            yield Label("LLM Provider:")
            yield Select(
                [
                    ("Anthropic (Claude)", "anthropic"),
                    ("OpenAI (GPT)", "openai"),
                    ("Google (Gemini)", "gemini"),
                ],
                value="anthropic",
                id="llm-provider",
            )

            yield Label("API Key:", id="api-key-label")
            yield Input(placeholder="sk-ant-... or sk-...", password=True, id="api-key")

            # Token limits section
            with Vertical(classes="config-section"):
                yield Static("Token Limits (optional)", classes="section-title")
                yield Label("Daily limit (leave empty for unlimited):")
                yield Input(placeholder="e.g., 100000", id="daily-limit")
                yield Label("Weekly limit (leave empty for unlimited):")
                yield Input(placeholder="e.g., 500000", id="weekly-limit")

            # Connections section
            with Vertical(classes="config-section"):
                yield Static("Connections (Data Warehouses)", classes="section-title")
                yield Vertical(id="connections-list", classes="section-list")
                yield Button("+ Add Connection", id="add-connection-btn", classes="add-btn")

            # Plugins section
            with Vertical(classes="config-section"):
                yield Static("Data Sources (Plugins)", classes="section-title")
                yield Vertical(id="plugins-list", classes="section-list")
                yield Button("+ Add Data Source", id="add-plugin-btn", classes="add-btn")

            yield Button("Save & Connect", id="save-btn", variant="primary")
        yield Footer()

    async def on_mount(self) -> None:
        app: DalcedoApp = self.app  # type: ignore

        if app.config:
            self.query_one("#llm-provider", Select).value = app.config.llm_provider
            if app.config.credentials:
                self.query_one("#api-key", Input).value = app.config.credentials.llm_api_key
            # Load token limits
            if app.config.daily_token_limit:
                self.query_one("#daily-limit", Input).value = str(app.config.daily_token_limit)
            if app.config.weekly_token_limit:
                self.query_one("#weekly-limit", Input).value = str(app.config.weekly_token_limit)

        await self._render_connections_list()
        await self._render_plugins_list()
        self._update_api_key_label()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "llm-provider":
            self._update_api_key_label()

    def _update_api_key_label(self) -> None:
        provider = self.query_one("#llm-provider", Select).value
        label = self.query_one("#api-key-label", Label)
        api_input = self.query_one("#api-key", Input)

        if provider == "anthropic":
            label.update("Anthropic API Key:")
            api_input.placeholder = "sk-ant-..."
        elif provider == "gemini":
            label.update("Google AI API Key:")
            api_input.placeholder = "AIza..."
        else:
            label.update("OpenAI API Key:")
            api_input.placeholder = "sk-..."

    async def _render_connections_list(self) -> None:
        app: DalcedoApp = self.app  # type: ignore
        connections_list = self.query_one("#connections-list", Vertical)
        await connections_list.remove_children()

        connections = app.config.connections if app.config else []
        plugins = app.config.plugins if app.config else []

        if not connections:
            await connections_list.mount(Static("No connections configured.", classes="no-items"))
        else:
            items = []
            for conn in connections:
                plugin_count = sum(1 for p in plugins if p.connection_id == conn.id)
                items.append(ConnectionListItem(conn, plugin_count))
            await connections_list.mount_all(items)

    async def _render_plugins_list(self) -> None:
        app: DalcedoApp = self.app  # type: ignore
        plugins_list = self.query_one("#plugins-list", Vertical)
        await plugins_list.remove_children()

        plugins = app.config.plugins if app.config else []

        if not plugins:
            await plugins_list.mount(Static("No data sources configured.", classes="no-items"))
        else:
            items = []
            for plugin in plugins:
                connection = app.config.get_connection_by_id(plugin.connection_id) if app.config else None
                items.append(PluginListItem(plugin, connection))
            await plugins_list.mount_all(items)

    # Connection handlers
    def on_connection_list_item_edit_requested(
        self, event: ConnectionListItem.EditRequested
    ) -> None:
        app: DalcedoApp = self.app  # type: ignore
        if app.config:
            connection = app.config.get_connection_by_id(event.connection_id)
            if connection:
                self.app.push_screen(
                    ConnectionEditorModal(connection), self._on_connection_edited
                )

    async def on_connection_list_item_remove_requested(
        self, event: ConnectionListItem.RemoveRequested
    ) -> None:
        app: DalcedoApp = self.app  # type: ignore
        if app.config:
            # Check if any plugins use this connection
            plugins_using = [p for p in app.config.plugins if p.connection_id == event.connection_id]
            if plugins_using:
                self.notify(
                    f"Cannot remove: {len(plugins_using)} data source(s) use this connection",
                    severity="error",
                )
                return

            app.config.connections = [
                c for c in app.config.connections if c.id != event.connection_id
            ]
            await self._render_connections_list()
            self.notify("Connection removed", severity="information")

    async def _on_connection_edited(self, connection: Connection | None) -> None:
        if connection is None:
            return

        app: DalcedoApp = self.app  # type: ignore

        if not app.config:
            from dalcedo.config.settings import AppConfig
            app.config = AppConfig()

        existing_idx = None
        for i, c in enumerate(app.config.connections):
            if c.id == connection.id:
                existing_idx = i
                break

        if existing_idx is not None:
            app.config.connections[existing_idx] = connection
            self.notify(f"Connection '{connection.name}' updated", severity="information")
        else:
            app.config.connections.append(connection)
            self.notify(f"Connection '{connection.name}' added", severity="information")

        await self._render_connections_list()
        await self._render_plugins_list()

    # Plugin handlers
    def on_plugin_list_item_edit_requested(self, event: PluginListItem.EditRequested) -> None:
        app: DalcedoApp = self.app  # type: ignore
        if app.config:
            plugin = app.config.get_plugin_by_id(event.plugin_id)
            if plugin:
                self.app.push_screen(
                    PluginEditorModal(app.config.connections, plugin),
                    self._on_plugin_edited,
                )

    async def on_plugin_list_item_remove_requested(self, event: PluginListItem.RemoveRequested) -> None:
        app: DalcedoApp = self.app  # type: ignore
        if app.config:
            app.config.plugins = [p for p in app.config.plugins if p.id != event.plugin_id]
            await self._render_plugins_list()
            await self._render_connections_list()  # Update plugin counts
            self.notify("Data source removed", severity="information")

    def on_plugin_list_item_enable_toggled(self, event: PluginListItem.EnableToggled) -> None:
        app: DalcedoApp = self.app  # type: ignore
        if app.config:
            for i, p in enumerate(app.config.plugins):
                if p.id == event.plugin_id:
                    app.config.plugins[i] = Plugin(
                        id=p.id,
                        name=p.name,
                        connection_id=p.connection_id,
                        source_config=p.source_config,
                        enabled=event.enabled,
                    )
                    break

    async def _on_plugin_edited(self, plugin: Plugin | None) -> None:
        if plugin is None:
            return

        app: DalcedoApp = self.app  # type: ignore

        if not app.config:
            from dalcedo.config.settings import AppConfig
            app.config = AppConfig()

        existing_idx = None
        for i, p in enumerate(app.config.plugins):
            if p.id == plugin.id:
                existing_idx = i
                break

        if existing_idx is not None:
            app.config.plugins[existing_idx] = plugin
            self.notify(f"Data source '{plugin.name}' updated", severity="information")
        else:
            app.config.plugins.append(plugin)
            self.notify(f"Data source '{plugin.name}' added", severity="information")

        await self._render_plugins_list()
        await self._render_connections_list()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            await self._save_config()
        elif event.button.id == "add-connection-btn":
            self.app.push_screen(ConnectionEditorModal(), self._on_connection_edited)
        elif event.button.id == "add-plugin-btn":
            app: DalcedoApp = self.app  # type: ignore
            connections = app.config.connections if app.config else []
            if not connections:
                self.notify("Add a connection first", severity="warning")
                return
            self.app.push_screen(PluginEditorModal(connections), self._on_plugin_edited)

    async def _save_config(self) -> None:
        from dalcedo.config.settings import AppConfig, Credentials
        from dalcedo.services.llm import AnthropicLLMService, GeminiLLMService, OpenAILLMService
        from dalcedo.services.llm.base import LLMError

        app: DalcedoApp = self.app  # type: ignore

        llm_provider = str(self.query_one("#llm-provider", Select).value)
        api_key = self.query_one("#api-key", Input).value.strip()

        # Parse token limits
        daily_limit_str = self.query_one("#daily-limit", Input).value.strip()
        weekly_limit_str = self.query_one("#weekly-limit", Input).value.strip()

        daily_limit = None
        weekly_limit = None
        if daily_limit_str:
            try:
                daily_limit = int(daily_limit_str)
                if daily_limit <= 0:
                    self.notify("Daily limit must be a positive number", severity="error")
                    return
            except ValueError:
                self.notify("Daily limit must be a number", severity="error")
                return

        if weekly_limit_str:
            try:
                weekly_limit = int(weekly_limit_str)
                if weekly_limit <= 0:
                    self.notify("Weekly limit must be a positive number", severity="error")
                    return
            except ValueError:
                self.notify("Weekly limit must be a number", severity="error")
                return

        if not api_key:
            self.notify("API Key is required", severity="error")
            return

        connections = app.config.connections if app.config else []
        plugins = app.config.plugins if app.config else []

        if not plugins:
            self.notify("At least one data source is required", severity="error")
            return

        # Verify all plugins have valid connections
        for plugin in plugins:
            if not any(c.id == plugin.connection_id for c in connections):
                self.notify(
                    f"Data source '{plugin.name}' has invalid connection", severity="error"
                )
                return

        try:
            provider_names = {"anthropic": "Anthropic", "openai": "OpenAI", "gemini": "Google AI"}
            self.notify(
                f"Testing {provider_names.get(llm_provider, llm_provider)} API...",
                severity="information",
            )

            if llm_provider == "anthropic":
                llm = AnthropicLLMService(api_key=api_key)
            elif llm_provider == "gemini":
                llm = GeminiLLMService(api_key=api_key)
            else:
                llm = OpenAILLMService(api_key=api_key)
            await llm.test_connection()

            config = AppConfig(
                connections=connections,
                plugins=plugins,
                credentials=Credentials(llm_api_key=api_key),
                llm_provider=llm_provider,
                daily_token_limit=daily_limit,
                weekly_token_limit=weekly_limit,
            )
            app.config_storage.save_config(config)
            app.config = config
            app.initialize_services()

            self.notify("Configuration saved!", severity="information")
            self.app.pop_screen()

        except LLMError as e:
            self.notify(str(e), severity="error")
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")

    def action_cancel(self) -> None:
        self.app.pop_screen()
