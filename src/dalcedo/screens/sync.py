"""Sync screen for data source synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Header, ProgressBar, Static
from textual.worker import Worker, get_current_worker

from dalcedo.config.settings import Plugin

if TYPE_CHECKING:
    from dalcedo.app import DalcedoApp


@dataclass
class PluginSyncState:
    """Track sync state for a single plugin."""

    plugin: Plugin
    selected: bool = True
    status: str = "waiting"  # "waiting", "syncing", "completed", "error"
    progress: float = 0.0
    message: str = ""
    current_table: str | None = None


class PluginSyncRow(Static):
    """A row showing plugin sync state."""

    class SelectionChanged(Message, bubble=True):
        """Plugin selection changed."""

        def __init__(self, plugin_id: str, selected: bool) -> None:
            self.plugin_id = plugin_id
            self.selected = selected
            super().__init__()

    def __init__(
        self, state: PluginSyncState, last_sync: str | None = None, connection_type: str = ""
    ) -> None:
        super().__init__()
        self.state = state
        self.last_sync = last_sync
        self.connection_type = connection_type

    def compose(self) -> ComposeResult:
        with Horizontal(classes="plugin-sync-row"):
            yield Checkbox(
                "",
                value=self.state.selected,
                id=f"sync-check-{self.state.plugin.id}",
                disabled=self.state.status == "syncing",
            )
            yield Static(self.state.plugin.name, classes="plugin-name")
            yield Static(f"({self.connection_type})", classes="plugin-type")
            yield Static(
                self._get_status_text(),
                id=f"status-{self.state.plugin.id}",
                classes="plugin-status",
            )

        # Last sync info
        if self.last_sync:
            yield Static(f"Last sync: {self.last_sync}", classes="plugin-last-sync")
        else:
            yield Static("Never synced", classes="plugin-last-sync")

    def _get_status_text(self) -> str:
        """Get status text based on state."""
        if self.state.status == "waiting":
            return ""
        elif self.state.status == "syncing":
            return f"⟳ {self.state.message}" if self.state.message else "⟳ Syncing..."
        elif self.state.status == "completed":
            return "✓ Done"
        elif self.state.status == "error":
            return f"✗ {self.state.message}" if self.state.message else "✗ Error"
        return ""

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox toggle."""
        self.post_message(self.SelectionChanged(self.state.plugin.id, event.value))

    def update_status(self, message: str, status: str) -> None:
        """Update the status display."""
        self.state.status = status
        self.state.message = message
        try:
            status_widget = self.query_one(f"#status-{self.state.plugin.id}", Static)
            status_widget.update(self._get_status_text())
        except Exception:
            pass


class SyncScreen(Screen):
    """Data source sync screen with multi-plugin support."""

    CSS = """
    #sync-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }

    #title {
        text-style: bold;
        margin-bottom: 1;
    }

    #description {
        color: $text-muted;
        margin-bottom: 1;
    }

    #sync-plugins-container {
        height: auto;
        max-height: 50%;
        margin: 1 0;
    }

    #global-progress-container {
        margin: 1 0;
        height: auto;
    }

    #global-progress {
        margin: 0;
        width: 100%;
    }

    #global-status {
        color: $primary;
        margin-top: 1;
    }

    PluginSyncRow {
        height: auto;
        margin-bottom: 1;
        width: 100%;
    }

    .plugin-sync-row {
        layout: horizontal;
        height: 3;
        align: left middle;
        width: 100%;
    }

    .plugin-sync-row Checkbox {
        width: auto;
        min-width: 5;
        margin-right: 1;
    }

    .plugin-name {
        width: auto;
        text-style: bold;
        margin-right: 1;
    }

    .plugin-type {
        color: $text-muted;
        width: auto;
        margin-right: 2;
    }

    .plugin-status {
        color: $primary;
        width: auto;
    }

    .plugin-last-sync {
        color: $text-muted;
        margin-left: 5;
    }

    .selection-buttons {
        layout: horizontal;
        margin: 1 0;
        align: left middle;
    }

    .selection-buttons Button {
        width: auto;
        min-width: 14;
        margin-right: 1;
    }

    .buttons {
        margin-top: 1;
    }

    .buttons Button {
        margin-right: 1;
    }

    #no-plugins {
        color: $text-muted;
        text-style: italic;
        margin: 2 0;
        text-align: center;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.plugin_states: dict[str, PluginSyncState] = {}
        self._syncing = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="sync-container"):
            yield Static("Sync Data Sources", id="title")
            yield Static("Select data sources to sync.", id="description")

            # Global progress bar
            with Container(id="global-progress-container"):
                yield ProgressBar(id="global-progress", total=100, show_eta=False, show_percentage=True)
                yield Static("", id="global-status")

            with Horizontal(classes="selection-buttons"):
                yield Button("Select All", id="select-all-btn")
                yield Button("Deselect All", id="deselect-all-btn")

            yield VerticalScroll(id="sync-plugins-container")

            with Horizontal(classes="buttons"):
                yield Button("Start Sync", id="sync-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn", variant="default")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize UI state."""
        app: DalcedoApp = self.app  # type: ignore

        # Hide progress bar initially
        self.query_one("#global-progress-container", Container).display = False

        # Initialize plugin states
        self.plugin_states = {}
        if app.config:
            for plugin in app.config.get_enabled_plugins():
                self.plugin_states[plugin.id] = PluginSyncState(plugin=plugin)

        self._render_plugins_list()

    def _render_plugins_list(self) -> None:
        """Render the list of plugins for sync."""
        app: DalcedoApp = self.app  # type: ignore
        container = self.query_one("#sync-plugins-container", VerticalScroll)
        container.remove_children()

        if not self.plugin_states:
            container.mount(
                Static("No plugins configured. Go to /login to add plugins.", id="no-plugins")
            )
            return

        for plugin_id, state in self.plugin_states.items():
            last_sync = app.config_storage.get_plugin_sync_age_description(plugin_id)
            # Get connection type for this plugin
            connection_type = ""
            if app.config:
                connection = app.config.get_plugin_connection(state.plugin)
                if connection:
                    connection_type = connection.type
            container.mount(PluginSyncRow(state, last_sync, connection_type))

    def on_plugin_sync_row_selection_changed(self, event: PluginSyncRow.SelectionChanged) -> None:
        """Handle plugin selection change."""
        if event.plugin_id in self.plugin_states:
            self.plugin_states[event.plugin_id].selected = event.selected

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "sync-btn":
            self._start_sync()
        elif event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "select-all-btn":
            self._select_all(True)
        elif event.button.id == "deselect-all-btn":
            self._select_all(False)

    def _select_all(self, selected: bool) -> None:
        """Select or deselect all plugins."""
        for plugin_id, state in self.plugin_states.items():
            state.selected = selected
            try:
                checkbox = self.query_one(f"#sync-check-{plugin_id}", Checkbox)
                checkbox.value = selected
            except Exception:
                pass

    def _start_sync(self) -> None:
        """Start sync in background worker."""
        app: DalcedoApp = self.app  # type: ignore

        if not app.db_service:
            self.notify("Please configure your connection first", severity="error")
            return

        # Get selected plugins
        selected_plugins = [state.plugin for state in self.plugin_states.values() if state.selected]

        if not selected_plugins:
            self.notify("Please select at least one plugin to sync", severity="warning")
            return

        self._syncing = True
        self.query_one("#sync-btn", Button).disabled = True
        self.query_one("#cancel-btn", Button).disabled = True
        self.query_one("#select-all-btn", Button).disabled = True
        self.query_one("#deselect-all-btn", Button).disabled = True
        self.query_one("#global-progress-container", Container).display = True
        self.query_one("#global-progress", ProgressBar).update(progress=0)
        self.query_one("#global-status", Static).update("Starting sync...")

        # Disable checkboxes
        for plugin_id in self.plugin_states:
            try:
                checkbox = self.query_one(f"#sync-check-{plugin_id}", Checkbox)
                checkbox.disabled = True
            except Exception:
                pass

        self.run_worker(self._do_sync, exclusive=True, thread=True)

    def _do_sync(self) -> None:
        """Execute sync operation for all selected plugins (runs in thread)."""
        app: DalcedoApp = self.app  # type: ignore
        worker = get_current_worker()

        if not app.db_service:
            self.app.call_from_thread(
                self._update_global_status, "Services not initialized", is_error=True
            )
            return

        # Get selected plugins in order
        selected_states = [state for state in self.plugin_states.values() if state.selected]
        total_plugins = len(selected_states)

        for idx, state in enumerate(selected_states):
            if worker.is_cancelled:
                break

            plugin = state.plugin
            state.status = "syncing"
            state.progress = 0.0
            state.message = "Starting..."

            # Update global progress
            global_progress = idx / total_plugins
            self.app.call_from_thread(
                self._update_global_progress,
                global_progress,
                f"Syncing {plugin.name}...",
            )

            # Update plugin row status
            self.app.call_from_thread(
                self._update_plugin_row_status,
                plugin.id,
                "Starting...",
                "syncing",
            )

            # Get connector for this plugin
            connector = app.connectors.get(plugin.name)
            if not connector:
                state.status = "error"
                state.message = "Connector not initialized"
                self.app.call_from_thread(
                    self._update_plugin_row_status,
                    plugin.id,
                    "Connector not initialized",
                    "error",
                )
                continue

            # Run sync for this plugin
            try:
                for update in connector.sync(app.db_service, schema_name=plugin.schema_name):
                    if worker.is_cancelled:
                        break

                    state.progress = update.progress
                    state.message = update.message
                    state.current_table = update.table

                    # Update global progress (plugin progress within this plugin's slice)
                    plugin_contribution = update.progress / total_plugins
                    global_progress = (idx / total_plugins) + plugin_contribution
                    self.app.call_from_thread(
                        self._update_global_progress,
                        global_progress,
                        f"Syncing {plugin.name}...",  # Simple global status
                    )

                    # Update plugin row with detailed message
                    self.app.call_from_thread(
                        self._update_plugin_row_status,
                        plugin.id,
                        update.message,
                        "syncing",
                    )

                if not worker.is_cancelled:
                    state.status = "completed"
                    state.message = ""
                    # Record sync time for this plugin
                    app.config_storage.record_plugin_sync_time(plugin.id)
                    self.app.call_from_thread(
                        self._update_plugin_row_status,
                        plugin.id,
                        "",
                        "completed",
                    )

            except Exception as e:
                state.status = "error"
                state.message = str(e)
                self.app.call_from_thread(
                    self._update_plugin_row_status,
                    plugin.id,
                    str(e),
                    "error",
                )

        if not worker.is_cancelled:
            self.app.call_from_thread(self._sync_complete)

    def _update_global_progress(self, progress: float, message: str) -> None:
        """Update global progress bar and status (called from main thread)."""
        try:
            progress_bar = self.query_one("#global-progress", ProgressBar)
            progress_bar.update(progress=int(progress * 100))
            status = self.query_one("#global-status", Static)
            status.update(message)
        except Exception:
            pass

    def _update_plugin_row_status(self, plugin_id: str, message: str, status: str) -> None:
        """Update status for a specific plugin row (called from main thread)."""
        if plugin_id in self.plugin_states:
            self.plugin_states[plugin_id].status = status
            self.plugin_states[plugin_id].message = message

        try:
            status_widget = self.query_one(f"#status-{plugin_id}", Static)
            state = self.plugin_states.get(plugin_id)
            if state:
                # Get the status text from the state
                if status == "waiting":
                    text = ""
                elif status == "syncing":
                    text = f"⟳ {message}" if message else "⟳ Syncing..."
                elif status == "completed":
                    text = "✓ Done"
                elif status == "error":
                    text = f"✗ {message}" if message else "✗ Error"
                else:
                    text = ""
                status_widget.update(text)
        except Exception:
            pass

    def _update_global_status(self, message: str, is_error: bool = False) -> None:
        """Update global sync status."""
        try:
            self.query_one("#global-status", Static).update(message)
        except Exception:
            pass
        if is_error:
            self.notify(message, severity="error")

    def _sync_complete(self) -> None:
        """Handle sync completion."""
        self._syncing = False
        self.query_one("#sync-btn", Button).disabled = False
        self.query_one("#cancel-btn", Button).disabled = False
        self.query_one("#select-all-btn", Button).disabled = False
        self.query_one("#deselect-all-btn", Button).disabled = False

        # Update global progress to 100%
        self.query_one("#global-progress", ProgressBar).update(progress=100)

        # Re-enable checkboxes
        for plugin_id in self.plugin_states:
            try:
                checkbox = self.query_one(f"#sync-check-{plugin_id}", Checkbox)
                checkbox.disabled = False
            except Exception:
                pass

        # Count completed vs errors
        completed = sum(
            1 for s in self.plugin_states.values() if s.status == "completed" and s.selected
        )
        errors = sum(1 for s in self.plugin_states.values() if s.status == "error" and s.selected)

        if errors > 0:
            self.query_one("#global-status", Static).update(
                f"Completed with errors: {completed} synced, {errors} failed"
            )
            self.notify(
                f"Sync finished: {completed} completed, {errors} errors", severity="warning"
            )
        else:
            self.query_one("#global-status", Static).update(f"Sync complete! {completed} data source(s) synced.")
            self.notify(f"Sync completed for {completed} data source(s)!", severity="information")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker completion."""
        if event.state.name == "SUCCESS":
            self.app.pop_screen()
        elif event.state.name in ("CANCELLED", "ERROR"):
            self._syncing = False
            self.query_one("#sync-btn", Button).disabled = False
            self.query_one("#cancel-btn", Button).disabled = False
            self.query_one("#select-all-btn", Button).disabled = False
            self.query_one("#deselect-all-btn", Button).disabled = False

            if event.state.name == "CANCELLED":
                self.query_one("#global-status", Static).update("Sync cancelled")
            else:
                self.query_one("#global-status", Static).update("Sync failed")

            # Re-enable checkboxes
            for plugin_id in self.plugin_states:
                try:
                    checkbox = self.query_one(f"#sync-check-{plugin_id}", Checkbox)
                    checkbox.disabled = False
                except Exception:
                    pass

    def action_cancel(self) -> None:
        """Cancel and return to previous screen."""
        self.workers.cancel_all()
        self.app.pop_screen()
