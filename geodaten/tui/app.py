"""A compact Textual form backed by the shared download service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Log,
    ProgressBar,
    Select,
)

from geodaten.cli import parse_provider_options
from geodaten.common.models import BBox, DownloadRequest, OutputFormat, TimeRange
from geodaten.common.progress import CancellationToken, ProgressEvent
from geodaten.downloader.registry import get_provider, provider_specs
from geodaten.downloader.service import DownloadService


class GeodatenTui(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #form { height: 1fr; padding: 1 2; }
    .row { height: auto; margin-bottom: 1; }
    .field-label { width: 18; padding-top: 1; }
    .field { width: 1fr; }
    #credentials { height: auto; }
    #actions { height: 3; align-horizontal: right; }
    #actions Button { margin-left: 1; }
    #progress { margin: 0 2; }
    #log { height: 12; margin: 1 2; border: solid $primary; }
    """
    TITLE = "geodaten"

    def __init__(self) -> None:
        super().__init__()
        self.service = DownloadService()
        self.cancel_token = CancellationToken()

    def compose(self) -> ComposeResult:
        specs = provider_specs()
        yield Header()
        with VerticalScroll(id="form"):
            with Horizontal(classes="row"):
                yield Label("Dataset", classes="field-label")
                yield Select(
                    [(spec.title, spec.id) for spec in specs],
                    value=specs[0].id,
                    id="dataset",
                    classes="field",
                )
            yield from self._input_row("Variables", "variables", specs[0].variables[0].id)
            with Horizontal(classes="row"):
                yield Label("BBox W S E N", classes="field-label")
                yield Input("-180 -60 180 90", id="bbox", classes="field")
            yield from self._input_row("Start", "start", "earliest")
            yield from self._input_row("End", "end", "latest")
            with Horizontal(classes="row"):
                yield Label("Format", classes="field-label")
                yield Select(
                    [("GeoTIFF", "tif"), ("NetCDF", "netcdf")],
                    value="tif",
                    id="format",
                    classes="field",
                )
            yield from self._input_row("Output", "output", "geodaten-output")
            yield from self._input_row(
                "Provider workers", "workers", "4", placeholder="concurrency hint"
            )
            yield from self._input_row(
                "Provider options",
                "options",
                "",
                placeholder="key=value; key=value",
            )
            yield Label("Credentials are read from the environment/config first.")
            yield VerticalScroll(id="credentials")
            with Horizontal(classes="row"):
                yield Checkbox("Overwrite", id="overwrite")
                yield Checkbox("Allow partial", id="allow-partial")
            with Horizontal(id="actions"):
                yield Button("Plan", id="plan")
                yield Button("Download", variant="primary", id="download")
                yield Button("Cancel", variant="error", id="cancel", disabled=True)
        yield ProgressBar(total=100, id="progress")
        yield Log(id="log", auto_scroll=True)
        yield Footer()

    @staticmethod
    def _input_row(
        label: str,
        identifier: str,
        value: str,
        *,
        placeholder: str = "",
    ) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Label(label, classes="field-label")
            yield Input(value, placeholder=placeholder, id=identifier, classes="field")

    async def on_mount(self) -> None:
        await self._refresh_credentials(str(self.query_one("#dataset", Select).value))

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "dataset" or event.value is Select.BLANK:
            return
        spec = get_provider(str(event.value)).spec
        self.query_one("#variables", Input).value = spec.variables[0].id
        self.query_one("#start", Input).disabled = not spec.temporal
        self.query_one("#end", Input).disabled = not spec.temporal
        await self._refresh_credentials(spec.id)

    async def _refresh_credentials(self, dataset: str) -> None:
        container = self.query_one("#credentials", VerticalScroll)
        await container.remove_children()
        for field in get_provider(dataset).spec.credentials:
            row = Horizontal(classes="row")
            await container.mount(row)
            await row.mount(Label(field.title, classes="field-label"))
            await row.mount(
                Input(
                    "",
                    password=field.secret,
                    placeholder="optional when configured externally",
                    id=f"credential--{field.id}",
                    classes="field",
                )
            )

    def _request(self) -> DownloadRequest:
        dataset = str(self.query_one("#dataset", Select).value)
        spec = get_provider(dataset).spec
        bbox_values = self.query_one("#bbox", Input).value.replace(",", " ").split()
        variables = tuple(
            value.strip()
            for value in self.query_one("#variables", Input).value.split(",")
            if value.strip()
        )
        option_text = self.query_one("#options", Input).value.strip()
        options = parse_provider_options(
            [value.strip() for value in option_text.split(";") if value.strip()], spec.options
        )
        credentials: dict[str, str] = {}
        for field in spec.credentials:
            value = self.query_one(f"#credential--{field.id}", Input).value
            if value:
                credentials[field.id] = value
        temporal = (
            TimeRange(
                self.query_one("#start", Input).value,
                self.query_one("#end", Input).value,
            )
            if spec.temporal
            else None
        )
        return DownloadRequest(
            dataset=dataset,
            bbox=BBox.from_values(tuple(map(float, bbox_values))),
            time=temporal,
            variables=variables,
            output_format=OutputFormat(str(self.query_one("#format", Select).value)),
            output_dir=Path(self.query_one("#output", Input).value),
            options=options,
            credentials=credentials,
            workers=int(self.query_one("#workers", Input).value),
            overwrite=self.query_one("#overwrite", Checkbox).value,
            allow_partial=self.query_one("#allow-partial", Checkbox).value,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.cancel_token.cancel()
            return
        try:
            request = self._request()
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        if event.button.id == "plan":
            self._set_running(True, cancellable=False)
            self._plan_worker(request)
        elif event.button.id == "download":
            self.cancel_token = CancellationToken()
            self._set_running(True, cancellable=True)
            self._download_worker(request)

    def _set_running(self, running: bool, *, cancellable: bool = False) -> None:
        self.query_one("#plan", Button).disabled = running
        self.query_one("#download", Button).disabled = running
        self.query_one("#cancel", Button).disabled = not (running and cancellable)

    def _show_plan(self, payload: str) -> None:
        self.query_one("#log", Log).write_line(payload)

    def _show_manifest(self, path: Path) -> None:
        self.query_one("#log", Log).write_line(f"Manifest: {path}")

    def _show_event(self, event: ProgressEvent) -> None:
        self.query_one("#log", Log).write_line(f"[{event.stage}] {event.message}")
        if event.total and event.current is not None:
            self.query_one("#progress", ProgressBar).update(
                progress=int(event.current * 100 / event.total)
            )
        elif event.bytes_total and event.bytes_done is not None:
            self.query_one("#progress", ProgressBar).update(
                progress=int(event.bytes_done * 100 / event.bytes_total)
            )

    @work(thread=True, exclusive=True, group="operation")
    def _plan_worker(self, request: DownloadRequest) -> None:
        try:
            plan = self.service.plan(request)
            self.call_from_thread(self._show_plan, str(plan.public_dict()))
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
        finally:
            self.call_from_thread(self._set_running, False)

    @work(thread=True, exclusive=True, group="operation")
    def _download_worker(self, request: DownloadRequest) -> None:
        try:
            result = self.service.download(
                request,
                progress=lambda event: self.call_from_thread(self._show_event, event),
                cancel=self.cancel_token,
            )
            self.call_from_thread(self._show_manifest, result.manifest)
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
        finally:
            self.call_from_thread(self._set_running, False)


def main() -> int:
    GeodatenTui().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
