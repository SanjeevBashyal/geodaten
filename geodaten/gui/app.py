"""QtPy controller for the packaged Qt Designer form."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from qtpy import uic
from qtpy.QtCore import QDate, QObject, QThread, Signal, Slot
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QWidget,
)

from geodaten.cli import parse_provider_options
from geodaten.common.models import (
    BBox,
    DatasetSpec,
    DownloadRequest,
    OptionSpec,
    OutputFormat,
    TimeRange,
)
from geodaten.common.progress import CancellationToken, ProgressEvent
from geodaten.downloader.registry import get_provider, provider_specs
from geodaten.downloader.service import DownloadService

_ALIAS_DATE = QDate(1900, 1, 1)
_VARIABLE_COLUMNS = 3
_FORMAT_TITLES = {"tif": "GeoTIFF", "netcdf": "NetCDF", "shp": "ESRI Shapefile"}


def state_path() -> Path:
    """Return the saved-inputs file, overridable with ``GEODATEN_STATE``.

    It lives in the user configuration directory rather than the working
    directory because it holds credentials in the clear and must not be picked
    up by a repository the user happens to launch the GUI from.
    """

    override = os.environ.get("GEODATEN_STATE")
    if override:
        return Path(override).expanduser()
    root = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(root).expanduser() / "geodaten" / "geodaten.json"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clear_layout(layout: Any) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.layout() is not None:
            _clear_layout(item.layout())
        if item.widget() is not None:
            item.widget().deleteLater()


def _parse_extent(text: str, name: str) -> tuple[float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"{name} extent requires two values: minimum, maximum")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"{name} extent values must be numbers") from exc


def _tooltip(option: OptionSpec) -> str:
    details = [option.description]
    if option.default is not None:
        details.append(f"Default: {option.default}")
    return "\n".join(value for value in details if value)


class ProviderOptionsDialog(QDialog):
    """Small metadata-driven editor that keeps provider controls out of the main form."""

    def __init__(
        self,
        spec: DatasetSpec,
        values: Mapping[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{spec.title} options")
        self._spec = spec
        self._values: dict[str, Any] | None = None
        self._controls: dict[str, Any] = {}
        form = QFormLayout(self)

        for option in spec.options:
            value = values.get(option.id, option.default)
            control: Any
            if option.choices and option.multiple:
                host = QWidget(self)
                grid = QGridLayout(host)
                grid.setContentsMargins(0, 0, 0, 0)
                selected = set(value or ())
                boxes: list[tuple[QCheckBox, Any]] = []
                for index, choice in enumerate(option.choices):
                    box = QCheckBox(str(choice), host)
                    box.setChecked(choice in selected)
                    grid.addWidget(box, index // _VARIABLE_COLUMNS, index % _VARIABLE_COLUMNS)
                    boxes.append((box, choice))
                control = boxes
                form.addRow(option.title, host)
            elif option.choices:
                combo = QComboBox(self)
                for choice in option.choices:
                    combo.addItem(str(choice), choice)
                selected_index = combo.findData(value)
                if selected_index >= 0:
                    combo.setCurrentIndex(selected_index)
                control = combo
                form.addRow(option.title, combo)
            elif option.kind == "bool":
                check = QCheckBox(self)
                check.setChecked(bool(value))
                control = check
                form.addRow(option.title, check)
            else:
                edit = QLineEdit(self)
                if option.multiple:
                    edit.setText(", ".join(map(str, value or ())))
                elif value is not None:
                    edit.setText(str(value))
                control = edit
                form.addRow(option.title, edit)
            control_widget = control[0][0].parentWidget() if isinstance(control, list) and control else control
            if isinstance(control_widget, QWidget):
                control_widget.setToolTip(_tooltip(option))
            self._controls[option.id] = control

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _read(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for option in self._spec.options:
            control = self._controls[option.id]
            if option.choices and option.multiple:
                value: Any = tuple(choice for box, choice in control if box.isChecked())
                if not value and option.default is None:
                    value = None
            elif option.choices:
                value = control.currentData()
            elif option.kind == "bool":
                value = control.isChecked()
            else:
                raw = control.text().strip()
                if not raw:
                    value = option.default
                else:
                    value = parse_provider_options(
                        [f"{option.id}={raw}"], (option,)
                    )[option.id]
            if option.required and (value is None or value == "" or value == ()):
                raise ValueError(f"{option.title} is required")
            if value is not None and value != option.default:
                values[option.id] = value
        return values

    def _accept(self) -> None:
        try:
            self._values = self._read()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid options", str(exc))
            return
        self.accept()

    @property
    def values(self) -> dict[str, Any]:
        return dict(self._values or {})


class ServiceWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, request: DownloadRequest, mode: str, token: CancellationToken) -> None:
        super().__init__()
        self.request = request
        self.mode = mode
        self.token = token

    @Slot()
    def run(self) -> None:
        try:
            service = DownloadService()
            if self.mode == "plan":
                result: Any = service.plan(self.request)
            else:
                result = service.download(
                    self.request,
                    progress=self.progress.emit,
                    cancel=self.token,
                )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class DownloadWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        uic.loadUi(str(Path(__file__).with_name("download.ui")), self)
        self._specs = provider_specs()
        self.variable_inputs: dict[str, QCheckBox] = {}
        self.credential_inputs: dict[str, QLineEdit] = {}
        self.provider_options: dict[str, dict[str, Any]] = {}
        # Per-dataset selections survive switching datasets and restarts.
        self.saved_variables: dict[str, list[str]] = {}
        self.saved_credentials: dict[str, dict[str, str]] = {}
        self._current_dataset: str | None = None
        self.thread: QThread | None = None
        self.worker: ServiceWorker | None = None
        self.token = CancellationToken()
        self._close_after_work = False
        self._running = False
        self._cancellable = False
        today = QDate.currentDate()
        self._concrete_dates = {"start": today.addYears(-1), "end": today}

        self.optionsButton = QPushButton("Options…", self)
        self.horizontalLayout_dataset.addWidget(self.optionsButton)
        self.variableLayout.setContentsMargins(0, 0, 0, 0)
        self.variableLayout.setSpacing(2)
        self.formatCombo.clear()
        for edit, value in (
            (self.dateEdit_startDate, self._concrete_dates["start"]),
            (self.dateEdit_endDate, self._concrete_dates["end"]),
        ):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd")
            edit.setMinimumDate(_ALIAS_DATE)
            edit.setDate(value)

        self.comboBox_datasetGroup.currentIndexChanged.connect(self._group_changed)
        self.datasetCombo.currentIndexChanged.connect(self._dataset_changed)
        self.checkBox_earliest.toggled.connect(
            lambda checked: self._toggle_date_alias("start", checked)
        )
        self.checkBox_latest.toggled.connect(
            lambda checked: self._toggle_date_alias("end", checked)
        )
        self.optionsButton.clicked.connect(self._edit_options)
        self.browseButton.clicked.connect(self._browse)
        self.planButton.clicked.connect(lambda: self._start("plan"))
        self.downloadButton.clicked.connect(lambda: self._start("download"))
        self.cancelButton.clicked.connect(self._cancel)

        self.checkBox_earliest.setChecked(True)
        self.checkBox_latest.setChecked(True)
        if self.comboBox_datasetGroup.currentIndex() < 0 and self.comboBox_datasetGroup.count():
            self.comboBox_datasetGroup.setCurrentIndex(0)
        else:
            self._group_changed()
        self.load_state()

    def _group_changed(self) -> None:
        group = self.comboBox_datasetGroup.currentText()
        matches = tuple(spec for spec in self._specs if spec.group == group)
        blocked = self.datasetCombo.blockSignals(True)
        self.datasetCombo.clear()
        for spec in matches:
            self.datasetCombo.addItem(spec.title, spec.id)
        self.datasetCombo.setEnabled(bool(matches))
        self.datasetCombo.blockSignals(blocked)
        self._dataset_changed()

    def _dataset_changed(self) -> None:
        # The outgoing dataset's widgets are about to be destroyed, so remember
        # what was chosen in them before they go.
        if self._current_dataset is not None:
            self._capture_dataset_state(self._current_dataset)
        _clear_layout(self.variableLayout)
        _clear_layout(self.credentialLayout)
        self.variable_inputs.clear()
        self.credential_inputs.clear()
        identifier = self.datasetCombo.currentData()
        self._current_dataset = None if identifier is None else str(identifier)
        if identifier is None:
            self.variableLayout.addWidget(QLabel("No datasets are available in this group"), 0, 0)
            self.credentialLayout.addRow(QLabel("Select an available dataset"))
            self.groupBox_temporalExtent.setEnabled(False)
            self.optionsButton.setEnabled(False)
            self.formatCombo.clear()
            self._set_running(self._running, cancellable=self._cancellable)
            return

        spec = get_provider(str(identifier)).spec
        self._refresh_formats(spec)
        remembered = set(self.saved_variables.get(spec.id, ()))
        for index, variable in enumerate(spec.variables):
            box = QCheckBox(variable.title, self.variableHostWidget)
            box.setChecked(variable.id in remembered)
            details = [f"Identifier: {variable.id}"]
            if variable.units:
                details.append(f"Units: {variable.units}")
            if variable.description:
                details.append(variable.description)
            box.setToolTip("\n".join(details))
            self.variableLayout.addWidget(
                box, index // _VARIABLE_COLUMNS, index % _VARIABLE_COLUMNS
            )
            self.variable_inputs[variable.id] = box

        self.groupBox_temporalExtent.setEnabled(spec.temporal)
        if spec.temporal:
            self._toggle_date_alias("start", self.checkBox_earliest.isChecked())
            self._toggle_date_alias("end", self.checkBox_latest.isChecked())
        self.optionsButton.setEnabled(bool(spec.options) and not self._running)
        self.optionsButton.setText(
            f"Options… ({len(self.provider_options.get(spec.id, {}))})"
            if self.provider_options.get(spec.id)
            else "Options…"
        )
        if not spec.credentials:
            self.credentialLayout.addRow(QLabel("Public dataset; no credentials required"))
        secrets = self.saved_credentials.get(spec.id, {})
        for field in spec.credentials:
            edit = QLineEdit(self.credentialWidget)
            edit.setPlaceholderText("Uses environment/config when left empty")
            edit.setToolTip(field.description)
            edit.setText(str(secrets.get(field.id, "")))
            if field.secret:
                edit.setEchoMode(QLineEdit.Password)
            self.credentialLayout.addRow(field.title, edit)
            self.credential_inputs[field.id] = edit
        self._set_running(self._running, cancellable=self._cancellable)

    def _effective_date(self, name: str) -> QDate:
        """Return the real date behind an Earliest/Latest alias."""

        alias = self.checkBox_earliest if name == "start" else self.checkBox_latest
        edit = self.dateEdit_startDate if name == "start" else self.dateEdit_endDate
        return self._concrete_dates[name] if alias.isChecked() else edit.date()

    def _capture_dataset_state(self, dataset: str | None = None) -> None:
        """Fold the visible widgets back into the per-dataset stores."""

        if dataset is None:
            identifier = self.datasetCombo.currentData()
            if identifier is None:
                return
            dataset = str(identifier)
        self.saved_variables[dataset] = [
            name for name, box in self.variable_inputs.items() if box.isChecked()
        ]
        secrets = {name: edit.text() for name, edit in self.credential_inputs.items() if edit.text()}
        if secrets:
            self.saved_credentials[dataset] = secrets
        else:
            self.saved_credentials.pop(dataset, None)

    def save_state(self) -> Path | None:
        """Write the current inputs so the next session starts where this one ended."""

        self._capture_dataset_state()
        state = {
            "group": self.comboBox_datasetGroup.currentText(),
            "dataset": self.datasetCombo.currentData(),
            "longitude": self.lineEdit_longitudeExtent.text(),
            "latitude": self.lineEdit_latitudeExtent.text(),
            "earliest": self.checkBox_earliest.isChecked(),
            "latest": self.checkBox_latest.isChecked(),
            "start": self._effective_date("start").toString("yyyy-MM-dd"),
            "end": self._effective_date("end").toString("yyyy-MM-dd"),
            "format": self.formatCombo.currentData(),
            "output_dir": self.outputEdit.text(),
            "overwrite": self.overwriteCheck.isChecked(),
            "allow_partial": self.partialCheck.isChecked(),
            "variables": self.saved_variables,
            "options": self.provider_options,
            "credentials": self.saved_credentials,
        }
        path = state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Credentials are stored in the clear, so the file must never be
            # group- or world-readable, including when it already existed.
            handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(handle, "w", encoding="utf-8") as target:
                json.dump(state, target, indent=2, sort_keys=True)
            os.chmod(path, 0o600)
        except OSError as exc:
            # Losing saved inputs must never take the download down with it.
            self.logEdit.appendPlainText(f"Could not save {path}: {exc}")
            return None
        return path

    def load_state(self) -> None:
        """Restore the previous session's inputs, ignoring an unusable file."""

        try:
            state = json.loads(state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(state, Mapping):
            return

        # Every nested container is checked: a hand-edited file must degrade to
        # first-run defaults rather than stop the GUI from opening.
        self.saved_variables = {
            str(key): [str(name) for name in value]
            for key, value in _mapping(state.get("variables")).items()
            if isinstance(value, (list, tuple))
        }
        self.saved_credentials = {
            str(key): {str(name): str(secret) for name, secret in value.items()}
            for key, value in _mapping(state.get("credentials")).items()
            if isinstance(value, Mapping)
        }
        self.provider_options = {
            str(key): dict(value)
            for key, value in _mapping(state.get("options")).items()
            if isinstance(value, Mapping)
        }
        for name in ("start", "end"):
            saved = QDate.fromString(str(state.get(name, "")), "yyyy-MM-dd")
            if saved.isValid():
                self._concrete_dates[name] = saved

        # Nothing on screen was chosen by the user yet, so restoring must not
        # capture the default widgets over the values just read from the file.
        self._current_dataset = None
        group = str(state.get("group") or "")
        index = self.comboBox_datasetGroup.findText(group)
        if index >= 0:
            self.comboBox_datasetGroup.setCurrentIndex(index)
        dataset = state.get("dataset")
        if dataset is not None:
            index = self.datasetCombo.findData(str(dataset))
            if index >= 0:
                self.datasetCombo.setCurrentIndex(index)
        # The dataset drives which variable and credential widgets exist, so
        # rebuild once more in case neither index actually changed.
        self._dataset_changed()

        output_format = state.get("format")
        if output_format is not None:
            index = self.formatCombo.findData(str(output_format))
            if index >= 0:
                self.formatCombo.setCurrentIndex(index)
        self.lineEdit_longitudeExtent.setText(str(state.get("longitude") or ""))
        self.lineEdit_latitudeExtent.setText(str(state.get("latitude") or ""))
        self.outputEdit.setText(str(state.get("output_dir") or ""))
        self.overwriteCheck.setChecked(bool(state.get("overwrite")))
        self.partialCheck.setChecked(bool(state.get("allow_partial")))
        self.checkBox_earliest.setChecked(bool(state.get("earliest", True)))
        self.checkBox_latest.setChecked(bool(state.get("latest", True)))

    def _refresh_formats(self, spec: DatasetSpec) -> None:
        """Offer only the output families the selected dataset can produce."""

        previous = self.formatCombo.currentData()
        blocked = self.formatCombo.blockSignals(True)
        self.formatCombo.clear()
        for value in spec.format_ids:
            self.formatCombo.addItem(_FORMAT_TITLES.get(value, value), value)
        restored = self.formatCombo.findData(previous)
        self.formatCombo.setCurrentIndex(max(restored, 0))
        self.formatCombo.blockSignals(blocked)

    def _toggle_date_alias(self, name: str, checked: bool) -> None:
        edit = self.dateEdit_startDate if name == "start" else self.dateEdit_endDate
        label = "Earliest" if name == "start" else "Latest"
        if checked:
            if edit.date() != _ALIAS_DATE:
                self._concrete_dates[name] = edit.date()
            edit.setSpecialValueText(label)
            edit.setDate(_ALIAS_DATE)
        else:
            edit.setSpecialValueText("")
            edit.setDate(self._concrete_dates[name])
        edit.setEnabled(not checked)

    def _edit_options(self) -> None:
        identifier = self.datasetCombo.currentData()
        if identifier is None:
            return
        spec = get_provider(str(identifier)).spec
        dialog = ProviderOptionsDialog(
            spec, self.provider_options.get(spec.id, {}), self
        )
        if dialog.exec() == QDialog.Accepted:
            self.provider_options[spec.id] = dialog.values
            self.optionsButton.setText(
                f"Options… ({len(dialog.values)})" if dialog.values else "Options…"
            )

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select output directory")
        if selected:
            self.outputEdit.setText(selected)

    def _request(self) -> DownloadRequest:
        identifier = self.datasetCombo.currentData()
        if identifier is None:
            raise ValueError("Select a dataset")
        spec = get_provider(str(identifier)).spec
        west, east = _parse_extent(self.lineEdit_longitudeExtent.text(), "Longitude")
        south, north = _parse_extent(self.lineEdit_latitudeExtent.text(), "Latitude")
        variables = tuple(
            variable.id
            for variable in spec.variables
            if self.variable_inputs[variable.id].isChecked()
        )
        credentials = {
            name: edit.text() for name, edit in self.credential_inputs.items() if edit.text()
        }
        output_dir = self.outputEdit.text().strip()
        if not output_dir:
            raise ValueError("Select an output directory")
        time = None
        if spec.temporal:
            start = (
                "earliest"
                if self.checkBox_earliest.isChecked()
                else self.dateEdit_startDate.date().toString("yyyy-MM-dd")
            )
            end = (
                "latest"
                if self.checkBox_latest.isChecked()
                else self.dateEdit_endDate.date().toString("yyyy-MM-dd")
            )
            time = TimeRange(start, end)
        return DownloadRequest(
            dataset=spec.id,
            bbox=BBox(west, south, east, north),
            time=time,
            variables=variables,
            output_format=OutputFormat(str(self.formatCombo.currentData())),
            output_dir=Path(output_dir),
            options=self.provider_options.get(spec.id, {}),
            credentials=credentials,
            overwrite=self.overwriteCheck.isChecked(),
            allow_partial=self.partialCheck.isChecked(),
        )

    def _set_running(self, running: bool, *, cancellable: bool = False) -> None:
        self._running = running
        self._cancellable = cancellable
        ready = self.datasetCombo.currentData() is not None
        self.planButton.setEnabled(not running and ready)
        self.downloadButton.setEnabled(not running and ready)
        self.optionsButton.setEnabled(
            not running
            and ready
            and bool(get_provider(str(self.datasetCombo.currentData())).spec.options)
        )
        self.cancelButton.setEnabled(running and cancellable)

    def _start(self, mode: str) -> None:
        try:
            request = self._request()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid request", str(exc))
            return
        self.save_state()
        self.token = CancellationToken()
        self.thread = QThread(self)
        self.worker = ServiceWorker(request, mode, self.token)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self._set_running(True, cancellable=mode == "download")
        self.thread.start()

    @Slot(object)
    def _progress(self, event: ProgressEvent) -> None:
        self.logEdit.appendPlainText(f"[{event.stage}] {event.message}")
        if event.total and event.current is not None:
            self.progressBar.setValue(int(event.current * 100 / event.total))
        elif event.bytes_total and event.bytes_done is not None:
            self.progressBar.setValue(int(event.bytes_done * 100 / event.bytes_total))

    @Slot(object)
    def _finished(self, result: Any) -> None:
        if hasattr(result, "public_dict"):
            text = json.dumps(result.public_dict(), indent=2, default=str)
        else:
            text = f"Manifest: {result.manifest}\n" + "\n".join(map(str, result.outputs))
        self.logEdit.appendPlainText(text)
        self.progressBar.setValue(100)

    @Slot(str)
    def _failed(self, message: str) -> None:
        self.logEdit.appendPlainText(f"ERROR: {message}")
        if not self._close_after_work:
            QMessageBox.critical(self, "Download failed", message)

    @Slot()
    def _thread_finished(self) -> None:
        close_after_work = self._close_after_work
        self.worker = None
        self.thread = None
        self._set_running(False)
        if close_after_work:
            self._close_after_work = False
            self.close()

    def _cancel(self) -> None:
        self.token.cancel()
        self.logEdit.appendPlainText("Cancellation requested")

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name
        self.save_state()
        if self.thread is not None and self.thread.isRunning():
            self._close_after_work = True
            self.token.cancel()
            self.logEdit.appendPlainText("Closing after the active operation stops")
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)


def main() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    window = DownloadWindow()
    window.show()
    return int(application.exec())


if __name__ == "__main__":
    raise SystemExit(main())
