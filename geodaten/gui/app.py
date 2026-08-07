"""QtPy controller for the packaged Qt Designer form."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from qtpy import uic
from qtpy.QtCore import QObject, QThread, Signal, Slot
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
)

from geodaten.cli import parse_provider_options
from geodaten.common.models import BBox, DownloadRequest, OutputFormat, TimeRange
from geodaten.common.progress import CancellationToken, ProgressEvent
from geodaten.downloader.registry import get_provider, provider_specs
from geodaten.downloader.service import DownloadService


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
        self.credential_inputs: dict[str, QLineEdit] = {}
        self.thread: QThread | None = None
        self.worker: ServiceWorker | None = None
        self.token = CancellationToken()
        self._close_after_work = False

        # Some QtPy/uic combinations do not preserve Designer's item userData.
        # Set the stable values explicitly while keeping the labels in the .ui.
        self.formatCombo.clear()
        self.formatCombo.addItem("GeoTIFF", "tif")
        self.formatCombo.addItem("NetCDF", "netcdf")
        for spec in provider_specs():
            self.datasetCombo.addItem(spec.title, spec.id)
        self.datasetCombo.currentIndexChanged.connect(self._dataset_changed)
        self.browseButton.clicked.connect(self._browse)
        self.planButton.clicked.connect(lambda: self._start("plan"))
        self.downloadButton.clicked.connect(lambda: self._start("download"))
        self.cancelButton.clicked.connect(self._cancel)
        self._dataset_changed()

    def _dataset_changed(self) -> None:
        identifier = self.datasetCombo.currentData()
        if identifier is None:
            return
        spec = get_provider(str(identifier)).spec
        self.variablesEdit.setText(spec.variables[0].id)
        self.startEdit.setEnabled(spec.temporal)
        self.endEdit.setEnabled(spec.temporal)
        while self.credentialLayout.count():
            item = self.credentialLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.credential_inputs.clear()
        if not spec.credentials:
            self.credentialLayout.addRow(QLabel("Public dataset; no credentials required"))
        for field in spec.credentials:
            edit = QLineEdit()
            edit.setPlaceholderText("Uses environment/config when left empty")
            if field.secret:
                edit.setEchoMode(QLineEdit.Password)
            self.credentialLayout.addRow(field.title, edit)
            self.credential_inputs[field.id] = edit

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select output directory")
        if selected:
            self.outputEdit.setText(selected)

    def _request(self) -> DownloadRequest:
        identifier = str(self.datasetCombo.currentData())
        spec = get_provider(identifier).spec
        bbox = tuple(map(float, self.bboxEdit.text().replace(",", " ").split()))
        variables = tuple(
            value.strip() for value in self.variablesEdit.text().split(",") if value.strip()
        )
        raw_options = [
            value.strip() for value in self.optionsEdit.text().split(";") if value.strip()
        ]
        options = parse_provider_options(raw_options, spec.options)
        credentials = {
            name: edit.text() for name, edit in self.credential_inputs.items() if edit.text()
        }
        return DownloadRequest(
            dataset=identifier,
            bbox=BBox.from_values(bbox),
            time=TimeRange(self.startEdit.text(), self.endEdit.text()) if spec.temporal else None,
            variables=variables,
            output_format=OutputFormat(str(self.formatCombo.currentData())),
            output_dir=Path(self.outputEdit.text()),
            options=options,
            credentials=credentials,
            workers=self.workersSpin.value(),
            overwrite=self.overwriteCheck.isChecked(),
            allow_partial=self.partialCheck.isChecked(),
        )

    def _set_running(self, running: bool, *, cancellable: bool = False) -> None:
        self.planButton.setEnabled(not running)
        self.downloadButton.setEnabled(not running)
        self.cancelButton.setEnabled(running and cancellable)

    def _start(self, mode: str) -> None:
        try:
            request = self._request()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid request", str(exc))
            return
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
