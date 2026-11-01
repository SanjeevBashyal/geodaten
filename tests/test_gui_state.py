from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest


#: Qt aborts if the single QApplication is garbage collected between windows.
_APPLICATION = None


def _window():
    """Build an offscreen main window, skipping when Qt is unavailable."""

    global _APPLICATION
    from qtpy.QtWidgets import QApplication

    from geodaten.gui.app import DownloadWindow

    _APPLICATION = QApplication.instance() or QApplication([])
    return DownloadWindow()


@pytest.fixture
def gui(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    state = tmp_path / "config" / "geodaten.json"
    monkeypatch.setenv("GEODATEN_STATE", str(state))
    try:
        import qtpy.QtWidgets  # noqa: F401
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Qt GUI runtime is unavailable: {exc}")
    return state


def test_state_path_prefers_the_user_config_directory(monkeypatch, tmp_path: Path) -> None:
    from geodaten.gui.app import state_path

    monkeypatch.delenv("GEODATEN_STATE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert state_path() == tmp_path / "geodaten" / "geodaten.json"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    # Never the working directory: the file holds credentials in the clear.
    assert state_path() == Path.home() / ".config" / "geodaten" / "geodaten.json"
    monkeypatch.setenv("GEODATEN_STATE", str(tmp_path / "elsewhere.json"))
    assert state_path() == tmp_path / "elsewhere.json"


def test_inputs_including_credentials_survive_a_restart(gui: Path) -> None:
    window = _window()
    try:
        window.comboBox_datasetGroup.setCurrentText("Climate")
        index = window.datasetCombo.findData("era5_land")
        window.datasetCombo.setCurrentIndex(index)
        window.variable_inputs["t2m"].setChecked(True)
        window.credential_inputs["cds_api_key"].setText("secret-token")
        window.provider_options["era5_land"] = {"hours": (0, 12)}
        window.lineEdit_longitudeExtent.setText("-10, 20")
        window.lineEdit_latitudeExtent.setText("40, 60")
        window.outputEdit.setText("/tmp/geodaten")
        window.overwriteCheck.setChecked(True)
        window.checkBox_earliest.setChecked(False)
        window.dateEdit_startDate.setDate(window.dateEdit_startDate.date().addYears(-3))
        expected_start = window.dateEdit_startDate.date().toString("yyyy-MM-dd")
        assert window.save_state() == gui
    finally:
        window.close()

    restored = _window()
    try:
        assert restored.comboBox_datasetGroup.currentText() == "Climate"
        assert restored.datasetCombo.currentData() == "era5_land"
        assert restored.variable_inputs["t2m"].isChecked()
        assert restored.credential_inputs["cds_api_key"].text() == "secret-token"
        assert restored.provider_options["era5_land"] == {"hours": [0, 12]}
        assert restored.lineEdit_longitudeExtent.text() == "-10, 20"
        assert restored.lineEdit_latitudeExtent.text() == "40, 60"
        assert restored.outputEdit.text() == "/tmp/geodaten"
        assert restored.overwriteCheck.isChecked()
        assert not restored.partialCheck.isChecked()
        assert not restored.checkBox_earliest.isChecked()
        assert restored.dateEdit_startDate.date().toString("yyyy-MM-dd") == expected_start
        assert restored.checkBox_latest.isChecked()
        assert restored._request().credentials == {"cds_api_key": "secret-token"}
    finally:
        restored.close()


def test_the_saved_file_is_not_readable_by_anyone_else(gui: Path) -> None:
    window = _window()
    try:
        window.credential_inputs and None
        window.save_state()
    finally:
        window.close()
    assert stat.S_IMODE(gui.stat().st_mode) == 0o600
    assert "credentials" in json.loads(gui.read_text(encoding="utf-8"))


def test_a_pre_existing_loose_file_is_tightened_on_the_next_save(gui: Path) -> None:
    gui.parent.mkdir(parents=True, exist_ok=True)
    gui.write_text("{}", encoding="utf-8")
    gui.chmod(0o644)
    window = _window()
    try:
        window.save_state()
    finally:
        window.close()
    assert stat.S_IMODE(gui.stat().st_mode) == 0o600


def test_per_dataset_selections_do_not_bleed_between_datasets(gui: Path) -> None:
    window = _window()
    try:
        window.comboBox_datasetGroup.setCurrentText("Terrain")
        window.variable_inputs["elevation"].setChecked(True)
        window.comboBox_datasetGroup.setCurrentText("Climate")
        window.datasetCombo.setCurrentIndex(window.datasetCombo.findData("era5_land"))
        window.credential_inputs["cds_api_key"].setText("other")
        window.save_state()
    finally:
        window.close()

    restored = _window()
    try:
        restored.comboBox_datasetGroup.setCurrentText("Terrain")
        assert restored.variable_inputs["elevation"].isChecked()
        assert restored.credential_inputs == {}
    finally:
        restored.close()


@pytest.mark.parametrize("content", ["not json at all", "[]", '{"group": 12, "variables": 7}'])
def test_an_unusable_state_file_never_blocks_the_gui(gui: Path, content: str) -> None:
    gui.parent.mkdir(parents=True, exist_ok=True)
    gui.write_text(content, encoding="utf-8")
    window = _window()
    try:
        # Falls back to the ordinary first-run defaults instead of raising.
        assert window.datasetCombo.currentData() is not None
        assert window.variable_inputs
    finally:
        window.close()


def test_an_unwritable_location_is_reported_without_raising(gui: Path, monkeypatch) -> None:
    import geodaten.gui.app as app_module

    monkeypatch.setattr(app_module, "state_path", lambda: gui.parent / "x" / "y.json")
    window = _window()
    try:
        monkeypatch.setattr(
            app_module.os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("denied"))
        )
        assert window.save_state() is None
        assert "Could not save" in window.logEdit.toPlainText()
    finally:
        window.close()
