from __future__ import annotations

import asyncio
from importlib import resources
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from geodaten.cli import build_parser
from geodaten.common.models import CredentialField, DatasetSpec, VariableSpec


def test_cli_exposes_all_interfaces() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in ("datasets", "info", "plan", "download", "doctor", "tui", "gui"):
        assert command in help_text


def test_qt_designer_form_is_packaged_and_contains_download_inputs() -> None:
    form = resources.files("geodaten.gui").joinpath("download.ui")
    root = ElementTree.fromstring(form.read_text(encoding="utf-8"))
    names = {element.attrib.get("name") for element in root.iter("widget")}
    assert {
        "comboBox_datasetGroup",
        "datasetCombo",
        "variableHostWidget",
        "lineEdit_longitudeExtent",
        "lineEdit_latitudeExtent",
        "dateEdit_startDate",
        "checkBox_earliest",
        "dateEdit_endDate",
        "checkBox_latest",
        "formatCombo",
        "outputEdit",
        "downloadButton",
    } <= names
    assert "workersSpin" not in names
    variable_host = next(
        element
        for element in root.iter("widget")
        if element.attrib.get("name") == "variableHostWidget"
    )
    variable_layout = variable_host.find("./layout")
    assert variable_layout is not None
    assert variable_layout.attrib == {"class": "QGridLayout", "name": "variableLayout"}


def test_dataset_specs_match_the_designer_groups() -> None:
    from geodaten.downloader.registry import provider_specs

    groups = {spec.id: spec.group for spec in provider_specs()}
    assert groups == {
        "soilgrids": "Soil",
        "srtm": "Terrain",
        "glc_fcs30d": "Land Use",
        "glim": "Geology",
        "gimms_lai4g": "LAI",
        "modis_lai": "LAI",
        "copernicus_lai": "LAI",
        "nex_gddp_cmip6": "Climate",
        "ifs_fesom": "Climate",
        "era5_land": "Climate",
        "sentinel2": "Imagery",
        "landsat9": "Imagery",
    }


def test_every_dataset_group_exists_in_the_designer_form() -> None:
    from geodaten.downloader.registry import provider_specs

    form = resources.files("geodaten.gui").joinpath("download.ui")
    root = ElementTree.fromstring(form.read_text(encoding="utf-8"))
    combo = next(
        element
        for element in root.iter("widget")
        if element.attrib.get("name") == "comboBox_datasetGroup"
    )
    designer_groups = {item.findtext("./property/string") for item in combo.iter("item")}
    assert {spec.group for spec in provider_specs()} <= designer_groups


def test_qt_controller_filters_datasets_and_builds_requests(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    try:
        from qtpy.QtWidgets import QApplication

        from geodaten.gui.app import DownloadWindow, ProviderOptionsDialog
        from geodaten.downloader.registry import get_provider
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Qt GUI runtime is unavailable: {exc}")

    application = QApplication.instance() or QApplication([])
    window = DownloadWindow()
    try:
        assert window.comboBox_datasetGroup.currentText() == "Terrain"
        assert window.datasetCombo.currentData() == "srtm"
        assert tuple(window.variable_inputs) == ("elevation",)
        assert window.variable_inputs["elevation"].text() == "Land surface elevation"
        window.variable_inputs["elevation"].setChecked(True)
        window.lineEdit_longitudeExtent.setText("-10, 20")
        window.lineEdit_latitudeExtent.setText("40, 60")
        request = window._request()
        assert request.bbox.as_tuple() == (-10.0, 40.0, 20.0, 60.0)
        assert request.time is None
        assert request.variables == ("elevation",)
        assert request.workers == 4

        window.comboBox_datasetGroup.setCurrentText("Climate")
        assert tuple(
            window.datasetCombo.itemData(index)
            for index in range(window.datasetCombo.count())
        ) == ("nex_gddp_cmip6", "ifs_fesom", "era5_land")
        assert window.groupBox_temporalExtent.isEnabled()
        assert window.dateEdit_startDate.text() == "Earliest"
        assert window.dateEdit_endDate.text() == "Latest"
        window.variable_inputs["hurs"].setChecked(True)
        window.checkBox_earliest.setChecked(False)
        window.checkBox_latest.setChecked(False)
        window.dateEdit_startDate.setDate(window.dateEdit_startDate.date().addYears(-1))
        climate_request = window._request()
        assert climate_request.time is not None
        assert climate_request.time.start == window.dateEdit_startDate.date().toString(
            "yyyy-MM-dd"
        )
        assert climate_request.time.end == window.dateEdit_endDate.date().toString("yyyy-MM-dd")

        options = ProviderOptionsDialog(get_provider("nex_gddp_cmip6").spec, {})
        assert options._read() == {}
        options._controls["horizontal_stride"].setText("2")
        assert options._read() == {"horizontal_stride": 2}
        options.close()

        window.comboBox_datasetGroup.setCurrentText("Geology")
        assert window.datasetCombo.currentData() == "glim"
        assert not window.groupBox_temporalExtent.isEnabled()
        assert tuple(
            window.formatCombo.itemData(index)
            for index in range(window.formatCombo.count())
        ) == ("shp",)
        window.variable_inputs["lithology"].setChecked(True)
        vector_request = window._request()
        assert vector_request.output_format.value == "shp"
        assert vector_request.time is None

        window.comboBox_datasetGroup.setCurrentText("Soil")
        assert tuple(
            window.formatCombo.itemData(index)
            for index in range(window.formatCombo.count())
        ) == ("tif", "netcdf")
    finally:
        window.close()
        application.processEvents()


def test_textual_form_mounts_dynamic_masked_credentials(monkeypatch) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    import geodaten.tui.app as tui_module

    spec = DatasetSpec(
        id="example",
        title="Example",
        summary="Example dataset",
        variables=(VariableSpec("value", "Value"),),
        credentials=(CredentialField("token", "Token"),),
    )
    provider = SimpleNamespace(spec=spec)
    monkeypatch.setattr(tui_module, "provider_specs", lambda: (spec,))
    monkeypatch.setattr(tui_module, "get_provider", lambda _identifier: provider)

    async def run() -> None:
        application = tui_module.GeodatenTui()
        async with application.run_test(size=(100, 50)):
            assert application.query_one("#dataset", Select).value == "example"
            assert application.query_one("#credential--token", Input).password

    asyncio.run(run())
