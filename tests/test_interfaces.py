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
        "datasetCombo",
        "variablesEdit",
        "bboxEdit",
        "startEdit",
        "endEdit",
        "formatCombo",
        "outputEdit",
        "downloadButton",
    } <= names
    workers = next(
        element for element in root.iter("widget") if element.attrib.get("name") == "workersSpin"
    )
    tooltip = workers.find("./property[@name='toolTip']/string")
    assert tooltip is not None and "hint" in tooltip.text.lower()


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
