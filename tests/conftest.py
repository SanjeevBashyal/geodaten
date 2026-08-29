from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_gui_state(monkeypatch, tmp_path):
    """Keep every test away from the developer's real saved-inputs file.

    The GUI writes its inputs on close, so without this a test run would both
    overwrite ``~/.config/geodaten/geodaten.json`` and read it back, making the
    suite depend on whatever was last done in the real application.
    """

    monkeypatch.setenv("GEODATEN_STATE", str(tmp_path / "gui-state" / "geodaten.json"))
