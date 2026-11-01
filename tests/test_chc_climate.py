from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from geodaten.common.errors import ConfigurationError
from geodaten.common.models import BBox, DownloadRequest, OutputFormat, TimeRange
from geodaten.common.progress import CancellationToken
from geodaten.downloader.chc_climate import (
    ChirpsProvider,
    ChirtsEra5Provider,
    iter_month_blocks,
)

REVISION = "a" * 40


class FakeArray:
    def __init__(self, values: np.ndarray, attrs: dict | None = None) -> None:
        self.values = np.asarray(values)
        self.attrs = attrs or {}

    def isel(self, **indices: object) -> "FakeArray":
        values = self.values
        if "time" in indices:
            values = values[np.asarray(indices["time"])]
        if "lat" in indices:
            values = values[:, np.asarray(indices["lat"]), :]
        if "lon" in indices:
            values = values[:, :, np.asarray(indices["lon"])]
        return FakeArray(values, self.attrs)


class FakeDataset:
    def __init__(
        self,
        times: tuple[str, ...] = ("2020-01-31", "2020-02-01", "2020-02-02"),
    ) -> None:
        shape = (len(times), 3, 3)
        self.arrays = {
            "time": FakeArray(np.asarray(times, dtype="datetime64[D]")),
            "lat": FakeArray(np.asarray((0.05, 0.0, -0.05))),
            "lon": FakeArray(np.asarray((84.95, 85.0, 85.05))),
            "precip": FakeArray(np.arange(np.prod(shape)).reshape(shape), {"units": "mm"}),
            "Tmax": FakeArray(np.full(shape, 30.0), {"units": "degC"}),
            "Tmin": FakeArray(np.full(shape, 20.0), {"units": "degC"}),
        }

    def __contains__(self, name: str) -> bool:
        return name in self.arrays

    def __getitem__(self, name: str) -> FakeArray:
        return self.arrays[name]

    def close(self) -> None:
        pass


def request(
    dataset: str,
    variables: tuple[str, ...],
    *,
    start: str = "2020-01-31",
    end: str = "2020-02-02",
    bbox: BBox = BBox(84.95, -0.05, 85.05, 0.05),
    allow_partial: bool = False,
) -> DownloadRequest:
    return DownloadRequest(
        dataset=dataset,
        bbox=bbox,
        time=TimeRange(start, end),
        variables=variables,
        output_format=OutputFormat.TIF,
        output_dir=Path("output"),
        allow_partial=allow_partial,
    )


def stub_store(monkeypatch, provider, factory=FakeDataset) -> None:
    monkeypatch.setattr(provider, "_resolve_revision", lambda: REVISION)
    monkeypatch.setattr(provider, "_open_dataset", lambda _url: factory())


def test_chc_specs_and_month_blocks() -> None:
    assert ChirpsProvider.spec.variable_ids == ("pr",)
    assert ChirtsEra5Provider.spec.variable_ids == ("tasmax", "tasmin")
    times = np.asarray(("2020-01-31", "2020-02-01", "2020-02-02"), dtype="datetime64[D]")
    assert [block.size for block in iter_month_blocks(times, cells=1)] == [1, 2]
    assert [
        block.size for block in iter_month_blocks(times, cells=1, maximum_bytes=4)
    ] == [1, 1, 1]
    url = ChirpsProvider.product.url(REVISION)
    assert REVISION in url
    assert "resolve/main" not in url


def test_chc_resolves_the_current_mirror_to_an_immutable_revision(monkeypatch) -> None:
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"sha": REVISION},
    )
    session = SimpleNamespace(get=lambda *_args, **_kwargs: response, close=lambda: None)
    monkeypatch.setattr(
        "geodaten.downloader.chc_climate.make_session", lambda: session
    )
    assert ChirpsProvider()._resolve_revision() == REVISION


def test_chc_plan_uses_store_coverage_and_native_grid(monkeypatch) -> None:
    provider = ChirpsProvider()
    stub_store(monkeypatch, provider)
    plan = provider.plan(request("chirps", ("pr",)))
    assert plan.period_count == plan.slice_count == 3
    assert plan.details["grid_shape"] == [3, 3]
    assert plan.details["available"] == ["2020-01-31", "2020-02-02"]
    assert plan.details["source_revision"] == REVISION
    assert plan.source_identity.endswith("@" + REVISION)


def test_chc_small_extent_selects_intersecting_native_cell(monkeypatch) -> None:
    provider = ChirpsProvider()
    stub_store(monkeypatch, provider)
    plan = provider.plan(
        request(
            "chirps",
            ("pr",),
            bbox=BBox(85.001, 0.001, 85.01, 0.01),
        )
    )
    assert plan.details["grid_shape"] == [1, 1]


def test_chc_missing_dates_are_strict_or_explicitly_partial(monkeypatch) -> None:
    provider = ChirpsProvider()
    stub_store(
        monkeypatch,
        provider,
        lambda: FakeDataset(("2020-02-01", "2020-02-02")),
    )
    with pytest.raises(ConfigurationError, match="does not cover"):
        provider.plan(request("chirps", ("pr",)))
    plan = provider.plan(request("chirps", ("pr",), allow_partial=True))
    assert plan.slice_count == 2
    assert "1 unavailable" in plan.warnings[0]


def test_chc_slices_emit_monthly_native_grid_blocks(monkeypatch, tmp_path: Path) -> None:
    provider = ChirtsEra5Provider()
    stub_store(monkeypatch, provider)
    item = request("chirts_era5", ("tasmax", "tasmin"))
    blocks = []
    iterator = provider.slices(item, tmp_path, lambda _event: None, CancellationToken())
    for block in iterator:
        assert Path(block.source).is_file()
        blocks.append((block.variable, len(block.times), block.attrs["units"]))
    assert blocks == [
        ("tasmax", 1, "degree_Celsius"),
        ("tasmax", 2, "degree_Celsius"),
        ("tasmin", 1, "degree_Celsius"),
        ("tasmin", 2, "degree_Celsius"),
    ]


def test_chc_rejects_unsupported_spatial_coverage() -> None:
    with pytest.raises(ConfigurationError, match="-60 through 60"):
        ChirpsProvider().validate(
            request("chirps", ("pr",), bbox=BBox(0, -61, 1, -59))
        )
    with pytest.raises(ConfigurationError, match="antimeridian"):
        ChirtsEra5Provider().validate(
            request("chirts_era5", ("tasmax",), bbox=BBox(170, 0, -170, 1))
        )
