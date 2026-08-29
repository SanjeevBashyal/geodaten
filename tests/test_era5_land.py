from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from geodaten.common.errors import AuthenticationError, ConfigurationError
from geodaten.common.models import BBox, DownloadRequest, OutputFormat, TimeRange
from geodaten.common.progress import CancellationToken
from geodaten.downloader.arco import resolve_cds_api_key
from geodaten.downloader.era5_land import (
    GROUP_BY_VARIABLE,
    Era5LandProvider,
    iter_month_blocks,
    store_url,
)


class FakeArray:
    def __init__(self, values: np.ndarray, attrs: dict | None = None) -> None:
        self.values = np.asarray(values)
        self.attrs = attrs or {}
        self.selections: list[dict[str, object]] = []

    def isel(self, **indices: object) -> "FakeArray":
        self.selections.append(indices)
        values = self.values
        if "time" in indices:
            values = values[np.asarray(indices["time"])]
        if "latitude" in indices:
            values = values[:, np.asarray(indices["latitude"]), :]
        if "longitude" in indices:
            values = values[:, :, np.asarray(indices["longitude"])]
        return FakeArray(values, self.attrs)


class FakeDataset:
    def __init__(self, times: tuple[str, ...], variables: tuple[str, ...]) -> None:
        latitude = np.array([-0.1, 0.0, 0.1])
        longitude = np.round(np.arange(-179.9, 180.01, 0.1), 1)
        self.arrays = {
            "time": FakeArray(np.asarray(times, dtype="datetime64[h]")),
            "latitude": FakeArray(latitude),
            "longitude": FakeArray(longitude),
        }
        shape = (len(times), len(latitude), len(longitude))
        for offset, variable in enumerate(variables):
            self.arrays[variable] = FakeArray(
                np.arange(np.prod(shape), dtype="float32").reshape(shape) + offset,
                {"units": "K"},
            )

    def __contains__(self, name: str) -> bool:
        return name in self.arrays

    def __getitem__(self, name: str) -> FakeArray:
        return self.arrays[name]

    def close(self) -> None:
        pass


def request(
    *,
    bbox: BBox = BBox(179.8, -0.1, -179.8, 0.1),
    start: str = "2020-01-02",
    end: str = "2020-01-02",
    hours: tuple[str, ...] = ("00:00", "12:00"),
    allow_partial: bool = False,
) -> DownloadRequest:
    return DownloadRequest(
        dataset="era5_land",
        bbox=bbox,
        time=TimeRange(start, end),
        variables=("t2m",),
        output_format=OutputFormat.TIF,
        output_dir=Path("output"),
        options={"hours": hours},
        credentials={"cds_api_key": "user-token"},
        allow_partial=allow_partial,
    )


def fake_open(times: tuple[str, ...], opened: list[FakeDataset] | None = None):
    def open_dataset(url: str, _key: str) -> FakeDataset:
        variables = GROUP_BY_VARIABLE["t2m"].variables
        assert url == store_url(GROUP_BY_VARIABLE["t2m"], "time")
        dataset = FakeDataset(times, variables)
        if opened is not None:
            opened.append(dataset)
        return dataset

    return open_dataset


def test_official_group_routes_and_variable_inventory() -> None:
    provider = Era5LandProvider()
    assert len(provider.spec.variables) == 19
    assert store_url(GROUP_BY_VARIABLE["stl1"], "geo").endswith(
        "/cadl-arco-geo-006/arco/reanalysis_era5_land/"
        "sfc-soil-temperature/geoChunked.zarr"
    )
    assert store_url(GROUP_BY_VARIABLE["ssrd"], "time").endswith(
        "/cadl-arco-time-010/arco/reanalysis_era5_land/"
        "sfc-radiation-heat/timeChunked.zarr"
    )


def test_month_blocks_follow_calendar_and_memory_ceiling() -> None:
    times = np.asarray(
        ("2020-01-31T23", "2020-02-01T00", "2020-02-01T01"),
        dtype="datetime64[h]",
    )
    assert [block.size for block in iter_month_blocks(times, cells=1)] == [1, 2]
    assert [block.size for block in iter_month_blocks(times, cells=1, maximum_bytes=4)] == [
        1,
        1,
        1,
    ]


def test_auto_chunking_uses_time_layout_for_monthly_retrieval() -> None:
    provider = Era5LandProvider()
    req = request(
        bbox=BBox(7, 50, 8, 51),
        start="2020-02",
        end="2020-02",
    )
    assert provider._chunking(req) == "time"


def test_cds_key_uses_only_user_sources(tmp_path: Path) -> None:
    rc = tmp_path / ".cdsapirc"
    rc.write_text("url: https://cds.climate.copernicus.eu/api\nkey: rc-token\n")
    assert resolve_cds_api_key({}, environ={}, rc_path=rc) == "rc-token"
    assert resolve_cds_api_key(
        {"cds_api_key": "request-token"},
        environ={"CDSAPI_KEY": "environment-token"},
        rc_path=rc,
    ) == "request-token"
    with pytest.raises(AuthenticationError, match="CDS API key"):
        resolve_cds_api_key({}, environ={}, rc_path=tmp_path / "missing")


def test_plan_uses_live_times_and_reports_missing_strictly(monkeypatch) -> None:
    provider = Era5LandProvider()
    monkeypatch.setattr(
        provider,
        "_open_dataset",
        fake_open(("2020-01-02T00", "2020-01-02T12")),
    )
    plan = provider.plan(request())
    assert plan.period_count == 1
    assert plan.slice_count == 2
    assert plan.details["grid_shape"] == [3, 5]
    assert plan.details["chunking"] == "time"
    assert plan.details["monthly_read_blocks_per_variable"] == 1

    monkeypatch.setattr(provider, "_open_dataset", fake_open(("2020-01-02T00",)))
    with pytest.raises(ConfigurationError, match="2020-01-02T12"):
        provider.plan(request())
    partial = provider.plan(request(allow_partial=True))
    assert partial.slice_count == 1
    assert "1 unavailable" in partial.warnings[0]


def test_slices_fetch_one_month_block_and_emit_hourly_bands(
    monkeypatch, tmp_path: Path
) -> None:
    provider = Era5LandProvider()
    opened: list[FakeDataset] = []
    monkeypatch.setattr(
        provider,
        "_open_dataset",
        fake_open(("2020-01-02T00", "2020-01-02T12"), opened),
    )
    req = request()
    iterator = provider.slices(req, tmp_path, lambda _event: None, CancellationToken())
    block = next(iterator)
    assert block.times == (datetime(2020, 1, 2), datetime(2020, 1, 2, 12))
    assert block.band == 1
    with __import__("rasterio").open(block.source) as dataset:
        assert (dataset.count, dataset.height, dataset.width) == (2, 3, 5)
        assert dataset.transform.c == pytest.approx(179.75)
        assert dataset.bounds.right == pytest.approx(180.25)
        assert dataset.nodata == -9999.0
    selections = opened[-1]["t2m"].selections
    assert len(selections) == 2  # one selection for each antimeridian segment
    assert np.array_equal(selections[0]["time"], np.array([0, 1]))
    with pytest.raises(StopIteration):
        next(iterator)
    assert not Path(block.source).exists()
