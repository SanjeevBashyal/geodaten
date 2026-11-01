from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import geodaten.downloader.service as service_module
from geodaten.common.models import (
    BBox,
    DatasetSpec,
    DownloadPlan,
    DownloadRequest,
    OutputFormat,
    RasterSlice,
    TimeRange,
    VariableSpec,
)
from geodaten.common.progress import ProgressEvent
from geodaten.downloader.base import Provider
from geodaten.downloader.service import DownloadService


class FakeProvider(Provider):
    spec = DatasetSpec(
        id="fake",
        title="Fake",
        summary="Test provider",
        variables=(VariableSpec("value", "Value", units="1"),),
    )

    def __init__(self) -> None:
        self.slice_calls = 0
        self.source_identity = "fake:v1"

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        return DownloadPlan(
            self.spec.id,
            period_count=2,
            slice_count=2,
            details={
                "download_url": "https://user:password@example.test/data?token=value",
                "api_token": "must-not-be-written",
            },
            source_identity=self.source_identity,
        )

    def slices(self, request, workspace, progress, cancel):
        self.slice_calls += 1
        path = workspace / "slice.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_origin(10, 22, 1, 1),
        ) as target:
            target.write(np.arange(4, dtype="uint16").reshape(2, 2), 1)
        for day in (date(2020, 1, 1), date(2020, 1, 2)):
            yield RasterSlice("value", path, time=day, attrs={"units": "1"})


class WarningProvider(FakeProvider):
    def slices(self, request, workspace, progress, cancel):
        progress(ProgressEvent("warning", "Omitted never-write-this"))
        progress(ProgressEvent("warning", "Omitted never-write-this"))
        yield from super().slices(request, workspace, progress, cancel)


def make_request(tmp_path: Path, output_format: OutputFormat) -> DownloadRequest:
    return DownloadRequest(
        dataset="fake",
        bbox=BBox(10, 20, 12, 22),
        time=TimeRange("2020-01-01", "2020-01-02"),
        variables=("value",),
        output_format=output_format,
        output_dir=tmp_path,
        credentials={"token": "never-write-this"},
    )


def test_service_writes_one_tiff_per_variable_and_time(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(service_module, "get_provider", lambda _identifier: FakeProvider())
    result = DownloadService().download(make_request(tmp_path, OutputFormat.TIF))

    assert len(result.outputs) == 2
    assert all(path.suffix == ".tif" for path in result.outputs)
    manifest = json.loads(result.manifest.read_text())
    assert manifest["request"]["variables"] == ["value"]
    assert "never-write-this" not in result.manifest.read_text()
    assert "user:password" not in result.manifest.read_text()
    assert "?token=" not in result.manifest.read_text()
    assert manifest["plan"]["details"]["api_token"] == "***"


def test_service_writes_one_netcdf_for_entire_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(service_module, "get_provider", lambda _identifier: FakeProvider())
    result = DownloadService().download(make_request(tmp_path, OutputFormat.NETCDF))

    assert len(result.outputs) == 1
    assert result.outputs[0].suffix == ".nc"
    import netCDF4

    with netCDF4.Dataset(result.outputs[0]) as dataset:
        assert dataset.source_identity == "fake:v1"


def test_service_resumes_complete_tiff_request_without_provider_slices(
    tmp_path: Path, monkeypatch
) -> None:
    provider = FakeProvider()
    monkeypatch.setattr(service_module, "get_provider", lambda _identifier: provider)
    item = make_request(tmp_path, OutputFormat.TIF)
    first = DownloadService().download(item)
    first.manifest.unlink()
    retry_cache = tmp_path / "unused-resume-cache"
    retry = DownloadRequest(**{**item.__dict__, "cache_dir": retry_cache})

    resumed = DownloadService().download(retry)

    assert provider.slice_calls == 1
    assert resumed.outputs == first.outputs
    assert resumed.skipped == 2
    assert resumed.manifest.is_file()
    assert not retry_cache.exists()


def test_service_separates_outputs_when_the_source_identity_changes(
    tmp_path: Path, monkeypatch
) -> None:
    provider = FakeProvider()
    monkeypatch.setattr(service_module, "get_provider", lambda _identifier: provider)
    item = make_request(tmp_path, OutputFormat.TIF)
    first = DownloadService().download(item)

    provider.source_identity = "fake:v2"
    second = DownloadService().download(item)

    assert provider.slice_calls == 2
    assert first.outputs[0].parent != second.outputs[0].parent
    assert first.outputs[0].is_file() and second.outputs[0].is_file()
    manifest = json.loads(second.manifest.read_text())
    assert manifest["plan"]["source_identity"] == "fake:v2"
    with rasterio.open(second.outputs[0]) as dataset:
        assert dataset.tags()["SOURCE_IDENTITY"] == "fake:v2"


def test_service_keeps_explicit_netcdf_existing_output_shortcut(
    tmp_path: Path, monkeypatch
) -> None:
    provider = FakeProvider()
    monkeypatch.setattr(service_module, "get_provider", lambda _identifier: provider)
    item = make_request(tmp_path, OutputFormat.NETCDF)
    DownloadService().download(item)
    retry_cache = tmp_path / "unused-netcdf-resume-cache"
    retry = DownloadRequest(**{**item.__dict__, "cache_dir": retry_cache})

    resumed = DownloadService().download(retry)

    assert provider.slice_calls == 1
    assert resumed.skipped == 1
    assert not retry_cache.exists()


def test_service_rebuilds_netcdf_with_mismatched_identity(
    tmp_path: Path, monkeypatch
) -> None:
    import netCDF4

    provider = FakeProvider()
    monkeypatch.setattr(service_module, "get_provider", lambda _identifier: provider)
    item = make_request(tmp_path, OutputFormat.NETCDF)
    first = DownloadService().download(item)
    with netCDF4.Dataset(first.outputs[0], "r+") as dataset:
        dataset.request_fingerprint = "unrelated"

    rebuilt = DownloadService().download(item)

    assert provider.slice_calls == 2
    assert rebuilt.skipped == 0
    with netCDF4.Dataset(rebuilt.outputs[0]) as dataset:
        assert dataset.request_fingerprint == item.fingerprint


@pytest.mark.parametrize("output_format", (OutputFormat.TIF, OutputFormat.NETCDF))
def test_service_latest_requests_refresh_provider_slices_and_outputs(
    tmp_path: Path, monkeypatch, output_format: OutputFormat
) -> None:
    provider = FakeProvider()
    monkeypatch.setattr(service_module, "get_provider", lambda _identifier: provider)
    base = make_request(tmp_path, output_format)
    dynamic = DownloadRequest(
        **{**base.__dict__, "time": TimeRange("2020-01-01", "latest")}
    )
    DownloadService().download(dynamic)

    refreshed = DownloadService().download(dynamic)

    assert provider.slice_calls == 2
    assert refreshed.skipped == 0


@pytest.mark.parametrize("damage", ("incomplete", "invalid", "extra", "arbitrary"))
def test_service_does_not_fast_resume_unsafe_tiff_sets(
    tmp_path: Path, monkeypatch, damage: str
) -> None:
    provider = FakeProvider()
    monkeypatch.setattr(service_module, "get_provider", lambda _identifier: provider)
    item = make_request(tmp_path, OutputFormat.TIF)
    first = DownloadService().download(item)

    if damage == "incomplete":
        first.outputs[0].unlink()
    elif damage == "invalid":
        first.outputs[0].write_bytes(b"not a raster")
    elif damage == "extra":
        shutil.copyfile(first.outputs[0], first.outputs[0].parent / "extra.tif")
    else:
        target = first.outputs[0]
        target.unlink()
        with rasterio.open(
            target,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_origin(10, 22, 1, 1),
        ) as dataset:
            dataset.write(np.ones((2, 2), dtype="uint16"), 1)

    DownloadService().download(item)

    assert provider.slice_calls == 2
    if damage == "arbitrary":
        with rasterio.open(first.outputs[0]) as dataset:
            assert dataset.tags()["REQUEST_FINGERPRINT"] == item.fingerprint
            assert not np.all(dataset.read(1) == 1)


def test_service_records_unique_sanitized_runtime_warnings(
    tmp_path: Path, monkeypatch
) -> None:
    provider = WarningProvider()
    monkeypatch.setattr(service_module, "get_provider", lambda _identifier: provider)
    delivered: list[ProgressEvent] = []

    result = DownloadService().download(
        make_request(tmp_path, OutputFormat.TIF), progress=delivered.append
    )
    manifest = json.loads(result.manifest.read_text())

    assert [event.message for event in delivered if event.stage == "warning"] == [
        "Omitted ***",
        "Omitted ***",
    ]
    assert manifest["provider"]["runtime_warnings"] == ["Omitted ***"]
    assert "never-write-this" not in result.manifest.read_text()

    resumed = DownloadService().download(make_request(tmp_path, OutputFormat.TIF))
    resumed_manifest = json.loads(resumed.manifest.read_text())
    assert provider.slice_calls == 1
    assert resumed_manifest["provider"]["runtime_warnings"] == ["Omitted ***"]
