from __future__ import annotations

import sys
import threading
import time
from datetime import date
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds, from_origin

from geodaten.cli import parse_provider_options
from geodaten.common.auth import redacted, resolve_credentials
from geodaten.common.errors import CancelledError, OutputError
from geodaten.common.gdal import run as run_process
from geodaten.common.gdal import translate_to_cog
from geodaten.common.models import (
    BBox,
    CredentialField,
    DownloadRequest,
    OptionSpec,
    OutputFormat,
    RasterSlice,
    TimeRange,
)
from geodaten.common.progress import CancellationToken
from geodaten.common.writers import NetCDFWriter, TiffWriter


def request(tmp_path: Path, output_format: OutputFormat) -> DownloadRequest:
    return DownloadRequest(
        dataset="example",
        bbox=BBox(10, 20, 12, 22),
        time=TimeRange("2020-01-01", "2020-01-02"),
        variables=("temperature",),
        output_format=output_format,
        output_dir=tmp_path,
        credentials={"secret": "must-not-appear"},
    )


def source_raster(tmp_path: Path) -> Path:
    path = tmp_path / "source.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(10, 22, 0.5, 0.5),
        nodata=-9999,
    ) as destination:
        destination.write(np.arange(12, dtype="float32").reshape(3, 4), 1)
    return path


def test_bbox_supports_antimeridian_segments() -> None:
    bbox = BBox(170, -10, -170, 10)
    assert bbox.crosses_antimeridian
    assert bbox.width == 20
    assert [segment.as_tuple() for segment in bbox.segments()] == [
        (170.0, -10.0, 180.0, 10.0),
        (-180.0, -10.0, -170.0, 10.0),
    ]


def test_request_fingerprint_and_public_dict_exclude_credentials(tmp_path: Path) -> None:
    first = request(tmp_path, OutputFormat.TIF)
    second = DownloadRequest(**{**first.__dict__, "credentials": {"secret": "different"}})
    third = DownloadRequest(**{**first.__dict__, "workers": first.workers + 1})
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint == third.fingerprint
    assert "secret" not in str(first.public_dict())
    assert redacted(first.credentials) == {"secret": "***"}


def test_provider_option_parser() -> None:
    specs = (
        OptionSpec("count", "Count", kind="int", default=2),
        OptionSpec("retries", "Retries", kind="integer", default=1),
        OptionSpec("resolution", "Resolution", kind="number", default=0.5),
        OptionSpec("layers", "Layers", multiple=True),
        OptionSpec("enabled", "Enabled", kind="bool", default=False),
    )
    parsed = parse_provider_options(
        [
            "count=3",
            "retries=4",
            "resolution=0.25",
            "layers=a,b",
            "enabled=yes",
        ],
        specs,
    )
    assert parsed == {
        "count": 3,
        "retries": 4,
        "resolution": 0.25,
        "layers": ("a", "b"),
        "enabled": True,
    }


def test_provider_option_parser_does_not_materialize_omitted_defaults(
    tmp_path: Path,
) -> None:
    specs = (
        OptionSpec("count", "Count", kind="int", default=2),
        OptionSpec("enabled", "Enabled", kind="bool", default=False),
    )
    parsed = parse_provider_options([], specs)
    ordinary = request(tmp_path, OutputFormat.TIF)
    interface = DownloadRequest(**{**ordinary.__dict__, "options": parsed})

    assert parsed == {}
    assert interface.fingerprint == ordinary.fingerprint


def test_credentials_preserve_user_value_and_ignore_none(monkeypatch) -> None:
    fields = (
        CredentialField("secret", "Secret", ("TEST_GEODATEN_SECRET",)),
        CredentialField("missing", "Missing"),
    )
    monkeypatch.delenv("TEST_GEODATEN_SECRET", raising=False)
    assert resolve_credentials(fields, {"secret": "  exact value  ", "missing": None}) == {
        "secret": "  exact value  "
    }


def test_tiff_writer_creates_single_band_cog(tmp_path: Path) -> None:
    source = source_raster(tmp_path)
    writer = TiffWriter(request(tmp_path, OutputFormat.TIF))
    output = writer.write(
        RasterSlice(
            "temperature",
            source,
            time=date(2020, 1, 1),
            attrs={"units": "K", "_FillValue": 0, "scale_factor": 0.1, "add_offset": 2},
        )
    )
    assert writer.close() == (output,)
    with rasterio.open(output) as dataset:
        assert dataset.count == 1
        assert dataset.width == 4
        assert dataset.tags()["VARIABLE"] == "temperature"
        assert dataset.tags()["REQUEST_FINGERPRINT"] == writer.request.fingerprint
        assert dataset.tags()["OUTPUT_NAME"] == output.name
        assert dataset.nodata == 0
        assert dataset.scales == (0.1,)
        assert dataset.offsets == (2.0,)


def test_netcdf_writer_appends_time_slices_blockwise(tmp_path: Path) -> None:
    source = source_raster(tmp_path)
    writer = NetCDFWriter(request(tmp_path, OutputFormat.NETCDF))
    writer.write(
        RasterSlice(
            "temperature",
            source,
            time=date(2020, 1, 1),
            attrs={
                "units": "K",
                "standard_name": "air_temperature",
                "coordinates": "time lat lon",
                "cell_measures": "area: areacella",
            },
        )
    )
    writer.write(
        RasterSlice(
            "temperature",
            source,
            time=date(2020, 1, 2),
            attrs={"units": "K", "standard_name": "air_temperature"},
        )
    )
    (output,) = writer.close()
    with netCDF4.Dataset(output) as dataset:
        assert dataset.Conventions == "CF-1.13"
        assert dataset.variables["temperature"].shape == (2, 3, 4)
        assert dataset.variables["temperature"].units == "K"
        assert dataset.variables["temperature"].coordinates == "y x"
        assert "cell_measures" not in dataset.variables["temperature"].ncattrs()
        assert dataset.variables["x"].standard_name == "longitude"
        assert dataset.geospatial_bounds.startswith("POLYGON(")
        assert dataset.geospatial_bounds_crs == "EPSG:4326"


def test_invalid_netcdf_overwrite_preserves_previous_output(tmp_path: Path, monkeypatch) -> None:
    source = source_raster(tmp_path)
    base = request(tmp_path, OutputFormat.NETCDF)
    overwrite = DownloadRequest(**{**base.__dict__, "overwrite": True})
    writer = NetCDFWriter(overwrite)
    writer.target.parent.mkdir(parents=True, exist_ok=True)
    writer.target.write_bytes(b"previous valid output")
    writer.write(RasterSlice("temperature", source, time=date(2020, 1, 1)))
    monkeypatch.setattr(writer, "_is_complete", lambda _path: False)

    with pytest.raises(OutputError, match="failed validation"):
        writer.close()

    assert writer.target.read_bytes() == b"previous valid output"


def test_netcdf_rejects_raster_without_crs(tmp_path: Path) -> None:
    source = tmp_path / "no-crs.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        transform=from_origin(0, 2, 1, 1),
    ) as target:
        target.write(np.ones((2, 2), dtype="uint8"), 1)

    writer = NetCDFWriter(request(tmp_path, OutputFormat.NETCDF))
    with pytest.raises(OutputError, match="require a CRS"):
        writer.write(RasterSlice("temperature", source))
    writer.abort()


def test_netcdf_completeness_rejects_coordinate_only_file(tmp_path: Path) -> None:
    path = tmp_path / "coordinates.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.Conventions = "CF-1.13"
        dataset.createDimension("x", 1)
        dataset.createDimension("y", 1)
        dataset.createVariable("x", "f8", ("x",))
        dataset.createVariable("y", "f8", ("y",))
        dataset.createVariable("crs", "i4")

    assert not NetCDFWriter._is_complete(path)


def test_netcdf_latest_request_never_short_circuits_existing_output(
    tmp_path: Path,
) -> None:
    source = source_raster(tmp_path)
    base = request(tmp_path, OutputFormat.NETCDF)
    dynamic = DownloadRequest(
        **{**base.__dict__, "time": TimeRange("2020-01-01", "latest")}
    )
    first = NetCDFWriter(dynamic)
    first.write(RasterSlice("temperature", source, time=date(2020, 1, 1)))
    first.close()

    retry = NetCDFWriter(dynamic)

    assert not retry.complete
    assert retry.outputs == []
    retry.abort()


def test_projected_crop_is_warped_to_exact_wgs84_extent(tmp_path: Path) -> None:
    source = tmp_path / "projected.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_bounds(-250000, -250000, 250000, 250000, 100, 100),
    ) as target:
        target.write(np.ones((100, 100), dtype="uint8"), 1)

    output = translate_to_cog(source, tmp_path / "cropped.tif", bbox=BBox(-1, -1, 1, 1))

    with rasterio.open(output) as dataset:
        assert dataset.crs.to_epsg() == 4326
        assert tuple(dataset.bounds) == pytest.approx((-1, -1, 1, 1), abs=1e-8)


def test_warp_selects_one_band_through_compatible_vrt(tmp_path: Path) -> None:
    source = tmp_path / "multiband.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=2,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(10, 22, 0.5, 0.5),
    ) as target:
        target.write(np.ones((4, 4), dtype="uint8"), 1)
        target.write(np.full((4, 4), 7, dtype="uint8"), 2)

    output = translate_to_cog(
        source,
        tmp_path / "band-2.tif",
        bbox=BBox(10, 20, 12, 22),
        band=2,
    )

    with rasterio.open(output) as dataset:
        assert dataset.count == 1
        assert np.all(dataset.read(1) == 7)


def test_external_process_honors_cancellation() -> None:
    token = CancellationToken()
    timer = threading.Timer(0.1, token.cancel)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(CancelledError):
            run_process(
                (sys.executable, "-c", "import time; time.sleep(10)"),
                cancel=token,
            )
    finally:
        timer.cancel()
    assert time.monotonic() - started < 3


def test_copernicus_reference_credentials_are_empty() -> None:
    source = Path("codes-to-merge/download_copernicus_lai_v2.py").read_text()
    assert 'S3_ACCESS_KEY = ""' in source
    assert 'S3_SECRET_KEY = ""' in source
