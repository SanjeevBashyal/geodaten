"""API-free daily CHIRPS v3 and CHIRTS-ERA5 subsets from public Zarr stores."""

from __future__ import annotations

import calendar
import math
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from rasterio.transform import from_origin

from geodaten.common.errors import ConfigurationError, DownloadError, OutputError
from geodaten.common.io import make_session, safe_component
from geodaten.common.models import (
    DatasetSpec,
    DownloadPlan,
    DownloadRequest,
    RasterBlock,
    VariableSpec,
)
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent
from geodaten.common.spatial import intersecting_cell_indices
from geodaten.downloader.base import Provider, require_module

HF_API = "https://huggingface.co/api/datasets"
NODATA = -9999.0
MAX_READ_BLOCK_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class Product:
    repository: str
    store: str
    source_variables: dict[str, str]
    latitude_max: float
    doi: str

    def url(self, revision: str) -> str:
        return f"https://huggingface.co/datasets/{self.repository}/resolve/{revision}/{self.store}"


@dataclass(frozen=True)
class Selection:
    times: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    time_indices: np.ndarray
    latitude_indices: np.ndarray
    longitude_indices: np.ndarray
    coverage_start: np.datetime64
    coverage_end: np.datetime64
    warnings: tuple[str, ...]
    source_revision: str
    source_url: str


def _parse_date(value: str, *, end: bool) -> date:
    text = value.strip().lower()
    for pattern, precision in (
        ("%Y-%m-%d", "day"),
        ("%Y-%m", "month"),
        ("%Y", "year"),
    ):
        try:
            parsed = datetime.strptime(text, pattern).date()
        except ValueError:
            continue
        if end and precision == "year":
            return date(parsed.year, 12, 31)
        if end and precision == "month":
            return date(
                parsed.year,
                parsed.month,
                calendar.monthrange(parsed.year, parsed.month)[1],
            )
        return parsed
    raise ConfigurationError(
        "Temporal bounds must be earliest/latest, YYYY, YYYY-MM, or YYYY-MM-DD"
    )


def iter_month_blocks(
    times: np.ndarray,
    cells: int,
    *,
    maximum_bytes: int = MAX_READ_BLOCK_BYTES,
) -> Iterator[np.ndarray]:
    maximum_steps = max(
        1,
        maximum_bytes // max(1, cells * np.dtype("float32").itemsize),
    )
    months = times.astype("datetime64[M]")
    start = 0
    while start < times.size:
        end = start + 1
        while end < times.size and months[end] == months[start]:
            end += 1
        for block_start in range(start, end, maximum_steps):
            yield times[block_start : min(end, block_start + maximum_steps)]
        start = end


def _regular(values: np.ndarray, name: str) -> float:
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise DownloadError(f"CHC store has invalid {name} coordinates")
    if values.size == 1:
        return 0.05
    differences = np.diff(values)
    resolution = float(np.median(differences))
    if resolution <= 0 or not np.allclose(differences, resolution, atol=1e-8):
        raise DownloadError(f"CHC store has a non-regular {name} grid")
    return resolution


class ChcProvider(Provider):
    product: Product

    def __init__(self) -> None:
        self._resolved: dict[str, tuple[float, Selection]] = {}

    def _resolve_revision(self) -> str:
        session = make_session()
        try:
            response = session.get(
                f"{HF_API}/{self.product.repository}", timeout=(30, 90)
            )
            response.raise_for_status()
            revision = str(response.json().get("sha", ""))
        except (requests.RequestException, AttributeError, TypeError, ValueError) as exc:
            raise DownloadError(
                f"Could not resolve the current {self.spec.title} mirror revision"
            ) from exc
        finally:
            session.close()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise DownloadError(f"{self.spec.title} mirror returned an invalid revision")
        return revision

    def _open_dataset(self, url: str) -> Any:
        xr = require_module("xarray", "arco")
        require_module("zarr", "arco")
        require_module("fsspec", "arco")

        try:
            return xr.open_zarr(url, consolidated=None, chunks=None)
        except Exception as exc:
            raise DownloadError(
                f"Could not open {self.spec.title} public Zarr store ({type(exc).__name__})"
            ) from exc

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        if request.bbox.crosses_antimeridian:
            raise ConfigurationError(f"{self.spec.title} subsets cannot cross the antimeridian")
        if request.bbox.south < -60 or request.bbox.north > self.product.latitude_max:
            raise ConfigurationError(
                f"{self.spec.title} latitude coverage is -60 through "
                f"{self.product.latitude_max:g} degrees"
            )

    def _selection(self, request: DownloadRequest) -> Selection:
        assert request.time is not None
        revision = self._resolve_revision()
        source_url = self.product.url(revision)
        dataset = self._open_dataset(source_url)
        try:
            missing = [
                source
                for variable, source in self.product.source_variables.items()
                if variable in request.variables and source not in dataset
            ]
            if missing:
                raise DownloadError(f"{self.spec.title} store is missing: {', '.join(missing)}")
            raw_times = np.asarray(dataset["time"].values).astype("datetime64[D]")
            latitudes = np.asarray(dataset["lat"].values, dtype="float64")
            longitudes = np.asarray(dataset["lon"].values, dtype="float64")
        except (KeyError, TypeError, ValueError) as exc:
            raise DownloadError(f"{self.spec.title} store metadata is invalid") from exc
        finally:
            close = getattr(dataset, "close", None)
            if close:
                close()
        if (
            raw_times.size == 0
            or np.any(np.isnat(raw_times))
            or np.any(np.diff(raw_times) <= np.timedelta64(0, "D"))
        ):
            raise DownloadError(f"{self.spec.title} store has invalid or unordered dates")

        coverage_start, coverage_end = raw_times[0], raw_times[-1]
        start = (
            coverage_start.astype(date)
            if request.time.start.lower() == "earliest"
            else _parse_date(request.time.start, end=False)
        )
        end = (
            coverage_end.astype(date)
            if request.time.end.lower() == "latest"
            else _parse_date(request.time.end, end=True)
        )
        if start > end:
            raise ConfigurationError("Temporal start must not be after end")
        expected = np.arange(
            np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D")
        )
        times = np.intersect1d(expected, raw_times, assume_unique=True)
        unavailable = expected[~np.isin(expected, times, assume_unique=True)]
        warnings: tuple[str, ...] = ()
        if unavailable.size:
            first = np.datetime_as_string(unavailable[0], unit="D")
            last = np.datetime_as_string(unavailable[-1], unit="D")
            if not request.allow_partial:
                raise ConfigurationError(
                    f"{self.spec.title} does not cover {first} through {last}"
                )
            warnings = (
                f"Omitting {unavailable.size:,} unavailable day(s): {first}..{last}.",
            )
        if times.size == 0:
            raise DownloadError(f"No requested {self.spec.title} dates are available")

        latitude_resolution = _regular(np.sort(latitudes), "latitude")
        longitude_resolution = _regular(np.sort(longitudes), "longitude")
        latitude_indices = intersecting_cell_indices(
            latitudes, request.bbox.south, request.bbox.north, latitude_resolution
        )
        longitude_indices = intersecting_cell_indices(
            longitudes, request.bbox.west, request.bbox.east, longitude_resolution
        )
        if latitude_indices.size == 0 or longitude_indices.size == 0:
            raise ConfigurationError(f"{self.spec.title} extent contains no native grid cells")
        selected_latitudes = latitudes[latitude_indices]
        selected_longitudes = longitudes[longitude_indices]
        _regular(selected_latitudes, "selected latitude")
        _regular(selected_longitudes, "selected longitude")
        return Selection(
            times=times,
            latitudes=selected_latitudes,
            longitudes=selected_longitudes,
            time_indices=np.searchsorted(raw_times, times),
            latitude_indices=latitude_indices,
            longitude_indices=longitude_indices,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            warnings=warnings,
            source_revision=revision,
            source_url=source_url,
        )

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        selection = self._selection(request)
        self._resolved.clear()
        self._resolved[request.fingerprint] = (time.monotonic(), selection)
        cells = selection.latitudes.size * selection.longitudes.size
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=int(selection.times.size),
            slice_count=int(selection.times.size * len(request.variables)),
            estimated_bytes=int(cells * selection.times.size * len(request.variables) * 4),
            details={
                "available": [
                    np.datetime_as_string(selection.coverage_start, unit="D"),
                    np.datetime_as_string(selection.coverage_end, unit="D"),
                ],
                "daily_steps": int(selection.times.size),
                "grid_shape": [int(selection.latitudes.size), int(selection.longitudes.size)],
                "resolution_degrees": 0.05,
                "maximum_decoded_block_bytes": MAX_READ_BLOCK_BYTES,
                "access": "anonymous Zarr v3",
                "source_revision": selection.source_revision,
            },
            warnings=selection.warnings,
            source_identity=f"hf:{self.product.repository}@{selection.source_revision}",
        )

    def slices(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Iterator[RasterBlock]:
        self.validate(request)
        cached = self._resolved.pop(request.fingerprint, None)
        selection = (
            cached[1]
            if cached is not None and time.monotonic() - cached[0] <= 300
            else self._selection(request)
        )
        for warning in selection.warnings:
            progress(ProgressEvent("warning", warning))
        resolution = _regular(selection.longitudes, "selected longitude")
        latitude_resolution = _regular(selection.latitudes, "selected latitude")
        transform = from_origin(
            float(selection.longitudes[0] - resolution / 2),
            float(selection.latitudes[-1] + latitude_resolution / 2),
            resolution,
            latitude_resolution,
        )
        cells = selection.latitudes.size * selection.longitudes.size
        dataset = self._open_dataset(selection.source_url)
        try:
            for variable in request.variables:
                source_name = self.product.source_variables[variable]
                source = dataset[source_name]
                spec = next(item for item in self.spec.variables if item.id == variable)
                attrs = dict(getattr(source, "attrs", {}))
                attrs.update(
                    long_name=spec.title,
                    units=spec.units,
                    standard_name=spec.standard_name,
                    _FillValue=NODATA,
                    calendar="proleptic_gregorian",
                    source=self.spec.title,
                )
                attrs["cell_methods"] = (
                    "time: sum"
                    if variable == "pr"
                    else f"time: {'maximum' if variable == 'tasmax' else 'minimum'}"
                )
                for block in iter_month_blocks(selection.times, cells):
                    cancel.check()
                    positions = np.searchsorted(selection.times, block)
                    time_indices = selection.time_indices[positions]
                    first = np.datetime_as_string(block[0], unit="D")
                    last = np.datetime_as_string(block[-1], unit="D")
                    progress(
                        ProgressEvent(
                            "download",
                            f"Reading {self.spec.title} {variable} {first}..{last}",
                        )
                    )
                    try:
                        data = np.asanyarray(
                            source.isel(
                                time=time_indices,
                                lat=selection.latitude_indices,
                                lon=selection.longitude_indices,
                            ).values
                        )
                        if np.ma.isMaskedArray(data):
                            data = data.filled(np.nan)
                        data = np.asarray(data, dtype="float32")
                    except Exception as exc:
                        raise DownloadError(
                            f"Could not read {self.spec.title} {variable} for {first}..{last}"
                        ) from exc
                    expected_shape = (
                        block.size,
                        selection.latitudes.size,
                        selection.longitudes.size,
                    )
                    if data.shape != expected_shape:
                        raise OutputError(
                            f"{self.spec.title} returned shape {data.shape}; "
                            f"expected {expected_shape}"
                        )
                    path = workspace / (
                        f"{safe_component(variable)}_{safe_component(first)}_"
                        f"{safe_component(last)}.tif"
                    )
                    with rasterio.open(
                        path,
                        "w",
                        driver="GTiff",
                        width=expected_shape[2],
                        height=expected_shape[1],
                        count=expected_shape[0],
                        dtype="float32",
                        crs="EPSG:4326",
                        transform=transform,
                        nodata=NODATA,
                        tiled=True,
                        blockxsize=256,
                        blockysize=256,
                        compress="deflate",
                        predictor=3,
                        bigtiff="IF_SAFER",
                    ) as target:
                        for band, layer in enumerate(data, 1):
                            cancel.check()
                            target.write(
                                np.flipud(np.where(np.isfinite(layer), layer, NODATA)),
                                band,
                            )
                    del data
                    try:
                        yield RasterBlock(
                            variable=variable,
                            source=path,
                            times=tuple(timestamp.astype(date) for timestamp in block),
                            attrs=attrs,
                        )
                    finally:
                        path.unlink(missing_ok=True)
        finally:
            close = getattr(dataset, "close", None)
            if close:
                close()

    def manifest_metadata(
        self, request: DownloadRequest, plan: DownloadPlan
    ) -> dict[str, Any]:
        metadata = dict(super().manifest_metadata(request, plan))
        metadata.update(
            mirror=(
                self.product.url(str(plan.details["source_revision"]))
                if plan.details.get("source_revision")
                else ""
            ),
            source_revision=plan.details.get("source_revision"),
            doi=self.product.doi,
        )
        return metadata


class ChirpsProvider(ChcProvider):
    product = Product(
        "aaguilar90/chirps-v3-daily-rnl",
        "chirps-v3-daily-rnl.zarr",
        {"pr": "precip"},
        60.0,
        "10.15780/G2JQ0P",
    )
    spec = DatasetSpec(
        id="chirps",
        title="CHIRPS v3 daily",
        summary="Daily 0.05 degree precipitation from the CHIRPS v3 final RNL product.",
        group="Climate",
        variables=(
            VariableSpec(
                "pr",
                "Daily precipitation",
                "mm",
                "precipitation_amount",
            ),
        ),
        source_url="https://www.chc.ucsb.edu/data/chirps3",
        citation="Climate Hazards Center, CHIRPS v3 Data Repository (2025).",
        license="CC BY 4.0",
    )


class ChirtsEra5Provider(ChcProvider):
    product = Product(
        "aaguilar90/chirts-era5-daily",
        "chirts-era5-daily.zarr",
        {"tasmax": "Tmax", "tasmin": "Tmin"},
        70.0,
        "10.15780/G2F08J",
    )
    spec = DatasetSpec(
        id="chirts_era5",
        title="CHIRTS-ERA5 daily",
        summary="Daily 0.05 degree bias-corrected maximum and minimum air temperature.",
        group="Climate",
        variables=(
            VariableSpec(
                "tasmax",
                "Daily maximum temperature",
                "degree_Celsius",
                "air_temperature",
            ),
            VariableSpec(
                "tasmin",
                "Daily minimum temperature",
                "degree_Celsius",
                "air_temperature",
            ),
        ),
        source_url="https://www.chc.ucsb.edu/data/chirts-era5",
        citation="Climate Hazards Center, CHIRTS-ERA5 Data Repository (2025).",
        license="CC BY 4.0",
    )
