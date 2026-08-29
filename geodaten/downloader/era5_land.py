"""Hourly ERA5-Land subsets from ECMWF's authenticated ARCO Zarr service."""

from __future__ import annotations

import calendar
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geodaten.common.errors import ConfigurationError, DownloadError, OutputError
from geodaten.common.io import safe_component
from geodaten.common.models import (
    CredentialField,
    DatasetSpec,
    DownloadPlan,
    DownloadRequest,
    OptionSpec,
    RasterBlock,
    VariableSpec,
)
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent
from geodaten.downloader.arco import open_arco_dataset, resolve_cds_api_key
from geodaten.downloader.base import Provider


ARCO_ROOT = "https://arco.datastores.ecmwf.int"
HOURS = tuple(f"{hour:02d}:00" for hour in range(24))
NODATA = -9999.0
MAX_READ_BLOCK_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ArcoGroup:
    id: str
    code: str
    slug: str
    variables: tuple[str, ...]


GROUPS = (
    ArcoGroup("2m_temperature", "007", "sfc-2m-temperature", ("t2m", "d2m")),
    ArcoGroup("pressure_precipitation", "009", "sfc-pressure-precipitation", ("tp", "sp")),
    ArcoGroup("wind", "008", "sfc-wind", ("u10", "v10")),
    ArcoGroup("radiation_heat", "010", "sfc-radiation-heat", ("ssrd", "strd")),
    ArcoGroup("skin_temperature", "043", "sfc-skin-temperature", ("skt",)),
    ArcoGroup("soil_water", "005", "sfc-soil-water", ("swvl1", "swvl2", "swvl3", "swvl4")),
    ArcoGroup("soil_temperature", "006", "sfc-soil-temperature", ("stl1", "stl2", "stl3", "stl4")),
    ArcoGroup("snow", "030", "sfc-snow", ("snowc", "sd")),
)
GROUP_BY_VARIABLE = {variable: group for group in GROUPS for variable in group.variables}


VARIABLES = (
    VariableSpec("t2m", "2 metre temperature", "K", "air_temperature"),
    VariableSpec("d2m", "2 metre dewpoint temperature", "K", "dew_point_temperature"),
    VariableSpec(
        "tp",
        "Hourly total precipitation",
        "m",
        "lwe_thickness_of_precipitation_amount",
        "Already de-accumulated by ECMWF.",
    ),
    VariableSpec("sp", "Surface pressure", "Pa", "surface_air_pressure"),
    VariableSpec("u10", "10 metre U wind component", "m s-1", "eastward_wind"),
    VariableSpec("v10", "10 metre V wind component", "m s-1", "northward_wind"),
    VariableSpec(
        "ssrd",
        "Hourly surface solar radiation downwards",
        "J m-2",
        description="Already de-accumulated by ECMWF.",
    ),
    VariableSpec(
        "strd",
        "Hourly surface thermal radiation downwards",
        "J m-2",
        description="Already de-accumulated by ECMWF.",
    ),
    VariableSpec("skt", "Skin temperature", "K", "surface_temperature"),
    VariableSpec(
        "swvl1", "Volumetric soil water layer 1", "m3 m-3", "volume_fraction_of_water_in_soil"
    ),
    VariableSpec(
        "swvl2", "Volumetric soil water layer 2", "m3 m-3", "volume_fraction_of_water_in_soil"
    ),
    VariableSpec(
        "swvl3", "Volumetric soil water layer 3", "m3 m-3", "volume_fraction_of_water_in_soil"
    ),
    VariableSpec(
        "swvl4", "Volumetric soil water layer 4", "m3 m-3", "volume_fraction_of_water_in_soil"
    ),
    VariableSpec("stl1", "Soil temperature level 1", "K", "soil_temperature"),
    VariableSpec("stl2", "Soil temperature level 2", "K", "soil_temperature"),
    VariableSpec("stl3", "Soil temperature level 3", "K", "soil_temperature"),
    VariableSpec("stl4", "Soil temperature level 4", "K", "soil_temperature"),
    VariableSpec("snowc", "Snow cover", "%", "surface_snow_area_fraction"),
    VariableSpec("sd", "Snow depth", "m", "surface_snow_thickness"),
)
VARIABLE_SPEC = {variable.id: variable for variable in VARIABLES}


def store_url(group: ArcoGroup, chunking: str) -> str:
    prefix = "geo" if chunking == "geo" else "time"
    return (
        f"{ARCO_ROOT}/cadl-arco-{prefix}-{group.code}/arco/"
        f"reanalysis_era5_land/{group.slug}/{prefix}Chunked.zarr"
    )


def _values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    elif isinstance(value, (tuple, list)):
        values = [str(item).strip() for item in value]
    else:
        values = [str(value).strip()]
    return tuple(dict.fromkeys(item for item in values if item))


def _parse_date(value: str, *, end: bool) -> date:
    text = value.strip()
    formats = (("%Y-%m-%d", None), ("%Y-%m", "month"), ("%Y", "year"))
    for pattern, precision in formats:
        try:
            parsed = datetime.strptime(text, pattern).date()
        except ValueError:
            continue
        if end and precision == "year":
            return date(parsed.year, 12, 31)
        if end and precision == "month":
            last_day = calendar.monthrange(parsed.year, parsed.month)[1]
            return date(parsed.year, parsed.month, last_day)
        return parsed
    raise ConfigurationError(
        "ERA5-Land temporal bounds must be earliest/latest, YYYY, YYYY-MM, or YYYY-MM-DD"
    )


def _requested_hours(value: Any) -> tuple[int, ...]:
    result: list[int] = []
    for item in _values(value):
        text = item.strip()
        if len(text) == 2 and text.isdigit():
            text += ":00"
        if text not in HOURS:
            raise ConfigurationError(
                f"Invalid ERA5-Land hour {item!r}; choose HH:00 from 00:00 to 23:00"
            )
        result.append(int(text[:2]))
    if not result:
        raise ConfigurationError("Select at least one ERA5-Land hour")
    return tuple(sorted(set(result)))


def _expected_times(start: date, end: date, hours: tuple[int, ...]) -> np.ndarray:
    days = np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D"))
    return (
        days.astype("datetime64[h]")[:, None]
        + np.asarray(hours, dtype="timedelta64[h]")[None, :]
    ).reshape(-1)


def _missing_summary(missing: np.ndarray) -> str:
    labels = [np.datetime_as_string(value, unit="h") for value in missing[:8]]
    suffix = f" and {missing.size - 8:,} more" if missing.size > 8 else ""
    return ", ".join(labels) + suffix


def iter_month_blocks(
    times: np.ndarray,
    cells: int,
    *,
    maximum_bytes: int = MAX_READ_BLOCK_BYTES,
) -> Iterator[np.ndarray]:
    """Yield calendar-month selections, splitting only to bound decoded memory."""

    bytes_per_step = max(1, int(cells) * np.dtype("float32").itemsize)
    maximum_steps = max(1, int(maximum_bytes) // bytes_per_step)
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
        raise DownloadError(f"ERA5-Land store has invalid {name} coordinates")
    if values.size == 1:
        return 0.1
    differences = np.diff(values)
    resolution = float(np.median(differences))
    if resolution <= 0 or not np.allclose(differences, resolution, atol=1e-8):
        raise DownloadError(f"ERA5-Land store has a non-regular {name} grid")
    return resolution


@dataclass(frozen=True)
class Selection:
    times: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    latitude_indices: np.ndarray
    longitude_parts: tuple[np.ndarray, ...]
    chunking: str
    coverage_start: np.datetime64
    coverage_end: np.datetime64
    warnings: tuple[str, ...]


class Era5LandProvider(Provider):
    spec = DatasetSpec(
        id="era5_land",
        title="ERA5-Land hourly ARCO",
        summary="Hourly 0.1 degree land reanalysis read directly from ECMWF ARCO Zarr stores.",
        group="Climate",
        variables=VARIABLES,
        credentials=(
            CredentialField(
                "cds_api_key",
                "CDS API key",
                environment=("CDSAPI_KEY",),
                description="Your own CDS token; ~/.cdsapirc is also supported.",
            ),
        ),
        options=(
            OptionSpec(
                "hours",
                "UTC hours",
                default=HOURS,
                choices=HOURS,
                required=True,
                multiple=True,
            ),
            OptionSpec(
                "chunking",
                "Zarr chunking",
                default="auto",
                choices=("auto", "geo", "time"),
                description="Auto uses time chunks for the provider's monthly retrieval blocks.",
            ),
        ),
        source_url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data",
        citation="Muñoz-Sabater et al. (2021), ERA5-Land: a state-of-the-art global reanalysis dataset for land applications.",
        license="Copernicus Licence to Use Copernicus Products",
    )

    def __init__(self) -> None:
        self._resolved: dict[str, tuple[float, Selection]] = {}

    def _open_dataset(self, url: str, api_key: str) -> Any:
        return open_arco_dataset(url, api_key)

    def _chunking(self, request: DownloadRequest) -> str:
        value = str(self.option(request, "chunking"))
        return "time" if value == "auto" else value

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        _requested_hours(self.option(request, "hours"))
        resolve_cds_api_key(request.credentials)

    @staticmethod
    def _spatial_indices(
        request: DownloadRequest, latitudes: np.ndarray, longitudes: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        lat_order = np.argsort(latitudes)
        sorted_latitudes = latitudes[lat_order]
        latitude_indices = lat_order[
            (sorted_latitudes >= request.bbox.south) & (sorted_latitudes <= request.bbox.north)
        ]
        if latitude_indices.size == 0:
            raise ConfigurationError("The ERA5-Land extent contains no 0.1 degree latitude cells")

        longitude_parts: list[np.ndarray] = []
        output_longitudes: list[np.ndarray] = []
        for part, segment in enumerate(request.bbox.segments()):
            indices = np.flatnonzero(
                (longitudes >= segment.west) & (longitudes <= segment.east)
            )
            indices = indices[np.argsort(longitudes[indices])]
            if indices.size == 0:
                raise ConfigurationError("The ERA5-Land extent contains no 0.1 degree longitude cells")
            values = longitudes[indices].astype("float64", copy=True)
            if request.bbox.crosses_antimeridian and part:
                values += 360.0
            longitude_parts.append(indices)
            output_longitudes.append(values)
        return (
            latitude_indices,
            tuple(longitude_parts),
            latitudes[latitude_indices],
            np.concatenate(output_longitudes),
        )

    def _selection(self, request: DownloadRequest) -> Selection:
        assert request.time is not None
        api_key = resolve_cds_api_key(request.credentials)
        chunking = self._chunking(request)
        groups = tuple(dict.fromkeys(GROUP_BY_VARIABLE[name] for name in request.variables))
        available: np.ndarray | None = None
        latitudes: np.ndarray | None = None
        longitudes: np.ndarray | None = None
        for group in groups:
            dataset = self._open_dataset(store_url(group, chunking), api_key)
            try:
                missing_variables = [
                    name
                    for name in request.variables
                    if GROUP_BY_VARIABLE[name] == group and name not in dataset
                ]
                if missing_variables:
                    raise DownloadError(
                        f"ERA5-Land store {group.id} is missing: {', '.join(missing_variables)}"
                    )
                raw_times = np.asarray(dataset["time"].values).astype("datetime64[h]")
                current = np.unique(raw_times[~np.isnat(raw_times)])
                if current.size == 0:
                    raise DownloadError(f"ERA5-Land store {group.id} has no valid times")
                if current.size != raw_times.size:
                    raise DownloadError(
                        f"ERA5-Land store {group.id} has duplicate or invalid times"
                    )
                available = (
                    current
                    if available is None
                    else np.intersect1d(available, current, assume_unique=True)
                )
                current_latitudes = np.asarray(
                    dataset["latitude"].values, dtype="float64"
                )
                current_longitudes = np.asarray(
                    dataset["longitude"].values, dtype="float64"
                )
                if latitudes is None:
                    latitudes = current_latitudes
                    longitudes = current_longitudes
                elif not (
                    np.array_equal(latitudes, current_latitudes)
                    and np.array_equal(longitudes, current_longitudes)
                ):
                    raise DownloadError("Selected ERA5-Land stores use different grids")
            except (KeyError, TypeError, ValueError) as exc:
                raise DownloadError(f"ERA5-Land store {group.id} metadata is invalid") from exc
            finally:
                close = getattr(dataset, "close", None)
                if close:
                    close()

        assert available is not None and latitudes is not None and longitudes is not None
        if available.size == 0:
            raise DownloadError("Selected ERA5-Land variable groups have no common timestamps")
        _regular(np.sort(latitudes), "latitude")
        _regular(np.sort(longitudes), "longitude")
        coverage_start, coverage_end = available[0], available[-1]
        start = (
            coverage_start.astype("datetime64[D]").astype(date)
            if request.time.start.lower() == "earliest"
            else _parse_date(request.time.start, end=False)
        )
        end = (
            coverage_end.astype("datetime64[D]").astype(date)
            if request.time.end.lower() == "latest"
            else _parse_date(request.time.end, end=True)
        )
        if start > end:
            raise ConfigurationError("ERA5-Land start date must not be after end date")
        expected = _expected_times(start, end, _requested_hours(self.option(request, "hours")))
        selected = np.intersect1d(expected, available, assume_unique=True)
        missing = expected[~np.isin(expected, selected, assume_unique=True)]
        warnings: tuple[str, ...] = ()
        if missing.size:
            summary = _missing_summary(missing)
            if not request.allow_partial:
                raise ConfigurationError(
                    f"ERA5-Land does not contain {missing.size:,} requested timestamp(s): {summary}"
                )
            warnings = (
                f"Omitting {missing.size:,} unavailable ERA5-Land timestamp(s): "
                f"{summary}.",
            )
        if selected.size == 0:
            raise DownloadError("No requested ERA5-Land timestamps are available")
        (
            latitude_indices,
            longitude_parts,
            output_latitudes,
            output_longitudes,
        ) = self._spatial_indices(request, latitudes, longitudes)
        _regular(output_latitudes, "selected latitude")
        _regular(output_longitudes, "selected longitude")
        return Selection(
            times=selected,
            latitudes=output_latitudes,
            longitudes=output_longitudes,
            latitude_indices=latitude_indices,
            longitude_parts=longitude_parts,
            chunking=chunking,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            warnings=warnings,
        )

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        selection = self._selection(request)
        self._resolved.clear()
        self._resolved[request.fingerprint] = (time.monotonic(), selection)
        cells = selection.latitudes.size * selection.longitudes.size
        days = np.unique(selection.times.astype("datetime64[D]")).size
        read_blocks = sum(1 for _ in iter_month_blocks(selection.times, cells))
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=int(days),
            slice_count=int(selection.times.size * len(request.variables)),
            estimated_bytes=int(cells * selection.times.size * len(request.variables) * 4),
            details={
                "available": [
                    np.datetime_as_string(selection.coverage_start, unit="h"),
                    np.datetime_as_string(selection.coverage_end, unit="h"),
                ],
                "hourly_steps": int(selection.times.size),
                "monthly_read_blocks_per_variable": read_blocks,
                "maximum_decoded_block_bytes": MAX_READ_BLOCK_BYTES,
                "grid_shape": [int(selection.latitudes.size), int(selection.longitudes.size)],
                "resolution_degrees": 0.1,
                "chunking": selection.chunking,
                "groups": sorted({GROUP_BY_VARIABLE[name].id for name in request.variables}),
            },
            warnings=selection.warnings,
        )

    @staticmethod
    def _read_block(
        variable: Any,
        time_indices: np.ndarray,
        latitude_indices: np.ndarray,
        longitude_parts: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        pieces = [
            np.asanyarray(
                variable.isel(
                    time=time_indices,
                    latitude=latitude_indices,
                    longitude=longitude_indices,
                ).values
            )
            for longitude_indices in longitude_parts
        ]
        data = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=-1)
        if np.ma.isMaskedArray(data):
            data = data.filled(np.nan)
        return np.asarray(data, dtype="float32")

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
        api_key = resolve_cds_api_key(request.credentials)
        resolution = _regular(selection.longitudes, "selected longitude")
        latitude_resolution = _regular(selection.latitudes, "selected latitude")
        transform = from_origin(
            float(selection.longitudes[0] - resolution / 2),
            float(selection.latitudes[-1] + latitude_resolution / 2),
            resolution,
            latitude_resolution,
        )
        groups = tuple(dict.fromkeys(GROUP_BY_VARIABLE[name] for name in request.variables))
        total = int(selection.times.size * len(request.variables))
        current = 0
        for group in groups:
            cancel.check()
            dataset = self._open_dataset(store_url(group, selection.chunking), api_key)
            try:
                dataset_times = np.asarray(dataset["time"].values).astype("datetime64[h]")
                if dataset_times.size > 1 and np.any(
                    np.diff(dataset_times) <= np.timedelta64(0, "h")
                ):
                    raise DownloadError(f"ERA5-Land store {group.id} times are not ordered")
                group_variables = [
                    name for name in request.variables if GROUP_BY_VARIABLE[name] == group
                ]
                cells = selection.latitudes.size * selection.longitudes.size
                for name in group_variables:
                    attrs = dict(getattr(dataset[name], "attrs", {}))
                    variable_spec = VARIABLE_SPEC[name]
                    attrs.update(
                        long_name=str(attrs.get("long_name", variable_spec.title)),
                        units=str(attrs.get("units", variable_spec.units or "")),
                        standard_name=variable_spec.standard_name,
                        _FillValue=NODATA,
                        calendar="proleptic_gregorian",
                        arco_chunking=selection.chunking,
                        arco_read_block="calendar_month",
                    )
                    for block in iter_month_blocks(selection.times, cells):
                        cancel.check()
                        time_indices = np.searchsorted(dataset_times, block)
                        if np.any(time_indices >= dataset_times.size) or not np.array_equal(
                            dataset_times[time_indices], block
                        ):
                            raise DownloadError(
                                "ERA5-Land store changed during download near "
                                f"{np.datetime_as_string(block[0], unit='h')}"
                            )
                        first = np.datetime_as_string(block[0], unit="h")
                        last = np.datetime_as_string(block[-1], unit="h")
                        month = np.datetime_as_string(
                            block[0].astype("datetime64[M]"), unit="M"
                        )
                        progress(
                            ProgressEvent(
                                "download",
                                f"Reading ERA5-Land {name} {month} ({block.size} hours)",
                                current=current + int(block.size),
                                total=total,
                            )
                        )
                        try:
                            data = self._read_block(
                                dataset[name],
                                time_indices,
                                selection.latitude_indices,
                                selection.longitude_parts,
                            )
                        except Exception as exc:
                            raise DownloadError(
                                f"Could not read ERA5-Land {name} for {first}..{last}"
                            ) from exc
                        expected_shape = (
                            block.size,
                            selection.latitudes.size,
                            selection.longitudes.size,
                        )
                        if data.shape != expected_shape:
                            raise OutputError(
                                f"ERA5-Land {name} returned shape {data.shape}; expected {expected_shape}"
                            )
                        path = workspace / (
                            f"{safe_component(name)}_{safe_component(first)}_"
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
                            for band in range(block.size):
                                cancel.check()
                                layer = np.where(
                                    np.isfinite(data[band]), data[band], NODATA
                                ).astype("float32", copy=False)
                                target.write(np.flipud(layer), band + 1)
                        del data
                        try:
                            yield RasterBlock(
                                variable=name,
                                source=path,
                                times=tuple(
                                    timestamp.astype("datetime64[us]").astype(datetime)
                                    for timestamp in block
                                ),
                                attrs=attrs,
                            )
                            current += int(block.size)
                        finally:
                            path.unlink(missing_ok=True)
            finally:
                close = getattr(dataset, "close", None)
                if close:
                    close()

    def manifest_metadata(self, request: DownloadRequest, plan: DownloadPlan) -> dict[str, Any]:
        metadata = dict(super().manifest_metadata(request, plan))
        metadata.update(
            service="ECMWF ARCO Zarr beta",
            chunking=plan.details.get("chunking"),
            deaccumulated_variables=("tp", "ssrd", "strd"),
        )
        return metadata
