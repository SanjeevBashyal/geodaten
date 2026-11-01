"""HydroSHEDS v1.1 core rasters from anonymous 10-degree tiles."""

from __future__ import annotations

import math
import re
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import rasterio
import requests

from geodaten.common.errors import ConfigurationError, DownloadError
from geodaten.common.gdal import mosaic_to_cog, validate_raster
from geodaten.common.io import atomic_path, cache_root, copy_stream, make_session, stream_download
from geodaten.common.models import (
    BBox,
    DatasetSpec,
    DownloadPlan,
    DownloadRequest,
    RasterSlice,
    VariableSpec,
)
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent
from geodaten.downloader.base import Provider

CATALOG_URL = "https://f000.backblazeb2.com/file/hydrosheds-v1-{code}/{code}_files.json"
MIN_LATITUDE = -56.0
MAX_LATITUDE = 60.0

LAYERS: dict[str, tuple[str, VariableSpec]] = {
    "void_filled_dem": (
        "dem",
        VariableSpec("void_filled_dem", "Void-filled DEM", "m", "surface_altitude"),
    ),
    "conditioned_dem": (
        "con",
        VariableSpec("conditioned_dem", "Hydrologically conditioned DEM", "m"),
    ),
    "flow_direction": (
        "dir",
        VariableSpec("flow_direction", "D8 flow direction", "1"),
    ),
    "flow_accumulation": (
        "acc",
        VariableSpec("flow_accumulation", "Flow accumulation", "cells"),
    ),
}


def tile_label(latitude_south: int, longitude_west: int) -> str:
    latitude = f"{'n' if latitude_south >= 0 else 's'}{abs(latitude_south):02d}"
    longitude = f"{'e' if longitude_west >= 0 else 'w'}{abs(longitude_west):03d}"
    return latitude + longitude


def intersecting_tiles(bbox: BBox) -> tuple[str, ...]:
    first_lon = math.floor(bbox.west / 10) * 10
    last_lon = math.floor(math.nextafter(bbox.east, -math.inf) / 10) * 10
    first_lat = math.floor(bbox.south / 10) * 10
    last_lat = math.floor(math.nextafter(bbox.north, -math.inf) / 10) * 10
    return tuple(
        tile_label(latitude, longitude)
        for latitude in range(first_lat, last_lat + 1, 10)
        for longitude in range(first_lon, last_lon + 1, 10)
    )


def catalog_urls(payload: Sequence[Mapping[str, Any]], code: str) -> dict[str, str]:
    pattern = re.compile(rf"^(?P<label>[ns]\d{{2}}[ew]\d{{3}})_{code}\.zip$", re.I)
    prefix = f"/file/hydrosheds-v1-{code}/"
    urls: dict[str, str] = {}
    for node in payload:
        attributes = node.get("a_attr")
        href = attributes.get("href") if isinstance(attributes, Mapping) else None
        if not isinstance(href, str):
            continue
        parsed = urlsplit(href)
        match = pattern.match(Path(parsed.path).name)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "data.hydrosheds.org"
            or not parsed.path.startswith(prefix)
            or match is None
        ):
            continue
        urls[match.group("label").lower()] = href
    return urls


def _extract_tile(
    archive_path: Path,
    destination: Path,
    *,
    label: str,
    code: str,
    cancel: CancellationToken,
) -> Path:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            expected = f"{label}_{code}"
            candidates = [
                item
                for item in archive.infolist()
                if not item.is_dir()
                and Path(item.filename).suffix.lower() in {".tif", ".tiff"}
                and Path(item.filename).stem.lower() == expected
            ]
            if len(candidates) != 1:
                raise DownloadError(f"HydroSHEDS archive has no unique {expected}.tif")
            info = candidates[0]
            with atomic_path(destination) as temporary:
                with archive.open(info) as source, temporary.open("wb") as target:
                    copied = copy_stream(source, target, cancel=cancel)
                if copied != info.file_size:
                    raise DownloadError(f"Incomplete HydroSHEDS tile {expected}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise DownloadError(f"Invalid HydroSHEDS archive {archive_path.name}") from exc
    return destination


class HydroShedsProvider(Provider):
    spec = DatasetSpec(
        id="hydrosheds",
        title="HydroSHEDS v1.1 core terrain",
        summary=(
            "Three arc-second void-filled and conditioned elevation, D8 flow "
            "direction, and flow accumulation cropped from official 10-degree tiles."
        ),
        group="Terrain",
        variables=tuple(item[1] for item in LAYERS.values()),
        temporal=False,
        source_url="https://www.hydrosheds.org/hydrosheds-core-downloads",
        citation="Lehner, Verdin and Jarvis (2008), New global hydrography derived from spaceborne elevation data.",
        license="HydroSHEDS core products licence; attribution required.",
    )

    def __init__(self) -> None:
        self._catalogs: dict[str, dict[str, str]] = {}
        self._resolved: dict[str, tuple[dict[str, tuple[tuple[str, str], ...]], tuple[str, ...]]] = {}

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        if request.bbox.crosses_antimeridian:
            raise ConfigurationError("HydroSHEDS 3s subsets cannot cross the antimeridian")
        if request.bbox.south < MIN_LATITUDE or request.bbox.north > MAX_LATITUDE:
            raise ConfigurationError(
                f"HydroSHEDS 3s coverage is limited to {MIN_LATITUDE:g} through "
                f"{MAX_LATITUDE:g} degrees latitude"
            )

    def _catalog(self, code: str, session: requests.Session) -> dict[str, str]:
        if code in self._catalogs:
            return self._catalogs[code]
        try:
            response = session.get(CATALOG_URL.format(code=code), timeout=(20, 90))
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise DownloadError(f"Could not read HydroSHEDS {code} tile catalog") from exc
        if not isinstance(payload, list):
            raise DownloadError(f"HydroSHEDS {code} tile catalog is invalid")
        urls = catalog_urls(payload, code)
        if not urls:
            raise DownloadError(f"HydroSHEDS {code} tile catalog contains no usable URLs")
        self._catalogs[code] = urls
        return urls

    def _resolve(
        self, request: DownloadRequest
    ) -> tuple[dict[str, tuple[tuple[str, str], ...]], tuple[str, ...]]:
        cached = self._resolved.get(request.fingerprint)
        if cached is not None:
            return cached
        labels = intersecting_tiles(request.bbox)
        resolved: dict[str, tuple[tuple[str, str], ...]] = {}
        warnings: list[str] = []
        session = make_session()
        try:
            for variable in request.variables:
                code = LAYERS[variable][0]
                catalog = self._catalog(code, session)
                available = tuple((label, catalog[label]) for label in labels if label in catalog)
                missing = [label for label in labels if label not in catalog]
                if not available:
                    raise DownloadError(
                        f"No HydroSHEDS {code} tiles cover the requested extent"
                    )
                if missing:
                    warnings.append(
                        f"HydroSHEDS {code} has no published tile(s): {', '.join(missing)}."
                    )
                resolved[variable] = available
        finally:
            session.close()
        result = resolved, tuple(warnings)
        self._resolved.clear()
        self._resolved[request.fingerprint] = result
        return result

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        resolved, warnings = self._resolve(request)
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=1,
            slice_count=len(request.variables),
            estimated_bytes=sum(len(items) for items in resolved.values()) * 150 * 1024 * 1024,
            details={
                "version": "1.1",
                "resolution": "3 arc-seconds",
                "tiles": {
                    variable: [label for label, _url in items]
                    for variable, items in resolved.items()
                },
            },
            warnings=warnings,
        )

    def slices(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Iterator[RasterSlice]:
        self.validate(request)
        cached = self._resolved.pop(request.fingerprint, None)
        if cached is None:
            resolved, warnings = self._resolve(request)
            self._resolved.pop(request.fingerprint, None)
        else:
            resolved, warnings = cached
        for warning in warnings:
            progress(ProgressEvent("warning", warning))
        root = cache_root(request) / "hydrosheds" / "v1.1-3s"
        root.mkdir(parents=True, exist_ok=True)
        session = make_session()
        try:
            for variable in request.variables:
                code = LAYERS[variable][0]
                sources: list[Path] = []
                for index, (label, url) in enumerate(resolved[variable], 1):
                    cancel.check()
                    tile = root / code / f"{label}_{code}.tif"
                    if validate_raster(tile, bands=1):
                        sources.append(tile)
                        continue
                    tile.unlink(missing_ok=True)
                    tile.parent.mkdir(parents=True, exist_ok=True)
                    archive = workspace / f"{label}_{code}.zip"
                    progress(
                        ProgressEvent(
                            "download",
                            f"Downloading HydroSHEDS {code} tile {label}",
                            current=index,
                            total=len(resolved[variable]),
                        )
                    )
                    try:
                        stream_download(
                            url,
                            archive,
                            session=session,
                            progress=progress,
                            cancel=cancel,
                        )
                        _extract_tile(
                            archive,
                            tile,
                            label=label,
                            code=code,
                            cancel=cancel,
                        )
                    finally:
                        archive.unlink(missing_ok=True)
                    if not validate_raster(tile, bands=1):
                        tile.unlink(missing_ok=True)
                        raise DownloadError(f"Extracted HydroSHEDS tile is invalid: {label}")
                    sources.append(tile)

                local = workspace / "hydrosheds" / f"{code}.tif"
                local.parent.mkdir(parents=True, exist_ok=True)
                progress(
                    ProgressEvent(
                        "transform",
                        f"Mosaicking {len(sources)} HydroSHEDS {code} tile(s)",
                    )
                )
                mosaic_to_cog(
                    sources,
                    local,
                    bbox=request.bbox,
                    metadata={"dataset": "HydroSHEDS v1.1", "layer": code},
                    cancel=cancel,
                )
                try:
                    with rasterio.open(local) as dataset:
                        nodata = dataset.nodata
                    spec = LAYERS[variable][1]
                    attrs: dict[str, Any] = {
                        "long_name": spec.title,
                        "units": spec.units or "1",
                        "source": "HydroSHEDS v1.1",
                    }
                    if spec.standard_name:
                        attrs["standard_name"] = spec.standard_name
                    if nodata is not None:
                        attrs["_FillValue"] = nodata
                    yield RasterSlice(variable=variable, source=local, attrs=attrs)
                finally:
                    local.unlink(missing_ok=True)
        finally:
            session.close()
