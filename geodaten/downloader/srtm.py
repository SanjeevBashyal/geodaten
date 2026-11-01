"""Anonymous NASA SRTMGL1 v3 tiles hosted by OpenTopography."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

from geodaten.common.errors import ConfigurationError, DownloadError
from geodaten.common.gdal import mosaic_to_cog, validate_raster
from geodaten.common.io import cache_root, make_session, remote_exists, stream_download
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

BASE_URL = (
    "https://opentopography.s3.sdsc.edu/raster/"
    "SRTM_GL1/SRTM_GL1_srtm"
)
MIN_LATITUDE = -56.0
MAX_LATITUDE = 60.0


def tile_label(latitude_south: int, longitude_west: int) -> str:
    latitude = f"{'N' if latitude_south >= 0 else 'S'}{abs(latitude_south):02d}"
    longitude = f"{'E' if longitude_west >= 0 else 'W'}{abs(longitude_west):03d}"
    return latitude + longitude


def intersecting_tiles(bbox: BBox) -> tuple[str, ...]:
    labels: list[str] = []
    for segment in bbox.segments():
        first_lon = math.floor(segment.west)
        last_lon = math.floor(math.nextafter(segment.east, -math.inf))
        first_lat = math.floor(segment.south)
        last_lat = math.floor(math.nextafter(segment.north, -math.inf))
        labels.extend(
            tile_label(latitude, longitude)
            for latitude in range(first_lat, last_lat + 1)
            for longitude in range(first_lon, last_lon + 1)
        )
    return tuple(dict.fromkeys(labels))


def tile_url(label: str) -> str:
    return f"{BASE_URL}/{label}.tif"


def _tile_available(session: object, label: str) -> bool:
    return remote_exists(session, tile_url(label), f"SRTM tile {label}")


class SrtmProvider(Provider):
    spec = DatasetSpec(
        id="srtm",
        title="NASA SRTMGL1 v3",
        summary=(
            "Void-filled one arc-second SRTM elevation from OpenTopography's "
            "anonymous object store, mosaicked and cropped to the requested extent."
        ),
        group="Terrain",
        variables=(
            VariableSpec(
                "elevation",
                "Land surface elevation",
                units="m",
                standard_name="surface_altitude",
            ),
        ),
        temporal=False,
        source_url="https://portal.opentopography.org/raster?jobId=rt1733102120551",
        citation=(
            "NASA JPL (2013), NASA Shuttle Radar Topography Mission Global "
            "1 arc second v3; distributed by OpenTopography."
        ),
        license="NASA policy; OpenTopography acknowledgement requested.",
    )

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        if request.bbox.crosses_antimeridian:
            raise ConfigurationError("SRTM subsets cannot cross the antimeridian")
        if request.bbox.south < MIN_LATITUDE or request.bbox.north > MAX_LATITUDE:
            raise ConfigurationError(
                f"SRTMGL1 coverage is limited to {MIN_LATITUDE:g} through "
                f"{MAX_LATITUDE:g} degrees latitude"
            )

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        tiles = intersecting_tiles(request.bbox)
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=1,
            slice_count=1,
            estimated_bytes=len(tiles) * 15 * 1024 * 1024,
            details={
                "tiles": tiles,
                "tile_count": len(tiles),
                "native_resolution": "1 arc-second",
                "version": "SRTMGL1.003",
                "source": "OpenTopography anonymous object storage",
            },
            warnings=(
                "Transfer estimate is approximate; ocean-only tiles may be absent.",
            ),
        )

    def slices(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Iterator[RasterSlice]:
        self.validate(request)
        labels = intersecting_tiles(request.bbox)
        tile_cache = cache_root(request) / "srtm" / "opentopography"
        tile_cache.mkdir(parents=True, exist_ok=True)
        sources: list[Path] = []
        session = make_session()
        try:
            for index, label in enumerate(labels, start=1):
                cancel.check()
                tile = tile_cache / f"{label}.tif"
                if validate_raster(tile, bands=1):
                    sources.append(tile)
                    progress(ProgressEvent("download", f"Using cached SRTM tile {label}"))
                    continue
                tile.unlink(missing_ok=True)
                if not _tile_available(session, label):
                    progress(ProgressEvent("warning", f"SRTM tile has no land data: {label}"))
                    continue
                progress(
                    ProgressEvent(
                        "download",
                        f"Downloading SRTM tile {label}",
                        current=index,
                        total=len(labels),
                    )
                )
                stream_download(
                    tile_url(label),
                    tile,
                    session=session,
                    progress=progress,
                    cancel=cancel,
                )
                if not validate_raster(tile, bands=1):
                    tile.unlink(missing_ok=True)
                    raise DownloadError(f"Downloaded SRTM tile is invalid: {label}")
                sources.append(tile)
        finally:
            session.close()

        if not sources:
            raise DownloadError("No SRTM tiles were available for the requested extent")
        local = workspace / "srtm" / "elevation.tif"
        local.parent.mkdir(parents=True, exist_ok=True)
        progress(ProgressEvent("transform", f"Mosaicking {len(sources)} SRTM tile(s)"))
        mosaic_to_cog(
            sources,
            local,
            bbox=request.bbox,
            metadata={"dataset": "SRTMGL1.003", "vertical_units": "m"},
            nodata=-32768,
            cancel=cancel,
        )
        try:
            yield RasterSlice(
                variable="elevation",
                source=local,
                attrs={
                    "long_name": "land surface elevation",
                    "standard_name": "surface_altitude",
                    "units": "m",
                    "_FillValue": -32768,
                    "source": "NASA SRTMGL1.003 via OpenTopography",
                },
            )
        finally:
            local.unlink(missing_ok=True)
