"""ESA WorldCover 2021 v200 subsets from public Cloud Optimized GeoTIFFs."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

from geodaten.common.errors import ConfigurationError, DownloadError
from geodaten.common.gdal import warp_to_cog
from geodaten.common.io import make_session
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

BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
NODATA = 0
CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_vegetation",
    70: "snow_and_ice",
    80: "permanent_water_bodies",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_and_lichen",
}


def tile_label(latitude_south: int, longitude_west: int) -> str:
    latitude = f"{'N' if latitude_south >= 0 else 'S'}{abs(latitude_south):02d}"
    longitude = f"{'E' if longitude_west >= 0 else 'W'}{abs(longitude_west):03d}"
    return latitude + longitude


def intersecting_tiles(bbox: BBox) -> tuple[str, ...]:
    first_lon = math.floor(bbox.west / 3) * 3
    last_lon = math.floor(math.nextafter(bbox.east, -math.inf) / 3) * 3
    first_lat = math.floor(bbox.south / 3) * 3
    last_lat = math.floor(math.nextafter(bbox.north, -math.inf) / 3) * 3
    return tuple(
        tile_label(latitude, longitude)
        for latitude in range(first_lat, last_lat + 1, 3)
        for longitude in range(first_lon, last_lon + 1, 3)
    )


def tile_url(label: str) -> str:
    return f"{BASE_URL}/ESA_WorldCover_10m_2021_v200_{label}_Map.tif"


def remote_tile(label: str) -> str:
    return "/vsicurl?max_retry=5&retry_delay=2&list_dir=no&url=" + tile_url(label)


def _tile_available(session: Any, label: str) -> bool:
    response: Any = None
    try:
        response = session.head(tile_url(label), allow_redirects=True, timeout=(30, 90))
        if response.status_code in {405, 501}:
            response.close()
            response = session.get(
                tile_url(label),
                headers={"Range": "bytes=0-0"},
                stream=True,
                allow_redirects=True,
                timeout=(30, 90),
            )
        with response:
            if response.status_code == 404:
                return False
            response.raise_for_status()
            return True
    except requests.RequestException as exc:
        raise DownloadError(f"Could not check ESA WorldCover tile {label}") from exc


class EsaWorldCoverProvider(Provider):
    spec = DatasetSpec(
        id="esa_worldcover",
        title="ESA WorldCover 2021 v200",
        summary=(
            "Global 10 m land-cover classes read by spatial window from public "
            "Cloud Optimized GeoTIFF tiles."
        ),
        group="Land Use",
        variables=(
            VariableSpec(
                "land_cover",
                "Land-cover class",
                units="1",
                description="ESA WorldCover 2021 categorical class code.",
            ),
        ),
        temporal=False,
        source_url="https://esa-worldcover.org/en/data-access",
        citation="Zanaga et al. (2022), ESA WorldCover 10 m 2021 v200.",
        license="CC BY 4.0",
    )

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        if request.bbox.crosses_antimeridian:
            raise ConfigurationError("ESA WorldCover subsets cannot cross the antimeridian")

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        tiles = intersecting_tiles(request.bbox)
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=1,
            slice_count=1,
            details={
                "tiles": tiles,
                "tile_count": len(tiles),
                "native_resolution": "10 m",
                "year": 2021,
                "version": "v200",
                "access": "anonymous HTTP range reads",
            },
        )

    def slices(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Iterator[RasterSlice]:
        self.validate(request)
        sources: list[str] = []
        session = make_session()
        try:
            labels = intersecting_tiles(request.bbox)
            for index, label in enumerate(labels, 1):
                cancel.check()
                if _tile_available(session, label):
                    sources.append(remote_tile(label))
                else:
                    progress(ProgressEvent("warning", f"WorldCover tile has no land data: {label}"))
                progress(
                    ProgressEvent(
                        "download",
                        f"Resolving ESA WorldCover tile {label}",
                        current=index,
                        total=len(labels),
                    )
                )
        finally:
            session.close()
        if not sources:
            raise DownloadError("No ESA WorldCover tiles cover the requested extent")

        local = workspace / "esa_worldcover" / "land_cover.tif"
        local.parent.mkdir(parents=True, exist_ok=True)
        progress(ProgressEvent("transform", f"Reading {len(sources)} WorldCover tile(s)"))
        warp_to_cog(
            sources,
            local,
            bbox=request.bbox,
            metadata={
                "dataset": "ESA WorldCover",
                "year": 2021,
                "version": "v200",
                "license": "CC BY 4.0",
            },
            nodata=NODATA,
            cancel=cancel,
        )
        try:
            yield RasterSlice(
                variable="land_cover",
                source=local,
                attrs={
                    "long_name": "ESA WorldCover 2021 land-cover class",
                    "units": "1",
                    "_FillValue": NODATA,
                    "flag_values": tuple(CLASSES),
                    "flag_meanings": " ".join(CLASSES.values()),
                    "class_names": json.dumps(CLASSES, separators=(",", ":")),
                    "source": "ESA WorldCover 2021 v200",
                },
            )
        finally:
            local.unlink(missing_ok=True)
