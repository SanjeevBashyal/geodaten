"""NASA SRTMGL1 v3 elevation tiles authenticated with Earthdata Login."""

from __future__ import annotations

import math
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

from geodaten.common.auth import resolve_credentials, scoped_environment
from geodaten.common.errors import AuthenticationError, ConfigurationError, DownloadError
from geodaten.common.gdal import mosaic_to_cog
from geodaten.common.io import atomic_path, copy_stream, stream_download
from geodaten.common.models import (
    BBox,
    CredentialField,
    DatasetSpec,
    DownloadPlan,
    DownloadRequest,
    RasterSlice,
    VariableSpec,
)
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent
from geodaten.downloader.base import Provider, require_module

BASE_URL = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003"
MIN_LATITUDE = -56.0
MAX_LATITUDE = 60.0


def tile_label(latitude_south: int, longitude_west: int) -> str:
    """Return the lower-left-corner SRTM tile identifier."""

    latitude = f"{'N' if latitude_south >= 0 else 'S'}{abs(latitude_south):02d}"
    longitude = f"{'E' if longitude_west >= 0 else 'W'}{abs(longitude_west):03d}"
    return latitude + longitude


def intersecting_tiles(bbox: BBox) -> tuple[str, ...]:
    """Compute every 1-degree tile intersecting a :class:`BBox`."""

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
    granule = f"{label}.SRTMGL1.hgt"
    return f"{BASE_URL}/{granule}/{granule}.zip"


@contextmanager
def _earthdata_session(request: DownloadRequest) -> Iterator[Any]:
    """Create a non-interactive session from supplied/env credentials or .netrc."""

    require_module("earthaccess", "earthdata")
    import earthaccess

    credentials = resolve_credentials(SrtmProvider.spec.credentials, request.credentials)
    username = credentials.get("earthdata_username")
    password = credentials.get("earthdata_password")
    token = credentials.get("earthdata_token")
    if not token and bool(username) != bool(password):
        raise AuthenticationError("Earthdata username and password must be supplied together")

    environment: dict[str, str] = {}
    strategy = "netrc"
    if token:
        environment["EARTHDATA_TOKEN"] = token
        strategy = "environment"
    elif username and password:
        environment.update(
            {"EARTHDATA_USERNAME": username, "EARTHDATA_PASSWORD": password}
        )
        strategy = "environment"

    try:
        with scoped_environment(environment):
            auth = earthaccess.Auth().login(strategy=strategy, persist=False)
        if not auth.authenticated:
            raise AuthenticationError("Earthdata authentication was not accepted")
        session = auth.get_session()
        session.headers.setdefault("User-Agent", "geodaten/0.1")
    except AuthenticationError:
        raise
    except Exception as exc:
        raise AuthenticationError(
            "Earthdata authentication failed; check your token, credentials, or .netrc"
        ) from exc
    try:
        yield session
    finally:
        session.close()


def _extract_hgt(
    archive_path: Path,
    destination: Path,
    *,
    expected_stem: str,
    cancel: CancellationToken,
) -> Path:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            candidates = tuple(
                name
                for name in archive.namelist()
                if not name.endswith("/") and Path(name).suffix.lower() == ".hgt"
            )
            member = next(
                (name for name in candidates if Path(name).stem == expected_stem),
                candidates[0] if len(candidates) == 1 else None,
            )
            if member is None:
                raise DownloadError(f"SRTM archive {archive_path.name} has no expected HGT tile")
            info = archive.getinfo(member)
            with atomic_path(destination) as temporary:
                with archive.open(member) as source, temporary.open("wb") as target:
                    copied = copy_stream(source, target, cancel=cancel)
                if copied != info.file_size:
                    raise DownloadError(f"Incomplete SRTM tile in {archive_path.name}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise DownloadError(f"Invalid SRTM archive {archive_path.name}") from exc
    return destination


def _tile_available(session: Any, label: str) -> bool:
    """Return false only for a confirmed absent granule.

    Authentication, server, and transport failures remain hard errors; otherwise
    they could be mistaken for normal ocean gaps.
    """

    url = tile_url(label)
    response: Any = None
    try:
        response = session.head(url, allow_redirects=True, timeout=(30.0, 90.0))
        if response.status_code in {405, 501}:
            response.close()
            response = session.get(
                url,
                headers={"Range": "bytes=0-0"},
                stream=True,
                allow_redirects=True,
                timeout=(30.0, 90.0),
            )
        with response:
            if response.status_code == 404:
                return False
            if response.status_code in {401, 403}:
                raise AuthenticationError(
                    "Earthdata did not authorize access to SRTMGL1.003"
                )
            response.raise_for_status()
            return True
    except AuthenticationError:
        raise
    except requests.RequestException as exc:
        raise DownloadError(f"Could not check availability of SRTM tile {label}") from exc


class SrtmProvider(Provider):
    spec = DatasetSpec(
        id="srtm",
        title="NASA SRTMGL1 v3",
        summary=(
            "Void-filled one arc-second Shuttle Radar Topography Mission elevation "
            "mosaicked and cropped to the requested extent."
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
        credentials=(
            CredentialField(
                "earthdata_username",
                "Earthdata username",
                environment=("EARTHDATA_USERNAME",),
                required=False,
                secret=False,
                description="Optional when an Earthdata token or ~/.netrc is configured.",
            ),
            CredentialField(
                "earthdata_password",
                "Earthdata password",
                environment=("EARTHDATA_PASSWORD",),
                required=False,
                description="Optional when an Earthdata token or ~/.netrc is configured.",
            ),
            CredentialField(
                "earthdata_token",
                "Earthdata token",
                environment=("EARTHDATA_TOKEN",),
                required=False,
                description="Optional alternative to username/password or ~/.netrc.",
            ),
        ),
        source_url="https://lpdaac.usgs.gov/products/srtmgl1v003/",
        citation="NASA JPL (2013), NASA Shuttle Radar Topography Mission Global 1 arc second v3.",
        license="NASA Earth Science Data and Information Policy",
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
        credentials = resolve_credentials(self.spec.credentials, request.credentials)
        token = credentials.get("earthdata_token")
        if not token and bool(credentials.get("earthdata_username")) != bool(
            credentials.get("earthdata_password")
        ):
            raise AuthenticationError("Earthdata username and password must be supplied together")

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
            },
            warnings=(
                "Transfer estimate is approximate; ocean-only granules may be absent.",
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
        source_dir = workspace / "srtm" / "tiles"
        source_dir.mkdir(parents=True, exist_ok=True)
        sources: list[Path] = []
        with _earthdata_session(request) as session:
            for index, label in enumerate(labels, start=1):
                cancel.check()
                progress(
                    ProgressEvent(
                        "download",
                        f"Downloading SRTM tile {label}",
                        current=index,
                        total=len(labels),
                    )
                )
                archive = source_dir / f"{label}.zip"
                if not _tile_available(session, label):
                    progress(
                        ProgressEvent(
                            "warning",
                            f"SRTM tile has no land granule: {label}",
                        )
                    )
                    continue
                try:
                    stream_download(
                        tile_url(label),
                        archive,
                        session=session,
                        progress=progress,
                        cancel=cancel,
                    )
                    hgt = source_dir / f"{label}.hgt"
                    _extract_hgt(
                        archive,
                        hgt,
                        expected_stem=f"{label}.SRTMGL1",
                        cancel=cancel,
                    )
                    sources.append(hgt)
                finally:
                    archive.unlink(missing_ok=True)

        if not sources:
            raise DownloadError("No SRTM tiles were available for the requested extent")
        cancel.check()
        local = workspace / "srtm" / "elevation.tif"
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
                    "source": "NASA SRTMGL1.003",
                },
            )
        finally:
            local.unlink(missing_ok=True)
