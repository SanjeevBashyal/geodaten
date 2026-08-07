"""GLC_FCS30D v2 annual land-cover subsets from range-readable Zenodo ZIPs."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import requests

from geodaten.common.errors import ConfigurationError, DownloadError
from geodaten.common.gdal import mosaic_to_cog, validate_raster
from geodaten.common.io import atomic_path, make_session
from geodaten.common.models import (
    BBox,
    DatasetSpec,
    DownloadPlan,
    DownloadRequest,
    RasterSlice,
    VariableSpec,
)
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent
from geodaten.downloader.base import Provider, require_module

ZENODO_RECORD_ID = 15063683
ZENODO_API = "https://zenodo.org/api/records/{record_id}"
HISTORICAL_YEARS = (1985, 1990, 1995)
ANNUAL_YEARS = tuple(range(2000, 2023))
AVAILABLE_YEARS = HISTORICAL_YEARS + ANNUAL_YEARS


def _year_bound(value: str, *, lower: bool) -> int:
    text = str(value).strip().lower()
    if text == "earliest":
        return AVAILABLE_YEARS[0]
    if text == "latest":
        return AVAILABLE_YEARS[-1]
    try:
        parts = text.split("-")
        if len(parts) == 1 and len(parts[0]) == 4:
            year = int(parts[0])
            date(year, 1, 1)
            return year
        if len(parts) == 2:
            year, month = map(int, parts)
            date(year, month, 1)
            return year
        if len(parts) == 3:
            parsed = date.fromisoformat(text)
            return parsed.year
        raise ValueError
    except ValueError as exc:
        which = "start" if lower else "end"
        raise ConfigurationError(f"Invalid GLC_FCS30D {which} time: {value!r}") from exc


def selected_years(start: str, end: str) -> tuple[int, ...]:
    first = _year_bound(start, lower=True)
    last = _year_bound(end, lower=False)
    if first > last:
        raise ConfigurationError(f"Start year {first} is later than end year {last}")
    years = tuple(year for year in AVAILABLE_YEARS if first <= year <= last)
    if not years:
        raise ConfigurationError(
            "No GLC_FCS30D epoch is available in the requested range; coverage is "
            "1985, 1990, 1995, and annually from 2000 through 2022"
        )
    return years


def _coordinate_label(value: int, positive: str, negative: str) -> str:
    return f"{positive if value >= 0 else negative}{abs(value)}"


def tile_label(longitude_west: int, latitude_north: int) -> str:
    return _coordinate_label(longitude_west, "E", "W") + _coordinate_label(
        latitude_north, "N", "S"
    )


def intersecting_tiles(bbox: BBox | Sequence[float]) -> tuple[tuple[int, int], ...]:
    """Return native 5-degree tiles, including western/antimeridian segments."""

    if not isinstance(bbox, BBox):
        bbox = BBox.from_values(bbox)
    result: list[tuple[int, int]] = []
    for segment in bbox.segments():
        first_lon = math.floor(segment.west / 5) * 5
        last_lon = math.floor(math.nextafter(segment.east, -math.inf) / 5) * 5
        top_lat = math.ceil(segment.north / 5) * 5
        bottom_top = math.floor(math.nextafter(segment.south, math.inf) / 5) * 5 + 5
        result.extend(
            (longitude, latitude)
            for longitude in range(first_lon, last_lon + 1, 5)
            for latitude in range(top_lat, bottom_top - 1, -5)
        )
    return tuple(dict.fromkeys(result))


def source_spec(year: int, longitude_west: int, latitude_north: int) -> tuple[str, int, int]:
    tile = tile_label(longitude_west, latitude_north)
    if year in HISTORICAL_YEARS:
        return (
            f"GLC_FCS30D_19852000_{tile}_5years_V1.1.tif",
            HISTORICAL_YEARS.index(year) + 1,
            len(HISTORICAL_YEARS),
        )
    if year in ANNUAL_YEARS:
        return (
            f"GLC_FCS30D_20002022_{tile}_Annual_V1.1.tif",
            year - 1999,
            len(ANNUAL_YEARS),
        )
    raise ConfigurationError(f"GLC_FCS30D has no epoch for {year}")


def archive_name(longitude_west: int) -> str:
    """Map a native tile west edge to its official two-column v2 archive."""

    if longitude_west >= 0:
        first = (longitude_west // 10) * 10
        pair = f"E{first}-E{first + 5}"
    else:
        magnitude = abs(longitude_west)
        first = ((magnitude - 1) // 10) * 10 + 5
        pair = f"W{first}-W{first + 5}"
    return f"GLC_FCS30D_19852022maps_{pair}.zip"


def _record_files(session: requests.Session) -> tuple[Mapping[str, Any], dict[str, str]]:
    try:
        response = session.get(ZENODO_API.format(record_id=ZENODO_RECORD_ID), timeout=60)
        response.raise_for_status()
        record = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise DownloadError(f"Could not read Zenodo record {ZENODO_RECORD_ID}") from exc
    if str(record.get("metadata", {}).get("version", "")).lower() != "v2":
        raise DownloadError(f"Zenodo record {ZENODO_RECORD_ID} is not GLC_FCS30D v2")
    files = {
        item["key"]: item.get("links", {}).get("self")
        for item in record.get("files", ())
        if item.get("key") and item.get("links", {}).get("self")
    }
    return record, files


def _member_lookup(remote: Any) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for info in remote.infolist():
        if info.is_dir():
            continue
        basename = Path(info.filename).name
        if basename in lookup:
            raise DownloadError(f"Duplicate member {basename!r} in GLC archive")
        lookup[basename] = info.filename
    return lookup


def _extract_member(
    remote: Any,
    member: str,
    destination: Path,
    *,
    expected_bands: int,
    progress: ProgressCallback,
    cancel: CancellationToken,
) -> Path:
    info = remote.getinfo(member)
    copied = 0
    with atomic_path(destination) as temporary:
        with remote.open(member) as source, temporary.open("wb") as target:
            while chunk := source.read(8 * 1024 * 1024):
                cancel.check()
                target.write(chunk)
                copied += len(chunk)
                progress(
                    ProgressEvent(
                        "download",
                        destination.name,
                        bytes_done=copied,
                        bytes_total=info.file_size,
                    )
                )
        if copied != info.file_size or not validate_raster(temporary, bands=expected_bands):
            raise DownloadError(f"Extracted GLC source failed validation: {destination.name}")
    return destination


class GlcFcs30dProvider(Provider):
    spec = DatasetSpec(
        id="glc_fcs30d",
        title="GLC_FCS30D v2",
        summary=(
            "Global 30 m fine-classification land-cover observations at historical "
            "epochs and annually from 2000 to 2022."
        ),
        variables=(
            VariableSpec(
                "land_cover",
                "Fine-classification land cover",
                units="1",
                description="Categorical GLC_FCS30D class code.",
            ),
        ),
        temporal=True,
        source_url=f"https://zenodo.org/records/{ZENODO_RECORD_ID}",
        citation="Zhang et al., GLC_FCS30D global 30 m land-cover dynamics.",
        license="CC BY 4.0",
    )

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        if request.bbox.crosses_antimeridian:
            raise ConfigurationError("GLC_FCS30D subsets cannot cross the antimeridian")
        assert request.time is not None
        selected_years(request.time.start, request.time.end)

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        assert request.time is not None
        years = selected_years(request.time.start, request.time.end)
        tiles = intersecting_tiles(request.bbox)
        archives = tuple(dict.fromkeys(archive_name(longitude) for longitude, _ in tiles))
        source_families = int(any(year in HISTORICAL_YEARS for year in years)) + int(
            any(year in ANNUAL_YEARS for year in years)
        )
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=len(years),
            slice_count=len(years),
            details={
                "years": years,
                "tiles": tuple(tile_label(*tile) for tile in tiles),
                "archives": archives,
                "source_tile_count": len(tiles) * source_families,
                "zenodo_record_id": ZENODO_RECORD_ID,
                "release": "v2",
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
        require_module("remotezip", "zenodo")
        from remotezip import RemoteZip

        assert request.time is not None
        years = selected_years(request.time.start, request.time.end)
        tiles = intersecting_tiles(request.bbox)
        source_dir = workspace / "glc_fcs30d" / "sources"
        slice_dir = workspace / "glc_fcs30d" / "slices"
        source_dir.mkdir(parents=True, exist_ok=True)
        slice_dir.mkdir(parents=True, exist_ok=True)

        required: dict[str, tuple[int, int]] = {}
        for year in years:
            for longitude, latitude in tiles:
                filename, _band, band_count = source_spec(year, longitude, latitude)
                required[filename] = (longitude, band_count)
        grouped: dict[str, list[tuple[str, int]]] = {}
        for filename, (longitude, band_count) in required.items():
            grouped.setdefault(archive_name(longitude), []).append((filename, band_count))

        session = make_session()
        try:
            _record, files = _record_files(session)
            for archive, members in grouped.items():
                cancel.check()
                url = files.get(archive)
                if not url:
                    raise DownloadError(
                        f"Official GLC_FCS30D v2 record does not contain {archive}"
                    )
                progress(ProgressEvent("download", f"Opening GLC archive {archive}"))
                try:
                    with RemoteZip(url, session=session) as remote:
                        lookup = _member_lookup(remote)
                        for filename, band_count in members:
                            cancel.check()
                            member = lookup.get(filename)
                            if member is None:
                                progress(
                                    ProgressEvent(
                                        "warning",
                                        f"GLC source has no land tile: {filename}",
                                    )
                                )
                                continue
                            _extract_member(
                                remote,
                                member,
                                source_dir / filename,
                                expected_bands=band_count,
                                progress=progress,
                                cancel=cancel,
                            )
                except DownloadError:
                    raise
                except Exception as exc:
                    raise DownloadError(f"Could not read GLC archive {archive}") from exc

            for index, year in enumerate(years, start=1):
                cancel.check()
                specs = tuple(
                    source_spec(year, longitude, latitude)
                    for longitude, latitude in tiles
                )
                bands = {band for _filename, band, _count in specs}
                if len(bands) != 1:
                    raise DownloadError(f"Inconsistent GLC source bands for {year}")
                sources = tuple(
                    source_dir / filename
                    for filename, _band, _count in specs
                    if (source_dir / filename).is_file()
                )
                if not sources:
                    raise DownloadError(
                        f"No GLC_FCS30D source data exist for the requested extent in {year}"
                    )
                local = slice_dir / f"land_cover_{year}.tif"
                progress(
                    ProgressEvent(
                        "transform",
                        f"Cropping GLC_FCS30D {year}",
                        current=index,
                        total=len(years),
                    )
                )
                mosaic_to_cog(
                    sources,
                    local,
                    bbox=request.bbox,
                    band=bands.pop(),
                    metadata={"dataset": "GLC_FCS30D", "year": year, "release": "v2"},
                    nodata=0,
                    cancel=cancel,
                )
                try:
                    yield RasterSlice(
                        variable="land_cover",
                        source=local,
                        time=date(year, 1, 1),
                        attrs={
                            "long_name": "GLC_FCS30D fine-classification land cover",
                            "units": "1",
                            "_FillValue": 0,
                            "source": f"GLC_FCS30D v2 Zenodo record {ZENODO_RECORD_ID}",
                        },
                    )
                finally:
                    local.unlink(missing_ok=True)
        finally:
            session.close()
