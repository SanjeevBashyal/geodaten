"""Sentinel-2 L2A and Landsat 9 C2 L2 scene windows from the Planetary Computer STAC API.

Both collections are published as per-band cloud-optimized GeoTIFFs, so one
search/sign/warp path serves both: only the requested extent is transferred.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import requests

from geodaten.common import gdal
from geodaten.common.auth import child_environment
from geodaten.common.errors import ConfigurationError, DownloadError
from geodaten.common.io import make_session, redacted_url, safe_component
from geodaten.common.models import (
    BBox,
    DatasetSpec,
    DownloadPlan,
    DownloadRequest,
    OptionSpec,
    OutputFormat,
    RasterSlice,
    VariableSpec,
)
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent
from geodaten.downloader.base import Provider

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SEARCH_URL = f"{STAC_URL}/search"
#: Container-scoped shared-access signatures for the public blob archives.
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token"

_PAGE_SIZE = 500
#: A guard against a paging loop; a request needing more pages is too broad to
#: download scene by scene anyway.
_MAX_PAGES = 200
_SEARCH_CACHE_SECONDS = 300.0
#: Renew a signature before GDAL can trip over its expiry mid-transfer.
_TOKEN_MARGIN = timedelta(minutes=5)
#: GDAL colour interpretations for a composite, in band order.
_COLOUR_INTERPRETATION = ("red", "green", "blue")


@dataclass(frozen=True)
class Band:
    """One selectable layer and the packing its archive documents."""

    asset: str
    title: str
    units: str = "1"
    scale: float | None = None
    offset: float | None = None
    description: str = ""
    #: Assets composited into one multiband image, in band order. Empty for an
    #: ordinary single-layer band.
    stack: tuple[str, ...] = ()

    @property
    def assets(self) -> tuple[str, ...]:
        return self.stack or (self.asset,)


def _parse_boundary(value: str, *, end: bool, earliest: date) -> date:
    normalized = str(value).strip().lower()
    if normalized == "earliest":
        return earliest
    if normalized == "latest":
        return date.today()
    if re.fullmatch(r"\d{4}", normalized):
        year = int(normalized)
        return date(year, 12, 31) if end else date(year, 1, 1)
    try:
        return date.fromisoformat(normalized)
    except ValueError as exc:
        raise ConfigurationError(
            f"Invalid temporal boundary {value!r}; use YYYY, YYYY-MM-DD, earliest, or latest"
        ) from exc


def resolve_date_window(request: DownloadRequest, earliest: date) -> tuple[date, date]:
    """Resolve request keywords to an inclusive acquisition interval."""

    if request.time is None:  # guarded by Provider.validate; useful to direct callers
        raise ConfigurationError("Imagery datasets require a temporal range")
    start = _parse_boundary(request.time.start, end=False, earliest=earliest)
    end = _parse_boundary(request.time.end, end=True, earliest=earliest)
    if start > end:
        raise ConfigurationError("Start date must be on or before end date")
    if end < earliest:
        raise ConfigurationError(f"The mission has no acquisitions before {earliest.isoformat()}")
    return start, end


def item_datetime(item: Mapping[str, Any]) -> datetime:
    """Return one scene's acquisition instant as naive UTC truncated to seconds."""

    properties = item.get("properties") or {}
    raw = properties.get("datetime") or properties.get("start_datetime")
    if not raw:
        raise DownloadError(f"STAC item {item.get('id')!r} carries no acquisition time")
    text = str(raw).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise DownloadError(f"STAC item {item.get('id')!r} has a non-ISO time {raw!r}") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed.replace(microsecond=0)


def group_by_date(items: Sequence[Mapping[str, Any]]) -> dict[date, list[Mapping[str, Any]]]:
    """Bucket scenes by acquisition date, in chronological order.

    Tiles of one overpass are what must be merged, and Landsat numbers adjacent
    WRS rows of a single pass seconds apart, so the timestamp cannot group them.
    """

    groups: dict[date, list[Mapping[str, Any]]] = {}
    for item in sorted(items, key=item_datetime):
        groups.setdefault(item_datetime(item).date(), []).append(item)
    return groups


def search(
    session: requests.Session,
    *,
    collection: str,
    bbox: BBox,
    start: date,
    end: date,
    max_cloud_cover: float | None,
    platform: str | None = None,
) -> list[dict[str, Any]]:
    """Return every matching STAC item, following the API's own paging links."""

    query: dict[str, Any] = {}
    if max_cloud_cover is not None:
        query["eo:cloud_cover"] = {"lte": max_cloud_cover}
    if platform is not None:
        query["platform"] = {"eq": platform}
    body: dict[str, Any] = {
        "collections": [collection],
        "bbox": list(bbox.as_tuple()),
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "limit": _PAGE_SIZE,
    }
    if query:
        body["query"] = query

    items: list[dict[str, Any]] = []
    url: str | None = SEARCH_URL
    payload: dict[str, Any] | None = body
    for _ in range(_MAX_PAGES):
        if url is None:
            break
        try:
            response = session.post(url, json=payload, timeout=(30, 120))
            response.raise_for_status()
            page = response.json()
        except (requests.RequestException, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            detail = f"HTTP {status}" if status is not None else type(exc).__name__
            raise DownloadError(
                f"Could not search {collection} at {redacted_url(url)} ({detail})"
            ) from exc
        items.extend(page.get("features") or ())
        url, payload = None, None
        for link in page.get("links") or ():
            if str(link.get("rel")) == "next":
                url = str(link.get("href"))
                payload = link.get("body") or body
                break
    else:
        raise DownloadError(
            f"The {collection} search exceeded {_MAX_PAGES} pages; narrow the extent or dates"
        )
    items.sort(key=item_datetime)
    return items


class SasTokens:
    """Cache one container signature per blob container for the run."""

    def __init__(self, session: requests.Session) -> None:
        self._session = session
        self._tokens: dict[tuple[str, str], tuple[datetime, str]] = {}

    @staticmethod
    def container(href: str) -> tuple[str, str]:
        parts = urlsplit(href)
        account = (parts.hostname or "").split(".")[0]
        container = parts.path.lstrip("/").split("/", 1)[0]
        if not account or not container:
            raise DownloadError(f"Cannot derive a signing container from {redacted_url(href)}")
        return account, container

    def token(self, href: str) -> str:
        key = self.container(href)
        cached = self._tokens.get(key)
        now = datetime.now(timezone.utc)
        if cached is not None and cached[0] - _TOKEN_MARGIN > now:
            return cached[1]
        url = f"{TOKEN_URL}/{key[0]}/{key[1]}"
        try:
            response = self._session.get(url, timeout=(30, 60))
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise DownloadError(f"Could not sign {key[0]}/{key[1]} for reading") from exc
        token = str(payload.get("token") or "")
        if not token:
            raise DownloadError(f"Signing service returned no token for {key[0]}/{key[1]}")
        expiry = str(payload.get("msft:expiry") or "").replace("Z", "+00:00")
        try:
            expires = datetime.fromisoformat(expiry)
        except ValueError:
            expires = now + timedelta(minutes=30)
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        self._tokens[key] = (expires, token)
        return token


def vsiaz(href: str) -> str:
    """Return the GDAL Azure path GDAL range-reads instead of downloading whole."""

    return "/vsiaz/" + urlsplit(href).path.lstrip("/")


def gdal_environment(account: str, token: str) -> dict[str, str]:
    """Hand the signature to GDAL out of band.

    A signed URL would otherwise reach the process arguments and GDAL's own
    console output, where it is both noisy and needlessly exposed.
    """

    return child_environment(
        {
            "AZURE_STORAGE_ACCOUNT": account,
            "AZURE_STORAGE_SAS_TOKEN": token,
            "GDAL_HTTP_TCP_KEEPALIVE": "YES",
        }
    )


class ImageryProvider(Provider):
    """Shared search-sign-warp behaviour for per-scene optical archives."""

    collection: str = ""
    platform: str | None = None
    first_date: date = date(1972, 7, 23)
    bands: Mapping[str, Band] = {}

    def __init__(self) -> None:
        self._items: dict[str, tuple[float, tuple[dict[str, Any], ...]]] = {}

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        if request.bbox.crosses_antimeridian:
            raise ConfigurationError(
                f"{self.spec.title} subsets cannot cross the antimeridian"
            )
        cloud = self.option(request, "max_cloud_cover")
        if cloud is not None and not 0 <= float(cloud) <= 100:
            raise ConfigurationError("max_cloud_cover must be between 0 and 100")
        resolve_date_window(request, self.first_date)

    def _nothing_matched(
        self,
        session: requests.Session,
        request: DownloadRequest,
        start: date,
        end: date,
        cloud: float | None,
    ) -> str:
        """Say why nothing matched: a cloud threshold is the usual reason.

        An empty result otherwise reads as missing archive coverage, which sends
        people looking for a second data source they do not need.
        """

        window = f"{start.isoformat()}..{end.isoformat()}"
        base = f"No {self.spec.title} scenes matched the requested extent over {window}"
        if cloud is None:
            return base
        limit = f"{base} with cloud cover at or below {float(cloud):g}%"
        try:
            covered = search(
                session,
                collection=self.collection,
                bbox=request.bbox,
                start=start,
                end=end,
                max_cloud_cover=None,
                platform=self.platform,
            )
        except DownloadError:
            return limit
        values = sorted(
            value
            for value in (
                (entry.get("properties") or {}).get("eo:cloud_cover") for entry in covered
            )
            if value is not None
        )
        if not values:
            return base
        listed = ", ".join(f"{value:g}%" for value in values[:5])
        more = ", ..." if len(values) > 5 else ""
        return (
            f"{limit}; {len(covered)} scene(s) do cover it, at {listed}{more}. "
            "Raise max_cloud_cover to include them."
        )

    def _items_for(self, request: DownloadRequest) -> tuple[dict[str, Any], ...]:
        """Search once and reuse the result for the matching plan and download."""

        cached = self._items.get(request.fingerprint)
        if cached is not None and monotonic() - cached[0] <= _SEARCH_CACHE_SECONDS:
            return cached[1]
        start, end = resolve_date_window(request, self.first_date)
        cloud = self.option(request, "max_cloud_cover")
        session = make_session()
        try:
            found = search(
                session,
                collection=self.collection,
                bbox=request.bbox,
                start=start,
                end=end,
                max_cloud_cover=cloud,
                platform=self.platform,
            )
            if not found:
                raise DownloadError(
                    self._nothing_matched(session, request, start, end, cloud)
                )
        finally:
            session.close()
        items = tuple(found)
        self._items.clear()
        self._items[request.fingerprint] = (monotonic(), items)
        return items

    def band_attrs(self, variable: str, item: Mapping[str, Any]) -> dict[str, Any]:
        band = self.bands[variable]
        attrs: dict[str, Any] = {"long_name": band.title, "units": band.units}
        if band.scale is not None:
            attrs["scale_factor"] = band.scale
            attrs["add_offset"] = band.offset if band.offset is not None else 0.0
        return attrs

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        start, end = resolve_date_window(request, self.first_date)
        items = self._items_for(request)
        groups = group_by_date(items)
        details: dict[str, Any] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "collection": self.collection,
            "catalogue": SEARCH_URL,
            "max_cloud_cover": self.option(request, "max_cloud_cover"),
            "first_acquisition": item_datetime(items[0]).isoformat(),
            "last_acquisition": item_datetime(items[-1]).isoformat(),
            "assets": tuple(
                asset for name in request.variables for asset in self.bands[name].assets
            ),
            "scene_count": len(items),
            # One merged output per date, however many tiles cover the extent.
            "dates": tuple(day.isoformat() for day in groups),
            "payload_source": "COG",
        }
        if len(items) <= 50:
            details["scenes"] = tuple(str(item.get("id")) for item in items)
        inventory = [
            {
                "id": str(item.get("id", "")),
                "datetime": item_datetime(item).isoformat(),
                "baseline": str(
                    (item.get("properties") or {}).get("s2:processing_baseline", "")
                ),
                "assets": {
                    asset: str(
                        ((item.get("assets") or {}).get(asset) or {}).get("href", "")
                    )
                    for name in request.variables
                    for asset in self.bands[name].assets
                },
            }
            for item in items
        ]
        source_identity = "stac:" + hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=len(groups),
            slice_count=len(groups) * len(request.variables),
            # Every scene is read through range requests, so the transferred
            # size follows the extent rather than the catalogued scene size.
            estimated_bytes=None,
            details=details,
            warnings=("Actual transfer size depends on GDAL COG range requests.",),
            source_identity=source_identity,
        )

    def _hrefs(
        self,
        group: Sequence[Mapping[str, Any]],
        asset: str,
        variable: str,
        request: DownloadRequest,
        progress: ProgressCallback,
    ) -> list[str]:
        """Collect one asset's href from every scene of a date group."""

        hrefs: list[str] = []
        for item in group:
            href = str(((item.get("assets") or {}).get(asset) or {}).get("href") or "")
            if href:
                hrefs.append(href)
                continue
            identifier = str(item.get("id", "scene"))
            if not request.allow_partial:
                raise DownloadError(f"Scene {identifier} has no {variable} asset ({asset})")
            progress(
                ProgressEvent("warning", f"Missing {variable} for scene {identifier}")
            )
        containers = {SasTokens.container(href)[1] for href in hrefs}
        if len(containers) > 1:
            # One gdalwarp call carries a single container signature, so a split
            # group would fail deep inside GDAL with an opaque auth error.
            raise DownloadError(
                f"{variable} for one date spans several storage containers "
                f"({', '.join(sorted(containers))}); cannot merge them in one pass"
            )
        return hrefs

    def _merge(
        self,
        hrefs: Sequence[str],
        destination: Path,
        request: DownloadRequest,
        tokens: SasTokens,
        cancel: CancellationToken,
        *,
        shape: tuple[int, int] | None = None,
        metadata: Mapping[str, Any] | None = None,
        scale: Any = None,
        offset: Any = None,
    ) -> Path:
        """Reproject and mosaic every scene of a group into one raster."""

        account, _ = SasTokens.container(hrefs[0])
        return gdal.warp_to_cog(
            [vsiaz(href) for href in hrefs],
            destination,
            bbox=request.bbox,
            shape=shape,
            metadata=metadata,
            scale=scale,
            offset=offset,
            env=gdal_environment(account, tokens.token(hrefs[0])),
            cancel=cancel,
        )

    def _compose(
        self,
        staging: Path,
        group: Sequence[Mapping[str, Any]],
        band: Band,
        variable: str,
        request: DownloadRequest,
        tokens: SasTokens,
        progress: ProgressCallback,
        cancel: CancellationToken,
        metadata: Mapping[str, Any],
    ) -> Path:
        """Stack several merged assets into one multiband image."""

        components: list[Path] = []
        shape: tuple[int, int] | None = None
        for position, asset in enumerate(band.stack):
            hrefs = self._hrefs(group, asset, variable, request, progress)
            if not hrefs:
                raise DownloadError(f"No {asset} asset available for {variable}")
            component = staging / f"component-{position}.tif"
            # The first merge sets the grid; the rest are pinned to it so the
            # bands stack exactly.
            self._merge(hrefs, component, request, tokens, cancel, shape=shape)
            if shape is None:
                size = gdal.info(component).get("size") or ()
                if len(size) != 2:
                    raise DownloadError(f"Could not read the merged size of {asset}")
                shape = (int(size[0]), int(size[1]))
            components.append(component)

        stacked = staging / "stacked.vrt"
        gdal.build_vrt(components, stacked, separate=True, cancel=cancel)
        coloured = staging / "composite.vrt"
        gdal.run(
            (
                "gdal_translate",
                "-q",
                "-of",
                "VRT",
                "-colorinterp",
                ",".join(_COLOUR_INTERPRETATION[: len(components)]),
                *(arg for key, value in metadata.items() if value is not None
                  for arg in ("-mo", f"{str(key).upper()}={value}")),
                str(stacked),
                str(coloured),
            ),
            cancel=cancel,
        )
        return coloured

    def _reject_mixed_baselines(
        self,
        group: Sequence[Mapping[str, Any]],
        day: date,
        variables: Sequence[str],
    ) -> None:
        baselines = {
            str((item.get("properties") or {}).get("s2:processing_baseline") or "unknown")
            for item in group
        }
        incompatible = any(
            len(
                {
                    self.band_attrs(variable, item).get("add_offset")
                    for item in group
                }
            )
            > 1
            for variable in variables
        )
        if incompatible:
            raise DownloadError(
                f"{day.isoformat()} has mixed Sentinel-2 processing baselines "
                f"{', '.join(sorted(baselines))}; they require different reflectance "
                "offsets and cannot be safely mosaicked"
            )

    def slices(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Iterator[RasterSlice]:
        self.validate(request)
        gdal.require_commands("gdalwarp", "gdal_translate", "gdalinfo", "gdalbuildvrt")
        groups = group_by_date(self._items_for(request))
        self._items.pop(request.fingerprint, None)
        session = make_session()
        tokens = SasTokens(session)
        slice_dir = workspace / "scenes"
        slice_dir.mkdir(parents=True, exist_ok=True)
        total = len(groups) * len(request.variables)
        current = 0
        produced = 0

        try:
            for day, group in groups.items():
                cancel.check()
                self._reject_mixed_baselines(group, day, request.variables)
                scenes = [str(item.get("id", "scene")) for item in group]
                for variable in request.variables:
                    cancel.check()
                    current += 1
                    band = self.bands[variable]
                    attrs = self.band_attrs(variable, group[0])
                    progress(
                        ProgressEvent(
                            "download",
                            f"Merging {variable} for {day.isoformat()} "
                            f"from {len(group)} scene(s)",
                            current=current,
                            total=total,
                        )
                    )
                    metadata = {
                        "dataset": self.spec.id,
                        "variable": variable,
                        "acquired": day.isoformat(),
                        "collection": self.collection,
                        "scenes": " ".join(scenes),
                        "scene_count": len(scenes),
                    }
                    with tempfile.TemporaryDirectory(
                        prefix="slice_", dir=slice_dir
                    ) as temporary:
                        staging = Path(temporary)
                        if band.stack:
                            source = self._compose(
                                staging,
                                group,
                                band,
                                variable,
                                request,
                                tokens,
                                progress,
                                cancel,
                                metadata,
                            )
                            selected: int | None = None
                        else:
                            hrefs = self._hrefs(
                                group, band.asset, variable, request, progress
                            )
                            if not hrefs:
                                continue
                            source = self._merge(
                                hrefs,
                                staging / "merged.tif",
                                request,
                                tokens,
                                cancel,
                                metadata=metadata,
                                scale=attrs.get("scale_factor"),
                                offset=attrs.get("add_offset"),
                            )
                            selected = 1
                        produced += 1
                        # The staged mosaic is request scratch data; the common
                        # writer consumes it before this directory is removed.
                        yield RasterSlice(
                            variable=variable,
                            source=source,
                            time=day,
                            band=selected,
                            attrs={
                                **attrs,
                                "source": (
                                    f"Planetary Computer {self.collection}; "
                                    f"{len(scenes)} scene(s)"
                                ),
                                "references": f"{STAC_URL}/collections/{self.collection}",
                            },
                        )
        finally:
            session.close()
        if not produced:
            raise DownloadError(f"No {self.spec.title} raster slices were produced")


def _variables(bands: Mapping[str, Band]) -> tuple[VariableSpec, ...]:
    return tuple(
        VariableSpec(
            identifier,
            band.title,
            units=band.units,
            description=band.description or f"STAC asset {band.asset}.",
        )
        for identifier, band in bands.items()
    )


def _cloud_option(default: float | None) -> OptionSpec:
    """Build the cloud filter, which is off wherever ``default`` is ``None``."""

    return OptionSpec(
        id="max_cloud_cover",
        title="Maximum cloud cover (%)",
        kind="number",
        default=default,
        description=(
            "Scene-level cloud percentage above which acquisitions are skipped. "
            "Leave empty to keep every scene regardless of cloud."
        ),
    )

#: Level-2A surface reflectance is packed with a fixed quantification value of
#: 10000; baseline 04.00 onwards additionally applies a -1000 DN offset.
_S2_SCALE = 1e-4
_S2_BASELINE_OFFSET = 4.0

SENTINEL2_BANDS: dict[str, Band] = {
    "b01": Band("B01", "Coastal aerosol (443 nm, 60 m)", scale=_S2_SCALE),
    "b02": Band("B02", "Blue (490 nm, 10 m)", scale=_S2_SCALE),
    "b03": Band("B03", "Green (560 nm, 10 m)", scale=_S2_SCALE),
    "b04": Band("B04", "Red (665 nm, 10 m)", scale=_S2_SCALE),
    "b05": Band("B05", "Red edge 1 (705 nm, 20 m)", scale=_S2_SCALE),
    "b06": Band("B06", "Red edge 2 (740 nm, 20 m)", scale=_S2_SCALE),
    "b07": Band("B07", "Red edge 3 (783 nm, 20 m)", scale=_S2_SCALE),
    "b08": Band("B08", "NIR (842 nm, 10 m)", scale=_S2_SCALE),
    "b8a": Band("B8A", "NIR narrow (865 nm, 20 m)", scale=_S2_SCALE),
    "b09": Band("B09", "Water vapour (945 nm, 60 m)", scale=_S2_SCALE),
    "b11": Band("B11", "SWIR 1 (1610 nm, 20 m)", scale=_S2_SCALE),
    "b12": Band("B12", "SWIR 2 (2190 nm, 20 m)", scale=_S2_SCALE),
    "scl": Band(
        "SCL",
        "Scene classification map (20 m)",
        description="Categorical L2A scene classes; unscaled flag values.",
    ),
    "rgb": Band(
        "B04",
        "True-colour image (red/green/blue, 10 m)",
        scale=_S2_SCALE,
        stack=("B04", "B03", "B02"),
        description=(
            "B04/B03/B02 stacked as one three-band image with red, green, and "
            "blue colour interpretation, for display in GIS."
        ),
    ),
}

#: Collection 2 Level-2 packing published by USGS for surface reflectance and
#: surface temperature.
LANDSAT9_BANDS: dict[str, Band] = {
    "coastal": Band("coastal", "Coastal/aerosol reflectance (443 nm)", scale=2.75e-05, offset=-0.2),
    "blue": Band("blue", "Blue reflectance (482 nm)", scale=2.75e-05, offset=-0.2),
    "green": Band("green", "Green reflectance (562 nm)", scale=2.75e-05, offset=-0.2),
    "red": Band("red", "Red reflectance (655 nm)", scale=2.75e-05, offset=-0.2),
    "nir08": Band("nir08", "NIR reflectance (865 nm)", scale=2.75e-05, offset=-0.2),
    "swir16": Band("swir16", "SWIR 1 reflectance (1609 nm)", scale=2.75e-05, offset=-0.2),
    "swir22": Band("swir22", "SWIR 2 reflectance (2201 nm)", scale=2.75e-05, offset=-0.2),
    "lwir11": Band(
        "lwir11",
        "Surface temperature (10.9 um)",
        units="K",
        scale=0.00341802,
        offset=149.0,
    ),
    "qa_pixel": Band(
        "qa_pixel",
        "Pixel quality assessment bitmask",
        description="Bit-packed cloud, shadow, snow, and water flags; unscaled.",
    ),
    "rgb": Band(
        "red",
        "True-colour image (red/green/blue, 30 m)",
        scale=2.75e-05,
        offset=-0.2,
        stack=("red", "green", "blue"),
        description=(
            "Red/green/blue stacked as one three-band image with matching "
            "colour interpretation, for display in GIS."
        ),
    ),
}


class Sentinel2Provider(ImageryProvider):
    collection = "sentinel-2-l2a"
    first_date = date(2015, 6, 27)
    bands = SENTINEL2_BANDS

    spec = DatasetSpec(
        id="sentinel2",
        title="Sentinel-2 MSI Level-2A",
        summary=(
            "Copernicus Sentinel-2 surface-reflectance scenes at 10-60 m. Only the "
            "requested window of each selected band is read from the official COGs."
        ),
        group="Imagery",
        variables=_variables(SENTINEL2_BANDS),
        formats=(OutputFormat.TIF,),
        options=(_cloud_option(20.0),),
        source_url=f"{STAC_URL}/collections/sentinel-2-l2a",
        citation="Copernicus Sentinel-2 MSI Level-2A, ESA",
        license="Copernicus free, full and open data policy",
    )

    def band_attrs(self, variable: str, item: Mapping[str, Any]) -> dict[str, Any]:
        attrs = super().band_attrs(variable, item)
        if "scale_factor" not in attrs:
            return attrs
        baseline = (item.get("properties") or {}).get("s2:processing_baseline")
        try:
            applies = float(str(baseline)) >= _S2_BASELINE_OFFSET
        except (TypeError, ValueError):
            # An unreadable baseline must not silently invent a radiometric shift.
            applies = False
        attrs["add_offset"] = -0.1 if applies else 0.0
        attrs["processing_baseline"] = str(baseline) if baseline is not None else ""
        return attrs


class Landsat9Provider(ImageryProvider):
    collection = "landsat-c2-l2"
    platform = "landsat-9"
    #: Landsat 9 launched 2021-09-27; the first Collection 2 Level-2 scenes are
    #: dated 2021-10-31.
    first_date = date(2021, 10, 31)
    bands = LANDSAT9_BANDS

    spec = DatasetSpec(
        id="landsat9",
        title="Landsat 9 Collection 2 Level-2",
        summary=(
            "USGS Landsat 9 OLI-2/TIRS-2 surface reflectance and surface temperature "
            "at 30 m, read window by window from the official COGs."
        ),
        group="Imagery",
        variables=_variables(LANDSAT9_BANDS),
        formats=(OutputFormat.TIF,),
        # Landsat 9 revisits every 16 days, so a cloud threshold discards scenes
        # that will not recur for weeks. Keep every acquisition unless the caller
        # asks for a limit.
        options=(_cloud_option(None),),
        source_url=f"{STAC_URL}/collections/landsat-c2-l2",
        citation="USGS Landsat 9 Collection 2 Level-2 Science Products",
        license="Public domain (USGS)",
    )


__all__: Sequence[str] = (
    "Band",
    "ImageryProvider",
    "group_by_date",
    "Landsat9Provider",
    "Sentinel2Provider",
    "SasTokens",
    "gdal_environment",
    "search",
    "vsiaz",
)
