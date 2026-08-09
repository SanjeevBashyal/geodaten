"""NASA MODIS Collection 6.1 LAI/FPAR subsets."""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import urlparse

from geodaten.common import gdal
from geodaten.common.auth import resolve_credentials, scoped_environment
from geodaten.common.errors import (
    AuthenticationError,
    CancelledError,
    ConfigurationError,
    DownloadError,
)
from geodaten.common.io import safe_component, stream_download
from geodaten.common.models import (
    CredentialField,
    DatasetSpec,
    DownloadPlan,
    DownloadRequest,
    OptionSpec,
    RasterSlice,
    VariableSpec,
)
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent
from geodaten.downloader.base import Provider, require_module


@dataclass(frozen=True)
class _Product:
    title: str
    first_date: date
    interval_days: int


PRODUCTS: dict[str, _Product] = {
    "MCD15A3H": _Product("Terra+Aqua 4-day, 500 m", date(2002, 7, 4), 4),
    "MCD15A2H": _Product("Terra+Aqua 8-day, 500 m", date(2002, 7, 4), 8),
    "MOD15A2H": _Product("Terra 8-day, 500 m", date(2000, 2, 18), 8),
    "MYD15A2H": _Product("Aqua 8-day, 500 m", date(2002, 7, 4), 8),
}
_CMR_CACHE_SECONDS = 300.0


@dataclass(frozen=True)
class _Layer:
    sds: str
    attrs: dict[str, Any]


LAYERS: dict[str, _Layer] = {
    "lai": _Layer(
        "Lai_500m",
        {
            "long_name": "leaf area index",
            "standard_name": "leaf_area_index",
            "units": "m2 m-2",
            "scale_factor": 0.1,
            "_FillValue": 255,
            "valid_range": (0, 100),
        },
    ),
    "fpar": _Layer(
        "Fpar_500m",
        {
            "long_name": "fraction of absorbed photosynthetically active radiation",
            "units": "1",
            "scale_factor": 0.01,
            "_FillValue": 255,
            "valid_range": (0, 100),
        },
    ),
    "lai_std_dev": _Layer(
        "LaiStdDev_500m",
        {
            "long_name": "leaf area index standard deviation",
            "units": "m2 m-2",
            "scale_factor": 0.1,
            "_FillValue": 255,
            "valid_range": (0, 100),
        },
    ),
    "fpar_std_dev": _Layer(
        "FparStdDev_500m",
        {
            "long_name": "FPAR standard deviation",
            "units": "1",
            "scale_factor": 0.01,
            "_FillValue": 255,
            "valid_range": (0, 100),
        },
    ),
    "fpar_lai_qc": _Layer(
        "FparLai_QC",
        {
            "long_name": "FPAR and LAI quality-control flags",
            "units": "1",
            "_FillValue": 255,
        },
    ),
    "fpar_extra_qc": _Layer(
        "FparExtra_QC",
        {
            "long_name": "FPAR and LAI extra quality-control flags",
            "units": "1",
            "_FillValue": 255,
        },
    ),
}

_ACQUISITION_RE = re.compile(r"\.A(?P<year>\d{4})(?P<doy>\d{3})\.", re.IGNORECASE)


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


def resolve_date_window(request: DownloadRequest, product: str) -> tuple[date, date]:
    """Resolve request keywords to an inclusive CMR search interval."""

    if request.time is None:  # guarded by Provider.validate; useful to direct callers
        raise ConfigurationError("MODIS LAI requires a temporal range")
    first = PRODUCTS[product].first_date
    start = _parse_boundary(request.time.start, end=False, earliest=first)
    end = _parse_boundary(request.time.end, end=True, earliest=first)
    if start > end:
        raise ConfigurationError("Start date must be on or before end date")
    return start, end


def _year_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    current = start
    while current <= end:
        finish = min(end, date(current.year, 12, 31))
        yield current, finish
        current = finish + timedelta(days=1)


def granule_filename(granule: Any) -> str:
    """Return the HDF name without assuming a particular earthaccess version."""

    links: Sequence[str] = ()
    method = getattr(granule, "data_links", None)
    if callable(method):
        for access in ("external", None):
            try:
                links = method(access=access) if access else method()
            except (TypeError, ValueError):
                continue
            if links:
                break
    for link in links:
        name = Path(urlparse(str(link)).path).name
        if name.lower().endswith(".hdf"):
            return name
    meta = granule.get("meta", {}) if hasattr(granule, "get") else {}
    native_id = str(meta.get("native-id", "")).strip()
    if native_id:
        return native_id if native_id.lower().endswith(".hdf") else native_id + ".hdf"
    raise DownloadError("CMR returned a granule without an HDF data link or native identifier")


def granule_url(granule: Any) -> str:
    """Return the authenticated HTTPS payload URL for an earthaccess granule."""

    method = getattr(granule, "data_links", None)
    if callable(method):
        for access in ("external", None):
            try:
                links = method(access=access) if access else method()
            except (TypeError, ValueError):
                continue
            for link in links or ():
                parsed = urlparse(str(link))
                if parsed.scheme in {"http", "https"} and parsed.path.lower().endswith(".hdf"):
                    return str(link)
    umm = granule.get("umm", {}) if hasattr(granule, "get") else {}
    for link in umm.get("RelatedUrls", ()):
        url = str(link.get("URL", ""))
        if urlparse(url).scheme in {"http", "https"} and urlparse(url).path.lower().endswith(
            ".hdf"
        ):
            return url
    raise DownloadError(f"CMR returned no HTTPS HDF link for {granule_filename(granule)}")


def _iso_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def granule_interval(granule: Any) -> tuple[date, date]:
    """Return a granule's temporal coverage, falling back to its MODIS name."""

    umm = granule.get("umm", {}) if hasattr(granule, "get") else {}
    temporal = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
    beginning = _iso_date(temporal.get("BeginningDateTime"))
    ending = _iso_date(temporal.get("EndingDateTime"))
    if beginning:
        return beginning, ending or beginning

    match = _ACQUISITION_RE.search(granule_filename(granule))
    if not match:
        raise DownloadError(f"Could not determine acquisition date for {granule_filename(granule)}")
    try:
        beginning = date(int(match["year"]), 1, 1) + timedelta(days=int(match["doy"]) - 1)
    except ValueError as exc:
        raise DownloadError(f"Invalid MODIS acquisition date in {granule_filename(granule)}") from exc
    return beginning, beginning


def _granule_bytes(granule: Any) -> int | None:
    value = granule.get("size") if hasattr(granule, "get") else None
    if callable(value):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            value = value()
    try:
        return max(0, round(float(value) * 1024 * 1024)) if value is not None else None
    except (TypeError, ValueError):
        return None


def subdataset_names(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Extract GDAL HDF subdataset connection strings from ``gdalinfo`` JSON."""

    values = metadata.get("metadata", {}).get("SUBDATASETS", {})
    names = [str(value) for key, value in values.items() if str(key).endswith("_NAME")]
    return tuple(names)


def find_subdataset(metadata: dict[str, Any], sds: str) -> str:
    suffix = ":" + sds.lower()
    matches = [name for name in subdataset_names(metadata) if name.lower().endswith(suffix)]
    if len(matches) != 1:
        raise DownloadError(f"Expected one HDF subdataset {sds!r}, found {len(matches)}")
    return matches[0]


def _search_data(product: str, start: date, end: date, bbox: tuple[float, ...]) -> list[Any]:
    require_module("earthaccess", "earthdata")
    import earthaccess

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return earthaccess.search_data(
            short_name=product,
            version="061",
            temporal=(start.isoformat(), end.isoformat()),
            bounding_box=bbox,
            downloadable=True,
            cloud_hosted=True,
            count=-1,
        )


def _groups_for_window(
    granules: Sequence[Any],
    *,
    window_start: date,
    window_end: date | None = None,
    first_window: bool = False,
) -> list[tuple[date, list[Any]]]:
    del first_window  # retained for compatibility with callers of the earlier helper
    groups: dict[date, list[Any]] = defaultdict(list)
    for granule in granules:
        beginning, _ = granule_interval(granule)
        # CMR searches by interval intersection and can therefore return a
        # composite whose coverage overlaps the request but whose nominal date
        # precedes it. Select strictly by the nominal (beginning) date. This
        # also removes the duplicate returned in the next yearly query when a
        # composite spans New Year.
        if beginning < window_start or (window_end is not None and beginning > window_end):
            continue
        groups[beginning].append(granule)
    return sorted(groups.items())


class ModisLaiProvider(Provider):
    spec = DatasetSpec(
        id="modis_lai",
        title="MODIS Collection 6.1 LAI/FPAR",
        summary=(
            "Native 500 m sinusoidal MODIS LAI, FPAR, uncertainty, and quality layers "
            "from Terra, Aqua, or their combined composites."
        ),
        group="LAI",
        variables=tuple(
            VariableSpec(
                id=identifier,
                title=str(layer.attrs["long_name"]),
                units=str(layer.attrs.get("units", "1")),
                standard_name=layer.attrs.get("standard_name"),
            )
            for identifier, layer in LAYERS.items()
        ),
        credentials=(
            CredentialField(
                "earthdata_token",
                "Earthdata token",
                ("EARTHDATA_TOKEN",),
                required=False,
                description="Optional token; otherwise provide a username/password pair or ~/.netrc.",
            ),
            CredentialField(
                "earthdata_username",
                "Earthdata username",
                ("EARTHDATA_USERNAME",),
                secret=False,
                required=False,
            ),
            CredentialField(
                "earthdata_password",
                "Earthdata password",
                ("EARTHDATA_PASSWORD",),
                required=False,
            ),
        ),
        options=(
            OptionSpec(
                "product",
                "MODIS product",
                default="MCD15A3H",
                choices=tuple(PRODUCTS),
                description="Collection 6.1 short name; combined 4-day is the default.",
            ),
        ),
        source_url="https://cmr.earthdata.nasa.gov/search/site/collections/directory/LPCLOUD/gov.nasa.eosdis",
        citation="MODIS Collection 6.1 LAI/FPAR product user guide and the selected product DOI",
        license="NASA Earth Science Data and Information System data policy",
    )

    def __init__(self) -> None:
        self._resolved: dict[
            str, tuple[float, tuple[tuple[date, tuple[Any, ...]], ...]]
        ] = {}

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        if request.bbox.crosses_antimeridian:
            raise ConfigurationError("MODIS LAI currently requires a bbox that does not cross the antimeridian")
        product = str(self.option(request, "product"))
        resolve_date_window(request, product)

    def _iter_groups(
        self,
        request: DownloadRequest,
        start: date,
        end: date,
        *,
        progress: ProgressCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> Iterator[tuple[date, list[Any]]]:
        product = str(self.option(request, "product"))
        windows = tuple(_year_windows(start, end))
        for index, (window_start, window_end) in enumerate(windows):
            if cancel:
                cancel.check()
            if progress:
                progress(
                    ProgressEvent(
                        "discover",
                        f"Searching {product} for {window_start.year}",
                        current=index + 1,
                        total=len(windows),
                    )
                )
            granules = _search_data(product, window_start, window_end, request.bbox.as_tuple())
            yield from _groups_for_window(
                granules,
                window_start=window_start,
                window_end=window_end,
                first_window=index == 0,
            )

    def _resolution(
        self,
        request: DownloadRequest,
        start: date,
        end: date,
        *,
        progress: ProgressCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[tuple[date, tuple[Any, ...]], ...]:
        key = request.fingerprint
        cached = self._resolved.get(key)
        if cached is not None and monotonic() - cached[0] <= _CMR_CACHE_SECONDS:
            if cancel:
                cancel.check()
            return cached[1]
        resolved = tuple(
            (composite, tuple(granules))
            for composite, granules in self._iter_groups(
                request, start, end, progress=progress, cancel=cancel
            )
        )
        self._resolved.clear()
        self._resolved[key] = (monotonic(), resolved)
        return resolved

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        product = str(self.option(request, "product"))
        start, end = resolve_date_window(request, product)
        periods = 0
        granule_count = 0
        estimated = 0
        sizes_known = True
        for _, granules in self._resolution(request, start, end):
            periods += 1
            granule_count += len(granules)
            for granule in granules:
                size = _granule_bytes(granule)
                if size is None:
                    sizes_known = False
                else:
                    estimated += size
        if not periods:
            raise DownloadError("No MODIS LAI composites matched the requested extent and time")
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=periods,
            slice_count=periods * len(request.variables),
            estimated_bytes=estimated if sizes_known else None,
            details={
                "product": product,
                "collection": "061",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "granules": granule_count,
                "composite_days": PRODUCTS[product].interval_days,
                "native_grid": "MODIS Sinusoidal, 500 m",
            },
        )

    @contextmanager
    def _earthdata_session(self, request: DownloadRequest) -> Iterator[Any]:
        require_module("earthaccess", "earthdata")
        import earthaccess

        credentials = resolve_credentials(self.spec.credentials, request.credentials)
        token = credentials.get("earthdata_token")
        username = credentials.get("earthdata_username")
        password = credentials.get("earthdata_password")
        if not token and bool(username) != bool(password):
            raise AuthenticationError("Earthdata username and password must be supplied together")

        try:
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
            with scoped_environment(environment):
                auth = earthaccess.Auth().login(strategy=strategy, persist=False)
            if not getattr(auth, "authenticated", False):
                raise AuthenticationError("Earthdata authentication was not accepted")
            session = auth.get_session()
            session.headers.setdefault("User-Agent", "geodaten/0.1")
        except AuthenticationError:
            raise
        except Exception as exc:
            raise AuthenticationError(
                "Earthdata authentication failed; supply your token, username/password, or ~/.netrc"
            ) from exc
        try:
            yield session
        finally:
            session.close()

    @staticmethod
    def _valid_hdf(
        path: Path,
        variables: Sequence[str],
        cancel: CancellationToken | None = None,
    ) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            metadata = gdal.info(path, cancel=cancel)
            for variable in variables:
                find_subdataset(metadata, LAYERS[variable].sds)
            return True
        except CancelledError:
            raise
        except Exception:
            return False

    def _cached_granules(
        self,
        request: DownloadRequest,
        workspace: Path,
        composite: date,
        granules: Sequence[Any],
        session: Any,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> list[Path]:
        product = str(self.option(request, "product"))
        cache_parent = request.cache_dir or workspace.parent
        cache = Path(cache_parent) / self.spec.id / product.lower()
        cache.mkdir(parents=True, exist_ok=True)

        resolved: dict[str, Path] = {}
        pending: list[Any] = []
        for granule in granules:
            cancel.check()
            name = granule_filename(granule)
            destination = cache / name
            if not request.overwrite and self._valid_hdf(
                destination, request.variables, cancel=cancel
            ):
                resolved[name] = destination
            else:
                destination.unlink(missing_ok=True)
                pending.append(granule)

        for index, granule in enumerate(pending, start=1):
            cancel.check()
            name = granule_filename(granule)
            destination = cache / name
            progress(
                ProgressEvent(
                    "download",
                    f"Downloading MODIS tile {name}",
                    current=index,
                    total=len(pending),
                )
            )
            stream_download(
                granule_url(granule),
                destination,
                session=session,
                progress=progress,
                cancel=cancel,
            )
            if not self._valid_hdf(destination, request.variables, cancel=cancel):
                destination.unlink(missing_ok=True)
                raise DownloadError(f"Downloaded MODIS HDF failed validation: {name}")
            resolved[name] = destination

        return [resolved[granule_filename(granule)] for granule in granules]

    def slices(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Iterator[RasterSlice]:
        self.validate(request)
        gdal.require_commands("gdalinfo", "gdalbuildvrt", "gdal_translate")
        product = str(self.option(request, "product"))
        start, end = resolve_date_window(request, product)

        produced = 0
        with self._earthdata_session(request) as session:
            groups = self._resolution(
                request, start, end, progress=progress, cancel=cancel
            )
            self._resolved.pop(request.fingerprint, None)
            for composite, granules in groups:
                cancel.check()
                hdfs = self._cached_granules(
                    request, workspace, composite, granules, session, progress, cancel
                )
                metadata = {path: gdal.info(path, cancel=cancel) for path in hdfs}
                for variable in request.variables:
                    cancel.check()
                    layer = LAYERS[variable]
                    sources = [find_subdataset(metadata[path], layer.sds) for path in hdfs]
                    destination = (
                        workspace
                        / "slices"
                        / f"{safe_component(product)}_{composite.isoformat()}_{safe_component(variable)}.tif"
                    )
                    progress(ProgressEvent("transform", f"Mosaicking {variable} for {composite}"))
                    gdal.mosaic_to_cog(
                        sources,
                        destination,
                        bbox=request.bbox,
                        metadata={"product": product, "collection": "061", "variable": variable},
                        cancel=cancel,
                    )
                    produced += 1
                    try:
                        # The service consumes each source synchronously. Remove
                        # derived slices as soon as the writer has copied them;
                        # original HDFs remain in the validated source cache.
                        yield RasterSlice(
                            variable=variable,
                            source=destination,
                            time=composite,
                            attrs={
                                **layer.attrs,
                                "product": product,
                                "collection": "061",
                                "source": "NASA LP DAAC via Earthdata",
                            },
                        )
                    finally:
                        destination.unlink(missing_ok=True)
        if not produced:
            raise DownloadError("No MODIS LAI raster slices were produced")
