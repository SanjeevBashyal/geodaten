"""NASA NEX-GDDP-CMIP6 daily subsets from the NCCS THREDDS server."""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin
from xml.etree import ElementTree

import requests

from geodaten.common.errors import ConfigurationError, DownloadError
from geodaten.common.gdal import require_commands, run
from geodaten.common.io import atomic_path, make_session, safe_component
from geodaten.common.models import (
    DatasetSpec,
    DownloadPlan,
    DownloadRequest,
    OptionSpec,
    RasterSlice,
    VariableSpec,
)
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent
from geodaten.downloader.base import Provider, require_module


THREDDS_ROOT = "https://ds.nccs.nasa.gov/thredds"
CATALOG_ROOT = f"{THREDDS_ROOT}/catalog/AMES/NEX/GDDP-CMIP6/catalog.xml"
_XLINK = "{http://www.w3.org/1999/xlink}"
DATASET_START = date(1950, 1, 1)
DATASET_END = date(2100, 12, 31)
SCENARIO_PERIODS = {
    "historical": (DATASET_START, date(2014, 12, 31)),
    "ssp126": (date(2015, 1, 1), DATASET_END),
    "ssp245": (date(2015, 1, 1), DATASET_END),
    "ssp370": (date(2015, 1, 1), DATASET_END),
    "ssp585": (date(2015, 1, 1), DATASET_END),
}


@dataclass(frozen=True)
class NexAsset:
    """One actual annual file advertised by a THREDDS catalog."""

    model: str
    scenario: str
    variant: str
    variable: str
    year: int
    grid: str
    url_path: str
    version: str = ""
    size: int | None = None


@dataclass(frozen=True)
class NexSubsetJob:
    asset: NexAsset
    start: date
    end: date


def _values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = (part.strip() for part in value.replace("/", ",").split(","))
        return tuple(dict.fromkeys(part for part in parts if part))
    if isinstance(value, Sequence):
        return tuple(dict.fromkeys(str(part).strip() for part in value if str(part).strip()))
    return (str(value).strip(),) if value is not None else ()


def _date(value: str, *, name: str, end: bool = False) -> date:
    if re.fullmatch(r"\d{4}", value):
        return date(int(value), 12 if end else 1, 31 if end else 1)
    if re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = map(int, value.split("-"))
        try:
            return date(year, month, calendar.monthrange(year, month)[1] if end else 1)
        except ValueError as exc:
            raise ConfigurationError(f"{name} contains an invalid month") from exc
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be 'earliest', 'latest', YYYY, YYYY-MM, or YYYY-MM-DD"
        ) from exc


def _merge_periods(periods: Sequence[tuple[date, date]]) -> tuple[tuple[date, date], ...]:
    merged: list[tuple[date, date]] = []
    for start, end in sorted(periods):
        if merged and start <= merged[-1][1] + timedelta(days=1):
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _omitted_periods(
    start: date, end: date, coverage: Sequence[tuple[date, date]]
) -> tuple[tuple[date, date], ...]:
    omitted: list[tuple[date, date]] = []
    cursor = start
    for covered_start, covered_end in _merge_periods(coverage):
        if covered_end < cursor or covered_start > end:
            continue
        if covered_start > cursor:
            omitted.append((cursor, min(end, covered_start - timedelta(days=1))))
        cursor = max(cursor, covered_end + timedelta(days=1))
        if cursor > end:
            break
    if cursor <= end:
        omitted.append((cursor, end))
    return tuple(omitted)


def _requested_period(
    request: DownloadRequest, scenarios: Sequence[str]
) -> tuple[date, date, dict[str, tuple[date, date] | None], tuple[str, ...]]:
    """Resolve nominal v2 coverage without silently clipping explicit bounds."""

    assert request.time is not None
    periods = [SCENARIO_PERIODS[scenario] for scenario in scenarios]
    start = min(item[0] for item in periods) if request.time.start == "earliest" else _date(
        request.time.start, name="Temporal start"
    )
    end = max(item[1] for item in periods) if request.time.end == "latest" else _date(
        request.time.end, name="Temporal end", end=True
    )
    if end < start:
        raise ConfigurationError("Temporal end must not be before start")
    intersections = {
        scenario: (
            max(start, SCENARIO_PERIODS[scenario][0]),
            min(end, SCENARIO_PERIODS[scenario][1]),
        )
        for scenario in scenarios
    }
    applicable = {
        scenario: period if period[0] <= period[1] else None
        for scenario, period in intersections.items()
    }
    coverage = [period for period in applicable.values() if period is not None]
    omitted = _omitted_periods(start, end, coverage)
    labels = tuple(f"{left.isoformat()}..{right.isoformat()}" for left, right in omitted)
    if omitted and not request.allow_partial:
        raise ConfigurationError(
            "Selected NEX scenarios do not cover requested interval(s): " + ", ".join(labels)
        )
    warnings = tuple(f"Omitting unsupported NEX interval {label}." for label in labels)
    return start, end, applicable, warnings


def _catalog_refs(xml: bytes | str, catalog_url: str) -> dict[str, str]:
    """Return catalog-reference titles and their absolute URLs."""

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise DownloadError(f"THREDDS returned an invalid catalog: {catalog_url}") from exc
    result: dict[str, str] = {}
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "catalogRef":
            continue
        title = (element.get(f"{_XLINK}title") or element.get("name") or "").strip()
        href = (element.get(f"{_XLINK}href") or "").strip()
        if title and href:
            result[title] = urljoin(catalog_url, href)
    return result


def _size_bytes(element: ElementTree.Element) -> int | None:
    for child in element:
        if child.tag.rsplit("}", 1)[-1] != "dataSize" or not child.text:
            continue
        units = child.get("units", "bytes").lower()
        factors = {
            "bytes": 1,
            "kbytes": 1_000,
            "mbytes": 1_000_000,
            "gbytes": 1_000_000_000,
            "kib": 1 << 10,
            "mib": 1 << 20,
            "gib": 1 << 30,
        }
        try:
            return int(float(child.text.strip()) * factors.get(units, 1))
        except ValueError:
            return None
    return None


def parse_asset_catalog(
    xml: bytes | str,
    *,
    model: str,
    scenario: str,
    variant: str,
    variable: str,
) -> tuple[NexAsset, ...]:
    """Parse file names, including their real grid label, without guessing URLs."""

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise DownloadError("THREDDS returned an invalid variable catalog") from exc
    suffix = re.compile(
        rf"^{re.escape(variable)}_day_.+?_{re.escape(scenario)}_"
        rf"{re.escape(variant)}_(?P<grid>[^_]+)_(?P<year>\d{{4}})"
        rf"(?:_(?P<version>v[^/]+))?\.nc$"
    )
    found: list[NexAsset] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "dataset":
            continue
        url_path = (element.get("urlPath") or "").strip()
        name = (element.get("name") or Path(url_path).name).strip()
        match = suffix.match(name)
        if not url_path or not match:
            continue
        found.append(
            NexAsset(
                model=model,
                scenario=scenario,
                variant=variant,
                variable=variable,
                year=int(match.group("year")),
                grid=match.group("grid"),
                url_path=url_path.lstrip("/"),
                version=match.group("version") or "",
                size=_size_bytes(element),
            )
        )
    return tuple(sorted(found, key=lambda asset: (asset.year, asset.version, asset.grid)))


def subset_jobs(
    assets: Sequence[NexAsset], start: date, end: date, *, allow_partial: bool = False
) -> tuple[NexSubsetJob, ...]:
    """Intersect exact request dates with advertised annual files."""

    if end < start:
        raise ConfigurationError("Temporal end must not be before start")
    by_year: dict[int, list[NexAsset]] = {}
    for asset in assets:
        by_year.setdefault(asset.year, []).append(asset)
    jobs: list[NexSubsetJob] = []
    missing: list[int] = []
    for year in range(start.year, end.year + 1):
        candidates = by_year.get(year, [])
        if not candidates:
            missing.append(year)
            continue
        # Prefer the newest advertised version, but never choose ambiguously
        # between two grids carrying the same newest version.
        newest = max((asset.version for asset in candidates), default="")
        selected = [asset for asset in candidates if asset.version == newest]
        grids = {asset.grid for asset in selected}
        if len(grids) != 1:
            raise DownloadError(
                f"THREDDS advertises ambiguous grids for {year}: {', '.join(sorted(grids))}"
            )
        asset = selected[-1]
        jobs.append(
            NexSubsetJob(
                asset=asset,
                start=max(start, date(year, 1, 1)),
                end=min(end, date(year, 12, 31)),
            )
        )
    if missing and not allow_partial:
        shown = ", ".join(map(str, missing[:8]))
        if len(missing) > 8:
            shown += ", ..."
        raise DownloadError(f"NEX-GDDP-CMIP6 catalog is missing requested year(s): {shown}")
    return tuple(jobs)


def ncss_query_url(job: NexSubsetJob, request: DownloadRequest, *, root: str = THREDDS_ROOT) -> str:
    if request.bbox.crosses_antimeridian:
        raise ConfigurationError("NEX NCSS subsets cannot safely cross the antimeridian")
    base = f"{root.rstrip('/')}/ncss/grid/{job.asset.url_path}"
    parameters = {
        "var": job.asset.variable,
        "north": f"{request.bbox.north:g}",
        "south": f"{request.bbox.south:g}",
        "west": f"{request.bbox.west:g}",
        "east": f"{request.bbox.east:g}",
        "horizStride": str(int(request.options.get("horizontal_stride", 1))),
        "time_start": f"{job.start.isoformat()}T12:00:00Z",
        "time_end": f"{job.end.isoformat()}T12:00:00Z",
        "accept": "netcdf4",
        "addLatLon": "true",
    }
    return base + "?" + urlencode(parameters)


def _magic_is_netcdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            magic = handle.read(8)
        return (
            magic.startswith(b"CDF\x01")
            or magic.startswith(b"CDF\x02")
            or magic.startswith(b"CDF\x05")
            or magic.startswith(b"\x89HDF\r\n\x1a\n")
        )
    except OSError:
        return False


def _calendar_date(value: Any) -> tuple[int, int, int]:
    return int(value.year), int(value.month), int(value.day)


def validate_ncss_subset(
    path: Path, job: NexSubsetJob, *, allow_partial: bool = False
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Validate the payload and return only its small time axis and attributes."""

    require_module("netCDF4", "netcdf")
    if not _magic_is_netcdf(path):
        raise DownloadError(f"NCSS response is not a NetCDF file: {path.name}")
    from netCDF4 import Dataset, num2date

    try:
        with Dataset(path) as dataset:
            if job.asset.variable not in dataset.variables:
                raise DownloadError(
                    f"NCSS response does not contain variable {job.asset.variable!r}"
                )
            variable = dataset.variables[job.asset.variable]
            if "time" not in variable.dimensions or "time" not in dataset.variables:
                raise DownloadError("NCSS response has no time dimension")
            time = dataset.variables["time"]
            units = getattr(time, "units", "")
            calendar = getattr(time, "calendar", "standard")
            if not units:
                raise DownloadError("NCSS response time coordinate has no units")
            values = tuple(
                num2date(
                    time[:],
                    units=units,
                    calendar=calendar,
                    only_use_cftime_datetimes=True,
                )
            )
            axis = variable.dimensions.index("time")
            if variable.shape[axis] != len(values) or not values:
                raise DownloadError("NCSS response contains no complete daily slices")
            attrs = {name: variable.getncattr(name) for name in variable.ncattrs()}
            attrs["calendar"] = calendar
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"Could not validate NCSS response {path.name}: {exc}") from exc

    actual_start = _calendar_date(values[0])
    actual_end = _calendar_date(values[-1])
    expected_start = (job.start.year, job.start.month, job.start.day)
    expected_end = (job.end.year, job.end.month, job.end.day)
    if any(
        int(getattr(value, "hour", -1)) != 12
        or int(getattr(value, "minute", -1)) != 0
        or int(getattr(value, "second", -1)) != 0
        for value in values
    ):
        raise DownloadError("NCSS response contains a timestamp that is not at 12:00:00")
    if actual_start < expected_start or actual_end > expected_end:
        raise DownloadError("NCSS response contains dates outside the requested annual subset")
    gaps = []
    for previous, current in zip(values, values[1:]):
        try:
            difference = current - previous
            if difference <= timedelta(0):
                raise DownloadError("NCSS response time axis is duplicate or out of order")
            if difference != timedelta(days=1):
                gaps.append((previous, current))
        except DownloadError:
            raise
        except (TypeError, ValueError) as exc:
            raise DownloadError("NCSS response time axis cannot be validated") from exc
    if not allow_partial and (
        actual_start != expected_start or actual_end != expected_end or gaps
    ):
        raise DownloadError(
            "NCSS response is temporally incomplete: "
            f"requested {job.start}..{job.end}, received "
            f"{values[0]}..{values[-1]} with {len(gaps)} gap(s)"
        )
    return values, attrs


def _missing_subset_periods(
    job: NexSubsetJob, values: Sequence[Any]
) -> tuple[tuple[date, date], ...]:
    received = {date(*_calendar_date(value)) for value in values}
    missing: list[tuple[date, date]] = []
    current = job.start
    while current <= job.end:
        if current not in received:
            missing.append((current, current))
        current += timedelta(days=1)
    return _merge_periods(missing)


class NexGddpProvider(Provider):
    spec = DatasetSpec(
        id="nex_gddp_cmip6",
        title="NASA NEX-GDDP-CMIP6",
        summary="Daily, 0.25 degree statistically downscaled CMIP6 climate projections.",
        group="Climate",
        variables=(
            VariableSpec("hurs", "Near-surface relative humidity", "%", "relative_humidity"),
            VariableSpec("huss", "Near-surface specific humidity", "1", "specific_humidity"),
            VariableSpec("pr", "Precipitation", "kg m-2 s-1", "precipitation_flux"),
            VariableSpec("rlds", "Surface downwelling longwave radiation", "W m-2"),
            VariableSpec("rsds", "Surface downwelling shortwave radiation", "W m-2"),
            VariableSpec("sfcWind", "Near-surface wind speed", "m s-1", "wind_speed"),
            VariableSpec("tas", "Near-surface air temperature", "K", "air_temperature"),
            VariableSpec("tasmax", "Daily maximum air temperature", "K", "air_temperature"),
            VariableSpec("tasmin", "Daily minimum air temperature", "K", "air_temperature"),
        ),
        options=(
            OptionSpec(
                "model",
                "Climate model",
                default=("ACCESS-CM2",),
                required=True,
                multiple=True,
                description="One or more model directory names advertised by THREDDS.",
            ),
            OptionSpec(
                "scenario",
                "Scenario",
                default=("historical",),
                choices=("historical", "ssp126", "ssp245", "ssp370", "ssp585"),
                required=True,
                multiple=True,
            ),
            OptionSpec(
                "variant",
                "Ensemble variant",
                default=("r1i1p1f1",),
                required=True,
                multiple=True,
            ),
            OptionSpec(
                "horizontal_stride",
                "Horizontal stride",
                kind="integer",
                default=1,
                description="NCSS grid stride; 1 preserves the native 0.25 degree grid.",
            ),
        ),
        source_url="https://www.nccs.nasa.gov/data-collections/nex-gddp-cmip6/",
        citation="Thrasher et al. (2022), NASA Global Daily Downscaled Projections, CMIP6.",
        license="CC0-1.0",
    )

    def __init__(self, *, catalog_root: str = CATALOG_ROOT) -> None:
        self.catalog_root = catalog_root
        self._resolved: dict[
            str, tuple[tuple[NexSubsetJob, ...], tuple[str, ...]]
        ] = {}

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        if request.bbox.crosses_antimeridian:
            raise ConfigurationError("NEX-GDDP-CMIP6 does not support antimeridian-crossing subsets")
        if request.bbox.south < -60:
            raise ConfigurationError("NEX-GDDP-CMIP6 latitude coverage starts at 60 degrees south")
        try:
            stride = int(self.option(request, "horizontal_stride"))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("horizontal_stride must be a positive integer") from exc
        if stride < 1:
            raise ConfigurationError("horizontal_stride must be a positive integer")
        if not _values(self.option(request, "model")):
            raise ConfigurationError("Select at least one climate model")
        if not _values(self.option(request, "variant")):
            raise ConfigurationError("Select at least one ensemble variant")

    def _fetch_catalog(self, url: str, session: requests.Session) -> bytes:
        try:
            response = session.get(url, timeout=(20, 90))
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            raise DownloadError(f"Could not read THREDDS catalog {url}: {exc}") from exc

    def _discover(
        self, request: DownloadRequest
    ) -> tuple[tuple[NexSubsetJob, ...], tuple[str, ...]]:
        models = _values(self.option(request, "model"))
        scenarios = _values(self.option(request, "scenario"))
        variants = _values(self.option(request, "variant"))
        _start, _end, applicable, temporal_warnings = _requested_period(request, scenarios)
        warnings = list(temporal_warnings)
        session = make_session()
        try:
            model_refs = _catalog_refs(self._fetch_catalog(self.catalog_root, session), self.catalog_root)
            unknown = [model for model in models if model not in model_refs]
            if unknown:
                raise ConfigurationError(
                    "Model(s) not advertised by NEX THREDDS: " + ", ".join(unknown)
                )
            collections: list[
                tuple[tuple[NexAsset, ...], date, date, str]
            ] = []
            for model in models:
                scenario_refs = _catalog_refs(
                    self._fetch_catalog(model_refs[model], session), model_refs[model]
                )
                for scenario in scenarios:
                    period = applicable[scenario]
                    if period is None:
                        continue
                    local_start, local_end = period
                    if scenario not in scenario_refs:
                        if request.allow_partial:
                            warnings.append(
                                f"Omitting {model}/{scenario} for "
                                f"{local_start.isoformat()}..{local_end.isoformat()}: "
                                "scenario catalog is unavailable."
                            )
                            continue
                        raise DownloadError(f"{model} does not advertise scenario {scenario}")
                    variant_refs = _catalog_refs(
                        self._fetch_catalog(scenario_refs[scenario], session),
                        scenario_refs[scenario],
                    )
                    for variant in variants:
                        if variant not in variant_refs:
                            if request.allow_partial:
                                warnings.append(
                                    f"Omitting {model}/{scenario}/{variant} for "
                                    f"{local_start.isoformat()}..{local_end.isoformat()}: "
                                    "variant catalog is unavailable."
                                )
                                continue
                            raise DownloadError(
                                f"{model}/{scenario} does not advertise variant {variant}"
                            )
                        variable_refs = _catalog_refs(
                            self._fetch_catalog(variant_refs[variant], session),
                            variant_refs[variant],
                        )
                        for variable in request.variables:
                            context = f"{model}/{scenario}/{variant}/{variable}"
                            if variable not in variable_refs:
                                if request.allow_partial:
                                    warnings.append(
                                        f"Omitting {context} for "
                                        f"{local_start.isoformat()}..{local_end.isoformat()}: "
                                        "variable catalog is unavailable."
                                    )
                                    continue
                                raise DownloadError(
                                    f"{model}/{scenario}/{variant} does not advertise {variable}"
                                )
                            xml = self._fetch_catalog(variable_refs[variable], session)
                            assets = parse_asset_catalog(
                                xml,
                                model=model,
                                scenario=scenario,
                                variant=variant,
                                variable=variable,
                            )
                            if not assets and not request.allow_partial:
                                raise DownloadError(
                                    f"No annual files found for {context}"
                                )
                            if not assets:
                                warnings.append(
                                    f"Omitting {context} for "
                                    f"{local_start.isoformat()}..{local_end.isoformat()}: "
                                    "no annual assets are advertised."
                                )
                                continue
                            collections.append((assets, local_start, local_end, context))
        finally:
            session.close()

        jobs: list[NexSubsetJob] = []
        for assets, local_start, local_end, context in collections:
            collection_jobs = subset_jobs(
                assets, local_start, local_end, allow_partial=request.allow_partial
            )
            if request.allow_partial:
                present = {job.asset.year for job in collection_jobs}
                for year in range(local_start.year, local_end.year + 1):
                    if year in present:
                        continue
                    omitted_start = max(local_start, date(year, 1, 1))
                    omitted_end = min(local_end, date(year, 12, 31))
                    warnings.append(
                        f"Omitting {context} for {omitted_start.isoformat()}.."
                        f"{omitted_end.isoformat()}: annual asset is unavailable."
                    )
            jobs.extend(collection_jobs)
        if not jobs:
            raise DownloadError("No requested dates are available in the selected NEX files")
        return tuple(jobs), tuple(dict.fromkeys(warnings))

    def _resolution(
        self, request: DownloadRequest
    ) -> tuple[tuple[NexSubsetJob, ...], tuple[str, ...]]:
        resolved = self._resolved.get(request.fingerprint)
        if resolved is None:
            resolved = self._discover(request)
            self._resolved.clear()
            self._resolved[request.fingerprint] = resolved
        return resolved

    def _jobs(self, request: DownloadRequest) -> tuple[NexSubsetJob, ...]:
        return self._resolution(request)[0]

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        jobs, warnings = self._resolution(request)
        days = sum((job.end - job.start).days + 1 for job in jobs)
        advertised = sum(job.asset.size or 0 for job in jobs) or None
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=len(jobs),
            slice_count=days,
            estimated_bytes=None,
            details={
                "annual_subsets": len(jobs),
                "daily_slices": days,
                "models": sorted({job.asset.model for job in jobs}),
                "scenarios": sorted({job.asset.scenario for job in jobs}),
                "variants": sorted({job.asset.variant for job in jobs}),
                "grids": sorted({job.asset.grid for job in jobs}),
                "asset_versions": sorted(
                    {job.asset.version for job in jobs if job.asset.version}
                ),
                "assets": [
                    {
                        "model": job.asset.model,
                        "scenario": job.asset.scenario,
                        "variant": job.asset.variant,
                        "variable": job.asset.variable,
                        "year": job.asset.year,
                        "grid": job.asset.grid,
                        "version": job.asset.version,
                        "url_path": job.asset.url_path,
                    }
                    for job in jobs
                ],
                "advertised_full_file_bytes": advertised,
            },
            warnings=warnings,
        )

    @staticmethod
    def _download(
        url: str,
        destination: Path,
        session: requests.Session,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Path:
        try:
            with session.get(url, stream=True, timeout=(30, 600)) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", "0") or 0) or None
                with atomic_path(destination, suffix=".part.nc") as temporary:
                    copied = 0
                    with temporary.open("wb") as target:
                        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                            cancel.check()
                            if not chunk:
                                continue
                            target.write(chunk)
                            copied += len(chunk)
                            progress(
                                ProgressEvent(
                                    "download",
                                    destination.name,
                                    bytes_done=copied,
                                    bytes_total=total,
                                )
                            )
                    if not _magic_is_netcdf(temporary):
                        raise DownloadError("NCSS returned a non-NetCDF response")
            return destination
        except requests.RequestException as exc:
            raise DownloadError(f"NCSS subset download failed: {exc}") from exc

    def slices(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Iterator[RasterSlice]:
        self.validate(request)
        require_commands("gdal_translate")
        jobs, warnings = self._resolution(request)
        self._resolved.pop(request.fingerprint, None)
        for warning in warnings:
            progress(ProgressEvent("warning", warning))
        session = make_session()
        try:
            for index, job in enumerate(jobs, 1):
                cancel.check()
                stem = "_".join(
                    safe_component(value)
                    for value in (
                        job.asset.variable,
                        job.asset.model,
                        job.asset.scenario,
                        job.asset.variant,
                        job.asset.grid,
                        job.asset.year,
                    )
                )
                netcdf = workspace / f"{stem}.nc"
                progress(
                    ProgressEvent(
                        "download",
                        f"NEX annual subset {job.asset.year}",
                        current=index,
                        total=len(jobs),
                    )
                )
                self._download(ncss_query_url(job, request), netcdf, session, progress, cancel)
                times, attrs = validate_ncss_subset(
                    netcdf, job, allow_partial=request.allow_partial
                )
                if request.allow_partial:
                    for missing_start, missing_end in _missing_subset_periods(job, times):
                        progress(
                            ProgressEvent(
                                "warning",
                                f"Omitting {job.asset.model}/{job.asset.scenario}/"
                                f"{job.asset.variant}/{job.asset.variable} for "
                                f"{missing_start.isoformat()}..{missing_end.isoformat()}: "
                                "NCSS response has no slice.",
                            )
                        )
                # NCSS files have a valid affine grid but omit a raster CRS. A
                # VRT assigns EPSG:4326 without touching the annual data.
                vrt = workspace / f"{stem}.vrt"
                source = f'NETCDF:"{netcdf}":{job.asset.variable}'
                run(
                    (
                        "gdal_translate",
                        "-q",
                        "-of",
                        "VRT",
                        "-a_srs",
                        "EPSG:4326",
                        "-projwin",
                        f"{request.bbox.west:g}",
                        f"{request.bbox.north:g}",
                        f"{request.bbox.east:g}",
                        f"{request.bbox.south:g}",
                        source,
                        vrt,
                    ),
                    cancel=cancel,
                )
                try:
                    for band, timestamp in enumerate(times, 1):
                        cancel.check()
                        yield RasterSlice(
                            variable=job.asset.variable,
                            source=vrt,
                            time=timestamp,
                            band=band,
                            dimensions={
                                "model": job.asset.model,
                                "scenario": job.asset.scenario,
                                "variant": job.asset.variant,
                                "grid": job.asset.grid,
                            },
                            attrs={
                                **attrs,
                                "source": "NASA NEX-GDDP-CMIP6",
                                "source_model": job.asset.model,
                                "experiment": job.asset.scenario,
                                "variant_label": job.asset.variant,
                            },
                        )
                finally:
                    vrt.unlink(missing_ok=True)
                    netcdf.unlink(missing_ok=True)
        finally:
            session.close()
