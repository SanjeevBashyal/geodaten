"""DestinE Climate DT Generation-2 IFS-FESOM regional subsets."""

from __future__ import annotations

import calendar
import json
import math
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import Resampling, reproject

from geodaten.common.errors import (
    AuthenticationError,
    ConfigurationError,
    DependencyError,
    DownloadError,
    OutputError,
)
from geodaten.common.io import atomic_path, safe_component
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


DEFAULT_ADDRESS = "polytope.lumi.apps.dte.destination-earth.eu"
COLLECTION = "destination-earth"
MAX_RASTER_PIXELS = 50_000_000
PRESSURE_LEVELS = frozenset(
    {
        "1000",
        "925",
        "850",
        "700",
        "600",
        "500",
        "400",
        "300",
        "250",
        "200",
        "150",
        "100",
        "70",
        "50",
        "30",
        "20",
        "10",
        "5",
        "1",
    }
)
SURFACE_LEVELS = {
    "10u": "10",
    "10v": "10",
    "2t": "2",
    "2d": "2",
    "avg_sdswrf": "",
    "avg_sdlwrf": "",
    "avg_tprate": "",
}


@dataclass(frozen=True)
class Experiment:
    activity: str
    code: str
    start: date
    end: date


EXPERIMENTS: dict[str, Experiment] = {
    "historical": Experiment("baseline", "hist", date(1990, 1, 1), date(2014, 12, 31)),
    "projection": Experiment("projections", "SSP3-7.0", date(2015, 1, 1), date(2049, 12, 31)),
    "control": Experiment("baseline", "cont", date(1990, 1, 1), date(1999, 12, 31)),
}


@dataclass(frozen=True)
class Parameter:
    clte_id: str
    clmn_id: str
    clte_name: str
    clmn_name: str
    title: str
    units: str
    standard_name: str | None
    levtypes: frozenset[str]

    def param_id(self, frequency: str) -> str:
        return self.clmn_id if frequency == "clmn" else self.clte_id

    def short_name(self, frequency: str) -> str:
        return self.clmn_name if frequency == "clmn" else self.clte_name


PARAMETERS: dict[str, Parameter] = {
    "z": Parameter(
        "129", "235129", "z", "avg_z", "Geopotential", "m2 s-2", "geopotential", frozenset({"pl"})
    ),
    "t": Parameter(
        "130", "235130", "t", "avg_t", "Temperature", "K", "air_temperature", frozenset({"pl"})
    ),
    "u": Parameter(
        "131", "235131", "u", "avg_u", "U wind component", "m s-1", "eastward_wind", frozenset({"pl", "hl"})
    ),
    "v": Parameter(
        "132", "235132", "v", "avg_v", "V wind component", "m s-1", "northward_wind", frozenset({"pl", "hl"})
    ),
    "q": Parameter(
        "133", "235133", "q", "avg_q", "Specific humidity", "kg kg-1", "specific_humidity", frozenset({"pl"})
    ),
    "10u": Parameter(
        "165", "235165", "10u", "avg_10u", "10 metre U wind", "m s-1", "eastward_wind", frozenset({"sfc"})
    ),
    "10v": Parameter(
        "166", "235166", "10v", "avg_10v", "10 metre V wind", "m s-1", "northward_wind", frozenset({"sfc"})
    ),
    "2t": Parameter(
        "167", "228004", "2t", "avg_2t", "2 metre temperature", "K", "air_temperature", frozenset({"sfc"})
    ),
    "2d": Parameter(
        "168",
        "235168",
        "2d",
        "avg_2d",
        "2 metre dewpoint temperature",
        "K",
        "dew_point_temperature",
        frozenset({"sfc"}),
    ),
    "avg_sdswrf": Parameter(
        "235035",
        "235035",
        "avg_sdswrf",
        "avg_sdswrf",
        "Mean surface downward shortwave radiation flux",
        "W m-2",
        "surface_downwelling_shortwave_flux_in_air",
        frozenset({"sfc"}),
    ),
    "avg_sdlwrf": Parameter(
        "235036",
        "235036",
        "avg_sdlwrf",
        "avg_sdlwrf",
        "Mean surface downward longwave radiation flux",
        "W m-2",
        "surface_downwelling_longwave_flux_in_air",
        frozenset({"sfc"}),
    ),
    "avg_tprate": Parameter(
        "235055",
        "235055",
        "avg_tprate",
        "avg_tprate",
        "Mean total precipitation rate",
        "kg m-2 s-1",
        "precipitation_flux",
        frozenset({"sfc"}),
    ),
}


@dataclass(frozen=True)
class IfsJob:
    experiment: str
    period: date
    request: Mapping[str, str]
    variables: tuple[str, ...]
    expected_times: tuple[str, ...]
    levels: tuple[str, ...]
    frequency: str


@dataclass(frozen=True)
class FieldDescriptor:
    field: Any
    variable: str
    timestamp: datetime
    time_code: str
    level: str | None
    attrs: Mapping[str, Any]


def _items(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = (part.strip() for part in value.replace(",", "/").split("/"))
        return tuple(dict.fromkeys(part for part in parts if part))
    if isinstance(value, Sequence):
        return tuple(dict.fromkeys(str(part).strip() for part in value if str(part).strip()))
    text = str(value).strip()
    return (text,) if text else ()


def _time_int(value: str) -> int:
    text = value.strip().replace(":", "")
    if len(text) <= 2:
        text += "00"
    if not text.isdigit():
        raise ConfigurationError(f"Invalid IFS time {value!r}; expected HHMM")
    number = int(text)
    hour, minute = divmod(number, 100)
    if hour > 23 or minute > 59:
        raise ConfigurationError(f"Invalid IFS time {value!r}; expected HHMM")
    return number


def expand_times(value: Any) -> tuple[str, ...]:
    """Expand common MARS time expressions without creating request jobs."""

    tokens = list(_items(value))
    if not tokens:
        raise ConfigurationError("Select at least one time for clte output")
    lower = [token.lower() for token in tokens]
    numbers: list[int]
    if "to" not in lower:
        numbers = [_time_int(token) for token in tokens]
    else:
        at = lower.index("to")
        if at < 1 or at + 1 >= len(tokens):
            raise ConfigurationError("Invalid IFS time range")
        start = _time_int(tokens[0])
        end = _time_int(tokens[at + 1])
        if "by" in lower:
            by = lower.index("by")
            if by + 1 >= len(tokens):
                raise ConfigurationError("Invalid IFS time step")
            step_clock = _time_int(tokens[by + 1])
            step = (step_clock // 100) * 60 + step_clock % 100
        elif at >= 2:
            second = _time_int(tokens[1])
            step = (second // 100) * 60 + second % 100 - (
                (start // 100) * 60 + start % 100
            )
        else:
            step = 60
        start_minutes = (start // 100) * 60 + start % 100
        end_minutes = (end // 100) * 60 + end % 100
        if step <= 0 or end_minutes < start_minutes:
            raise ConfigurationError("IFS time range must increase with a positive step")
        minutes = range(start_minutes, end_minutes + 1, step)
        numbers = [(minute // 60) * 100 + minute % 60 for minute in minutes]
        if not numbers or numbers[-1] != end:
            raise ConfigurationError("IFS time step does not land on the requested end time")
    return tuple(dict.fromkeys(f"{number:04d}" for number in numbers))


def _month_start(value: date) -> date:
    return value.replace(day=1)


def _month_end(value: date) -> date:
    return value.replace(day=calendar.monthrange(value.year, value.month)[1])


def _parse_date(value: str, *, end: bool, monthly: bool) -> date:
    text = value.strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m").date()
        except ValueError as exc:
            raise ConfigurationError(
                "IFS temporal bounds must be earliest/latest, YYYY-MM, or YYYY-MM-DD"
            ) from exc
    if monthly:
        return _month_end(parsed) if end else _month_start(parsed)
    return parsed


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


def _period_label(period: tuple[date, date], *, frequency: str) -> str:
    start, end = period
    if frequency == "clmn":
        return f"{start:%Y-%m}..{end:%Y-%m}"
    return f"{start.isoformat()}..{end.isoformat()}"


def resolve_experiment_periods(
    time_range: Any,
    experiments: Sequence[str],
    *,
    frequency: str,
    allow_partial: bool,
) -> tuple[tuple[tuple[str, date, date], ...], tuple[str, ...]]:
    """Resolve selected experiment intersections and report every coverage gap."""

    if not experiments:
        raise ConfigurationError("Select at least one IFS-FESOM experiment")
    unknown = sorted(set(experiments) - set(EXPERIMENTS))
    if unknown:
        raise ConfigurationError(f"Unknown IFS-FESOM experiment(s): {', '.join(unknown)}")
    monthly = frequency == "clmn"
    selected = [EXPERIMENTS[name] for name in experiments]
    start = (
        min(item.start for item in selected)
        if time_range.start == "earliest"
        else _parse_date(time_range.start, end=False, monthly=monthly)
    )
    end = (
        max(item.end for item in selected)
        if time_range.end == "latest"
        else _parse_date(time_range.end, end=True, monthly=monthly)
    )
    if end < start:
        raise ConfigurationError("Temporal end must not be before start")

    intersections = {
        name: (max(start, EXPERIMENTS[name].start), min(end, EXPERIMENTS[name].end))
        for name in experiments
    }
    groups: list[tuple[str, ...]] = []
    remaining = list(experiments)
    if "historical" in remaining and "projection" in remaining:
        groups.append(("historical", "projection"))
        remaining = [name for name in remaining if name not in groups[0]]
    groups.extend((name,) for name in remaining)

    omissions: list[tuple[str | None, tuple[date, date]]] = []
    for group in groups:
        coverage = [
            intersections[name]
            for name in group
            if intersections[name][0] <= intersections[name][1]
        ]
        context = group[0] if len(group) == 1 and len(experiments) > 1 else None
        omissions.extend(
            (context, period) for period in _omitted_periods(start, end, coverage)
        )
    labels = tuple(
        (f"{context} " if context else "") + _period_label(period, frequency=frequency)
        for context, period in omissions
    )
    if omissions and not allow_partial:
        raise ConfigurationError(
            "Selected IFS-FESOM experiments do not cover requested interval(s): "
            + ", ".join(labels)
        )
    resolved = tuple(
        (name, max(start, EXPERIMENTS[name].start), min(end, EXPERIMENTS[name].end))
        for name in experiments
        if max(start, EXPERIMENTS[name].start) <= min(end, EXPERIMENTS[name].end)
    )
    warnings = tuple(f"Omitting unsupported IFS-FESOM interval {label}." for label in labels)
    return resolved, warnings


def _days(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _months(start: date, end: date) -> Iterator[date]:
    current = _month_start(start)
    final = _month_start(end)
    while current <= final:
        yield current
        current = (
            date(current.year + 1, 1, 1)
            if current.month == 12
            else date(current.year, current.month + 1, 1)
        )


def target_resolution(request: DownloadRequest) -> float:
    value = request.options.get("target_resolution")
    if value in (None, ""):
        value = 0.05 if request.options.get("resolution", "high") == "high" else 0.5
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("target_resolution must be a positive number in degrees") from exc
    if not math.isfinite(result) or result <= 0 or result > 10:
        raise ConfigurationError("target_resolution must be between 0 and 10 degrees")
    return result


def _target_shape(request: DownloadRequest) -> tuple[int, int]:
    resolution = target_resolution(request)
    def cells(span: float) -> int:
        ratio = span / resolution
        # Decimal geographic bounds rarely divide exactly in binary floating
        # point; do not invent an extra row/column for a near-integer ratio.
        return max(1, math.ceil(ratio - 1e-12 * max(1.0, abs(ratio))))

    width = cells(request.bbox.width)
    height = cells(request.bbox.north - request.bbox.south)
    if width * height > MAX_RASTER_PIXELS:
        raise ConfigurationError(
            f"Target grid has {width * height:,} cells; increase target_resolution"
        )
    return width, height


def _job_options(request: DownloadRequest) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    frequency = str(request.options.get("frequency", "clte"))
    levtype = str(request.options.get("levtype", "sfc"))
    levels = _items(request.options.get("levels"))
    times = ("0000",) if frequency == "clmn" else expand_times(
        request.options.get("times", "0000/0100/to/2300")
    )
    return frequency, levtype, levels, times


def iter_jobs(request: DownloadRequest) -> Iterator[IfsJob]:
    """Yield one bounded day/month request at a time; never materialize decades."""

    assert request.time is not None
    frequency, levtype, levels, times = _job_options(request)
    experiments = _items(request.options.get("experiment", ("historical",)))
    periods, _warnings = resolve_experiment_periods(
        request.time,
        experiments,
        frequency=frequency,
        allow_partial=request.allow_partial,
    )
    params = "/".join(PARAMETERS[name].param_id(frequency) for name in request.variables)
    west, south, east, north = request.bbox.as_tuple()
    resolution = str(request.options.get("resolution", "high"))
    for experiment, start, end in periods:
        spec = EXPERIMENTS[experiment]
        base = {
            "activity": spec.activity,
            "class": "d1",
            "dataset": "climate-dt",
            "experiment": spec.code,
            "expver": "0001",
            "generation": "2",
            "model": "IFS-FESOM",
            "realization": "1",
            "resolution": resolution,
            "type": "fc",
            "levtype": levtype,
            "param": params,
            # Polytope/MARS area order is north/west/south/east.
            "area": f"{north:g}/{west:g}/{south:g}/{east:g}",
        }
        if levels:
            base["levelist"] = "/".join(levels)
        dates = _months(start, end) if frequency == "clmn" else _days(start, end)
        for current in dates:
            payload = dict(base)
            if frequency == "clmn":
                payload.update(
                    stream="clmn",
                    year=f"{current.year:04d}",
                    month=f"{current.month:02d}",
                    time="0000",
                )
            else:
                payload.update(
                    stream="clte",
                    date=current.strftime("%Y%m%d"),
                    time="/".join(times),
                )
            yield IfsJob(
                experiment=experiment,
                period=current,
                request=payload,
                variables=request.variables,
                expected_times=times,
                levels=levels,
                frequency=frequency,
            )


def count_jobs(request: DownloadRequest) -> int:
    assert request.time is not None
    frequency = str(request.options.get("frequency", "clte"))
    periods, _warnings = resolve_experiment_periods(
        request.time,
        _items(request.options.get("experiment", ("historical",))),
        frequency=frequency,
        allow_partial=request.allow_partial,
    )
    count = 0
    for _experiment, start, end in periods:
        if frequency == "clmn":
            count += (end.year - start.year) * 12 + end.month - start.month + 1
        else:
            count += (end - start).days + 1
    return count


def resolve_polytope_credentials(
    supplied: Mapping[str, str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    rc_path: Path | None = None,
) -> dict[str, str]:
    """Resolve only the caller's request, environment, or standard rc file."""

    supplied = {} if supplied is None else supplied
    environ = os.environ if environ is None else environ

    def credential(source: Mapping[str, Any], name: str) -> str | None:
        value = source.get(name)
        if value is None:
            return None
        text = value if isinstance(value, str) else str(value)
        return text if text.strip() else None

    values = {
        "user_email": credential(supplied, "user_email")
        or credential(environ, "POLYTOPE_USER_EMAIL"),
        "user_key": credential(supplied, "user_key")
        or credential(environ, "POLYTOPE_USER_KEY"),
    }
    path = rc_path or (Path.home() / ".polytopeapirc")
    if (not values["user_key"] or not values["user_email"]) and path.is_file():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AuthenticationError(f"Polytope credential file is invalid: {path}") from exc
        if not values["user_key"]:
            values["user_key"] = credential(config, "user_key")
        if not values["user_email"]:
            values["user_email"] = credential(config, "user_email")
    if not values["user_key"]:
        raise AuthenticationError(
            "Provide a Polytope key with request credentials, POLYTOPE_USER_KEY, "
            "or ~/.polytopeapirc"
        )
    return {name: value for name, value in values.items() if value is not None and value != ""}


def _complete_grib(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 8:
            return False
        with path.open("rb") as handle:
            if handle.read(4) != b"GRIB":
                return False
            handle.seek(-4, os.SEEK_END)
            return handle.read(4) == b"7777"
    except OSError:
        return False


def _metadata(field: Any, key: str, default: Any = None) -> Any:
    try:
        value = field.metadata(key)
    except Exception:
        return default
    return default if value is None else value


def _valid_datetime(field: Any) -> tuple[datetime, str]:
    value = _metadata(field, "valid_datetime")
    if value is not None:
        if isinstance(value, datetime):
            result = value
        else:
            text = str(value).strip().replace("Z", "+00:00")
            try:
                result = datetime.fromisoformat(text)
            except ValueError as exc:
                raise DownloadError(f"Polytope field has invalid valid_datetime={value!r}") from exc
    else:
        raw_date = _metadata(field, "validityDate")
        raw_time = _metadata(field, "validityTime", _metadata(field, "time"))
        if raw_date is None or raw_time is None:
            raise DownloadError("Polytope field is missing validity date/time metadata")
        try:
            result = datetime.strptime(
                str(raw_date).strip() + f"{int(raw_time):04d}", "%Y%m%d%H%M"
            )
        except (ValueError, TypeError) as exc:
            raise DownloadError(
                f"Polytope field has invalid validity date/time {raw_date!r}/{raw_time!r}"
            ) from exc
    return result, result.strftime("%H%M")


def _field_levtype(field: Any) -> str:
    value = _metadata(field, "levtype")
    if value is None:
        value = _metadata(field, "typeOfLevel")
    if value is None:
        raise DownloadError("Polytope field is missing level-type metadata")
    aliases = {
        "surface": "sfc",
        "heightaboveground": "sfc",
        "isobaricinhpa": "pl",
        "isobaricinpa": "pl",
        "hybrid": "hl",
    }
    text = str(value).strip()
    return aliases.get(text.lower(), text.lower())


def _field_level(field: Any, levtype: str) -> str:
    value = _metadata(field, "level")
    if levtype == "sfc":
        if value in (None, "", 0, "0"):
            return ""
    if value is None or value == "":
        raise DownloadError("Polytope field is missing level metadata")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    return str(int(number)) if number.is_integer() else f"{number:g}"


def describe_fields(fieldlist: Any, job: IfsJob, *, allow_partial: bool) -> tuple[FieldDescriptor, ...]:
    """Validate metadata completeness before any output is published."""

    requested_ids = {
        PARAMETERS[name].param_id(job.frequency): name for name in job.variables
    }
    requested_names = {
        PARAMETERS[name].short_name(job.frequency): name for name in job.variables
    }
    descriptors: list[FieldDescriptor] = []
    signatures: set[tuple[str, str, str, str, str]] = set()
    for field in fieldlist:
        param_id = str(_metadata(field, "paramId", ""))
        variable = requested_ids.get(param_id)
        if variable is None:
            short_name = str(_metadata(field, "shortName", ""))
            variable = requested_names.get(short_name) if not param_id else None
            if variable is not None and not param_id:
                param_id = PARAMETERS[variable].param_id(job.frequency)
            else:
                identity = param_id or short_name or "<unknown>"
                raise DownloadError(f"Polytope returned unexpected parameter {identity!r}")
        timestamp, time_code = _valid_datetime(field)
        if job.frequency == "clte":
            if timestamp.date() != job.period or time_code not in job.expected_times:
                raise DownloadError(
                    f"Polytope returned unexpected CLTE time {timestamp.isoformat()} "
                    f"for {job.period}"
                )
            period_code = timestamp.strftime("%Y%m%d")
        else:
            if (timestamp.year, timestamp.month) != (job.period.year, job.period.month):
                raise DownloadError(
                    f"Polytope returned unexpected CLMN month {timestamp:%Y-%m} "
                    f"for {job.period:%Y-%m}"
                )
            if time_code != "0000":
                raise DownloadError(
                    f"Polytope returned unexpected CLMN time {time_code}; expected 0000"
                )
            period_code = timestamp.strftime("%Y%m")
        levtype = _field_levtype(field)
        if levtype != job.request["levtype"]:
            raise DownloadError(
                f"Polytope returned levtype={levtype}; expected {job.request['levtype']}"
            )
        level_code = _field_level(field, levtype)
        expected_levels = (
            (SURFACE_LEVELS[variable],) if levtype == "sfc" else job.levels
        )
        if level_code not in expected_levels:
            raise DownloadError(
                f"Polytope returned unexpected level {level_code or 'surface'}"
            )
        signature = (period_code, time_code, param_id, levtype, level_code)
        if signature in signatures:
            raise OutputError(
                f"Polytope returned duplicate field "
                f"{variable}/{time_code}/{level_code or 'surface'}"
            )
        signatures.add(signature)
        level = None if levtype == "sfc" else level_code
        attrs = {
            "long_name": str(_metadata(field, "name", PARAMETERS[variable].title)),
            "units": str(_metadata(field, "units", PARAMETERS[variable].units)),
            "standard_name": PARAMETERS[variable].standard_name,
            "param_id": param_id,
            "type_of_level": levtype,
            "source_level": level_code or "surface",
        }
        descriptors.append(FieldDescriptor(field, variable, timestamp, time_code, level, attrs))

    expected = {
        (
            job.period.strftime("%Y%m" if job.frequency == "clmn" else "%Y%m%d"),
            time_code,
            PARAMETERS[name].param_id(job.frequency),
            job.request["levtype"],
            SURFACE_LEVELS[name] if job.request["levtype"] == "sfc" else level,
        )
        for name in job.variables
        for time_code in job.expected_times
        for level in (job.levels or ("",))
    }
    missing = sorted(expected - signatures)
    if missing and not allow_partial:
        example = ", ".join(
            "/".join(value or "surface" for value in item) for item in missing[:5]
        )
        raise DownloadError(
            f"Polytope response is incomplete for {job.period}: missing {example}"
            + (" ..." if len(missing) > 5 else "")
        )
    if not descriptors and not allow_partial:
        raise DownloadError(f"Polytope returned no requested fields for {job.period}")
    return tuple(descriptors)


def _field_arrays(field: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        latitudes, longitudes = field.geography.latlons()
        values = field.to_numpy(dtype=np.float32)
    except Exception as exc:
        raise OutputError(f"GRIB field has no usable geographic coordinates: {exc}") from exc
    if latitudes is None or longitudes is None:
        raise OutputError("GRIB field has no latitude/longitude coordinates")
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)
    values = np.asanyarray(values)
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    values = np.asarray(values, dtype="float32")
    if latitudes.size != values.size or longitudes.size != values.size:
        raise OutputError("GRIB coordinate and value sizes differ; rasterization is unsafe")
    return latitudes, longitudes, values


def _rectilinear(
    latitudes: np.ndarray, longitudes: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if values.ndim != 2:
        return None
    if latitudes.shape == values.shape and longitudes.shape == values.shape:
        if not np.allclose(latitudes, latitudes[:, :1], equal_nan=False):
            return None
        if not np.allclose(longitudes, longitudes[:1, :], equal_nan=False):
            return None
        return latitudes[:, 0], longitudes[0, :], values
    if latitudes.ndim == longitudes.ndim == 1:
        if values.shape == (latitudes.size, longitudes.size):
            return latitudes, longitudes, values
    return None


def _profile(request: DownloadRequest, dtype: str = "float32") -> tuple[dict[str, Any], Any]:
    width, height = _target_shape(request)
    transform = from_bounds(*request.bbox.as_tuple(), width, height)
    return (
        {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": 1,
            "dtype": dtype,
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": -9999.0,
            "tiled": True,
            "blockxsize": min(512, 16 * max(1, math.ceil(min(512, width) / 16))),
            "blockysize": min(512, 16 * max(1, math.ceil(min(512, height) / 16))),
            "compress": "deflate",
            "bigtiff": "IF_SAFER",
        },
        transform,
    )


def _validate_tiff(path: Path) -> None:
    try:
        with rasterio.open(path) as dataset:
            if dataset.count != 1 or dataset.width < 1 or dataset.height < 1:
                raise OutputError("Rasterized IFS field has an invalid shape")
            if not dataset.crs or dataset.crs.to_epsg() != 4326:
                raise OutputError("Rasterized IFS field is not in EPSG:4326")
    except OutputError:
        raise
    except Exception as exc:
        raise OutputError(f"Could not validate rasterized IFS field: {exc}") from exc


def _write_regular(
    path: Path,
    request: DownloadRequest,
    rectilinear: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    interpolation: str,
) -> Path:
    latitudes, longitudes, values = rectilinear
    latitudes = np.asarray(latitudes, dtype="float64")
    longitudes = ((np.asarray(longitudes, dtype="float64") + 180) % 360) - 180
    if latitudes.size < 2 or longitudes.size < 2:
        raise OutputError("A regular GRIB raster needs at least two coordinates per axis")
    lat_order = np.argsort(latitudes)[::-1]
    lon_order = np.argsort(longitudes)
    latitudes, longitudes = latitudes[lat_order], longitudes[lon_order]
    values = values[np.ix_(lat_order, lon_order)]
    dy = float(np.median(np.abs(np.diff(latitudes))))
    dx = float(np.median(np.abs(np.diff(longitudes))))
    if dx <= 0 or dy <= 0 or not np.allclose(np.abs(np.diff(latitudes)), dy, rtol=1e-4):
        raise OutputError("Regridded latitude coordinates are not a regular grid")
    if not np.allclose(np.diff(longitudes), dx, rtol=1e-4):
        raise OutputError("Regridded longitude coordinates are not a regular grid")
    source_transform = from_origin(
        longitudes[0] - dx / 2,
        latitudes[0] + dy / 2,
        dx,
        dy,
    )
    profile, destination_transform = _profile(request)
    destination = np.full((profile["height"], profile["width"]), -9999.0, dtype="float32")
    reproject(
        source=values,
        destination=destination,
        src_transform=source_transform,
        src_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_transform=destination_transform,
        dst_crs="EPSG:4326",
        dst_nodata=-9999.0,
        resampling=Resampling.nearest if interpolation == "nearest" else Resampling.bilinear,
    )
    if not np.any(destination != -9999.0):
        raise OutputError("Regridded field does not cover the requested extent")
    with atomic_path(path, suffix=".part.tif") as temporary:
        with rasterio.open(temporary, "w", **profile) as target:
            target.write(destination, 1)
        _validate_tiff(temporary)
    return path


def _xyz(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(latitudes)
    lon = np.deg2rad(longitudes)
    cosine = np.cos(lat)
    return np.column_stack((cosine * np.cos(lon), cosine * np.sin(lon), np.sin(lat)))


def _write_nearest_fallback(
    path: Path,
    request: DownloadRequest,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
    cancel: CancellationToken,
) -> Path:
    """Validated, row-chunked spherical nearest-neighbour fallback."""

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise DependencyError(
            "Irregular IFS grids require earthkit-regrid, or scipy for nearest fallback"
        ) from exc
    latitudes = np.asarray(latitudes, dtype="float64").reshape(-1)
    longitudes = ((np.asarray(longitudes, dtype="float64").reshape(-1) + 180) % 360) - 180
    values = np.asarray(values, dtype="float32").reshape(-1)
    valid = np.isfinite(latitudes) & np.isfinite(longitudes) & np.isfinite(values)
    valid &= (latitudes >= -90) & (latitudes <= 90)
    # Polytope normally applies ``area`` server-side. Keep a small halo here
    # as a defensive bound in case a service response still contains a global
    # HEALPix field, avoiding a needlessly huge spatial index.
    resolution = target_resolution(request)
    halo = max(1.0, resolution * 4)
    valid &= (latitudes >= request.bbox.south - halo) & (
        latitudes <= request.bbox.north + halo
    )
    valid &= (longitudes >= request.bbox.west - halo) & (
        longitudes <= request.bbox.east + halo
    )
    if np.count_nonzero(valid) < 4:
        raise OutputError("Too few valid unstructured points to rasterize safely")
    latitudes, longitudes, values = latitudes[valid], longitudes[valid], values[valid]
    tree = cKDTree(_xyz(latitudes, longitudes), compact_nodes=True, balanced_tree=True)

    # Estimate source spacing and reject extrapolation beyond a conservative
    # neighbourhood. This is what keeps the fallback from painting holes or an
    # unsupported ocean mesh as if it had coverage.
    sample = np.linspace(0, len(values) - 1, min(4096, len(values)), dtype=int)
    distances, _ = tree.query(_xyz(latitudes[sample], longitudes[sample]), k=2, workers=1)
    nearest = distances[:, 1]
    spacing = math.degrees(2 * math.asin(min(1.0, float(np.median(nearest)) / 2)))
    maximum = max(spacing * 4, resolution * math.sqrt(2) * 3)

    profile, transform = _profile(request)
    width, height = profile["width"], profile["height"]
    x = transform.c + (np.arange(width) + 0.5) * transform.a
    with atomic_path(path, suffix=".part.tif") as temporary:
        with rasterio.open(temporary, "w", **profile) as target:
            for row in range(0, height, 256):
                cancel.check()
                rows = min(256, height - row)
                y = transform.f + (np.arange(row, row + rows) + 0.5) * transform.e
                target_lon, target_lat = np.meshgrid(x, y)
                chord, indices = tree.query(
                    _xyz(target_lat.reshape(-1), target_lon.reshape(-1)),
                    k=1,
                    workers=1,
                )
                angle = np.rad2deg(2 * np.arcsin(np.minimum(1, chord / 2)))
                if np.any(angle > maximum):
                    raise OutputError(
                        "Unstructured field does not safely cover the target grid; "
                        "refusing nearest-neighbour extrapolation"
                    )
                block = values[indices].reshape(rows, width)
                target.write(block, 1, window=((row, row + rows), (0, width)))
        _validate_tiff(temporary)
    return path


def _earthkit_regrid(field: Any, request: DownloadRequest, interpolation: str) -> Any | None:
    try:
        import earthkit.regrid as ekr
    except ImportError:
        return None
    resolution = target_resolution(request)
    west, south, east, north = request.bbox.as_tuple()
    grid = {"grid": [resolution, resolution], "area": [north, west, south, east]}
    try:
        if hasattr(ekr, "regrid"):
            result = ekr.regrid(field, grid=grid, interpolation=interpolation)
        elif hasattr(ekr, "interpolate"):
            method = "nearest-neighbour" if interpolation == "nearest" else "linear"
            result = ekr.interpolate(field, out_grid=grid, method=method)
        else:
            return None
        if hasattr(result, "to_fieldlist"):
            result = result.to_fieldlist()
        if hasattr(result, "__len__") and not hasattr(result, "geography"):
            return result[0] if len(result) else None
        return result
    except (RuntimeError, TypeError, ValueError):
        return None


def rasterize_field(
    field: Any,
    path: Path,
    request: DownloadRequest,
    cancel: CancellationToken,
) -> Path:
    interpolation = str(request.options.get("interpolation", "nearest"))
    latitudes, longitudes, values = _field_arrays(field)
    regular = _rectilinear(latitudes, longitudes, values)
    if regular is not None:
        return _write_regular(path, request, regular, interpolation=interpolation)

    regridded = _earthkit_regrid(field, request, interpolation)
    if regridded is not None:
        r_lat, r_lon, r_values = _field_arrays(regridded)
        regular = _rectilinear(r_lat, r_lon, r_values)
        if regular is not None:
            return _write_regular(path, request, regular, interpolation=interpolation)
    if interpolation != "nearest":
        raise OutputError(
            "Linear interpolation of this unstructured IFS grid requires a compatible "
            "earthkit-regrid backend; no safe fallback is available"
        )
    return _write_nearest_fallback(path, request, latitudes, longitudes, values, cancel)


class IfsFesomProvider(Provider):
    spec = DatasetSpec(
        id="ifs_fesom",
        title="DestinE Climate DT IFS-FESOM Generation 2",
        summary="Hourly atmosphere and monthly climate fields from the coupled IFS-FESOM simulations.",
        variables=tuple(
            VariableSpec(name, parameter.title, parameter.units, parameter.standard_name)
            for name, parameter in PARAMETERS.items()
        ),
        credentials=(
            CredentialField(
                "user_email",
                "Polytope user email",
                environment=("POLYTOPE_USER_EMAIL",),
                secret=False,
                required=False,
            ),
            CredentialField(
                "user_key",
                "Polytope user key",
                environment=("POLYTOPE_USER_KEY",),
                required=False,
                description="Use your own DestinE/Polytope key or ~/.polytopeapirc.",
            ),
        ),
        options=(
            OptionSpec(
                "experiment",
                "Experiment",
                default=("historical",),
                choices=tuple(EXPERIMENTS),
                required=True,
                multiple=True,
            ),
            OptionSpec("frequency", "Frequency", default="clte", choices=("clte", "clmn")),
            OptionSpec(
                "levtype",
                "Level type",
                default="sfc",
                choices=("sfc", "pl", "hl"),
            ),
            OptionSpec("levels", "Levels", default=(), multiple=True),
            OptionSpec(
                "times",
                "Times",
                default="0000/0100/to/2300",
                description="HHMM values or a MARS range such as 0000/0100/to/2300.",
            ),
            OptionSpec("resolution", "Model resolution", default="high", choices=("high", "standard")),
            OptionSpec(
                "address",
                "Polytope endpoint",
                default=DEFAULT_ADDRESS,
                description="Override only when DestinE publishes another Polytope endpoint.",
            ),
            OptionSpec(
                "target_resolution",
                "Target grid resolution (degrees)",
                kind="number",
                default=None,
                description="Defaults to 0.05 for high and 0.5 for standard model output.",
            ),
            OptionSpec(
                "interpolation",
                "Interpolation",
                default="nearest",
                choices=("nearest", "linear"),
            ),
            OptionSpec("retries", "Retrieval attempts", kind="integer", default=3),
        ),
        source_url="https://platform.destine.eu/",
        citation="Destination Earth Climate Change Adaptation Digital Twin, Generation 2.",
        license="Destination Earth data licence and applicable ECMWF terms.",
    )

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        if request.bbox.crosses_antimeridian:
            raise ConfigurationError("IFS-FESOM regional output cannot cross the antimeridian")
        experiments = _items(self.option(request, "experiment"))
        if not experiments:
            raise ConfigurationError("Select at least one IFS-FESOM experiment")
        levtype = str(self.option(request, "levtype"))
        invalid = [name for name in request.variables if levtype not in PARAMETERS[name].levtypes]
        if invalid:
            raise ConfigurationError(
                f"Variable(s) {', '.join(invalid)} are not valid for levtype={levtype}"
            )
        levels = _items(self.option(request, "levels"))
        if levtype in {"pl", "hl"} and not levels:
            raise ConfigurationError(f"levels is required for levtype={levtype}")
        if levtype == "sfc" and levels:
            raise ConfigurationError("levels is not valid for levtype=sfc")
        invalid_levels = sorted(set(levels) - PRESSURE_LEVELS) if levtype == "pl" else []
        if invalid_levels:
            raise ConfigurationError(
                "Unsupported pressure level(s): " + ", ".join(invalid_levels)
            )
        if levtype == "hl" and levels != ("100",):
            raise ConfigurationError("levtype=hl supports only level 100")
        if str(self.option(request, "frequency")) == "clte":
            expand_times(self.option(request, "times"))
        try:
            retries = int(self.option(request, "retries"))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("retries must be a positive integer") from exc
        if retries < 1:
            raise ConfigurationError("retries must be a positive integer")
        target_resolution(request)
        _target_shape(request)
        address = self.option(request, "address")
        if not isinstance(address, str) or not address.strip():
            raise ConfigurationError("address must be a non-empty Polytope endpoint")
        if any(marker in address for marker in ("@", "?", "#")):
            raise ConfigurationError(
                "address must not contain user information, a query, or a fragment"
            )
        if count_jobs(request) == 0:
            raise ConfigurationError("Requested dates do not overlap the selected experiments")

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        jobs = count_jobs(request)
        frequency, _levtype, levels, times = _job_options(request)
        assert request.time is not None
        _periods, warnings = resolve_experiment_periods(
            request.time,
            _items(self.option(request, "experiment")),
            frequency=frequency,
            allow_partial=request.allow_partial,
        )
        per_job = len(request.variables) * len(levels or ("",)) * len(times)
        slices = jobs * per_job
        width, height = _target_shape(request)
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=jobs,
            slice_count=slices,
            estimated_bytes=width * height * 4 * slices,
            details={
                "request_granularity": "month" if frequency == "clmn" else "day",
                "stream": frequency,
                "parameters": {
                    name: PARAMETERS[name].param_id(frequency)
                    for name in request.variables
                },
                "experiments": [
                    {
                        "name": name,
                        "activity": EXPERIMENTS[name].activity,
                        "experiment": EXPERIMENTS[name].code,
                    }
                    for name in _items(self.option(request, "experiment"))
                ],
                "target_resolution": target_resolution(request),
                "target_shape": [height, width],
                "interpolation": self.option(request, "interpolation"),
                "regridding": {
                    "preferred": "earthkit-regrid",
                    "nearest_fallback": "scipy.spatial.cKDTree",
                },
                "address": self.option(request, "address"),
                "collection": COLLECTION,
            },
            warnings=warnings,
        )

    @staticmethod
    def _retrieve(
        ekd: Any,
        job: IfsJob,
        destination: Path,
        auth: Mapping[str, str],
        address: str,
        retries: int,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Path:
        for attempt in range(1, retries + 1):
            cancel.check()
            try:
                with atomic_path(destination, suffix=".part.grib") as temporary:
                    source = ekd.from_source(
                        "polytope",
                        COLLECTION,
                        request=dict(job.request),
                        address=address,
                        stream=True,
                        **dict(auth),
                    )
                    source.to_target("file", str(temporary))
                    cancel.check()
                    if not _complete_grib(temporary):
                        raise DownloadError("Polytope response is not a complete GRIB file")
                return destination
            except Exception as exc:
                if attempt >= retries:
                    if isinstance(exc, DownloadError):
                        raise
                    raise DownloadError(
                        f"Polytope retrieval failed for {job.period} "
                        f"({type(exc).__name__}); credentials were not logged"
                    ) from exc
                progress(
                    ProgressEvent(
                        "download",
                        f"Retrying {job.period} after failed attempt {attempt}",
                    )
                )
                remaining = min(30.0, float(2 ** (attempt - 1)))
                while remaining > 0:
                    cancel.check()
                    delay = min(0.5, remaining)
                    time.sleep(delay)
                    remaining -= delay
        raise AssertionError("unreachable")

    def slices(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Iterator[RasterSlice]:
        self.validate(request)
        require_module("earthkit.data", "destine")
        import earthkit.data as ekd

        auth = resolve_polytope_credentials(request.credentials)
        retries = int(self.option(request, "retries"))
        address = str(self.option(request, "address"))
        total = count_jobs(request)
        assert request.time is not None
        frequency = str(self.option(request, "frequency"))
        _periods, warnings = resolve_experiment_periods(
            request.time,
            _items(self.option(request, "experiment")),
            frequency=frequency,
            allow_partial=request.allow_partial,
        )
        for warning in warnings:
            progress(ProgressEvent("warning", warning))
        for job_index, job in enumerate(iter_jobs(request), 1):
            cancel.check()
            progress(
                ProgressEvent(
                    "download",
                    f"IFS-FESOM {job.experiment} {job.period}",
                    current=job_index,
                    total=total,
                )
            )
            grib = workspace / f"ifs_fesom_{job.experiment}_{job.period:%Y%m%d}.grib"
            self._retrieve(ekd, job, grib, auth, address, retries, progress, cancel)
            try:
                fields = ekd.from_source("file", str(grib)).to_fieldlist()
                descriptors = describe_fields(fields, job, allow_partial=request.allow_partial)
                expected_fields = (
                    len(job.variables) * len(job.expected_times) * len(job.levels or ("",))
                )
                if len(descriptors) < expected_fields:
                    progress(
                        ProgressEvent(
                            "warning",
                            f"Skipping {expected_fields - len(descriptors)} missing "
                            f"IFS-FESOM field(s) for {job.period}.",
                        )
                    )
            except (DownloadError, OutputError):
                raise
            except Exception as exc:
                raise DownloadError(f"Could not inspect GRIB response for {job.period}: {exc}") from exc

            try:
                for field_index, descriptor in enumerate(descriptors, 1):
                    cancel.check()
                    name = "_".join(
                        (
                            safe_component(descriptor.variable),
                            safe_component(descriptor.timestamp.isoformat()),
                            safe_component(descriptor.level or "surface"),
                            str(field_index),
                        )
                    )
                    raster = workspace / f"{name}.tif"
                    progress(
                        ProgressEvent(
                            "transform",
                            f"Rasterizing {descriptor.variable} {descriptor.timestamp.isoformat()}",
                            current=field_index,
                            total=len(descriptors),
                        )
                    )
                    rasterize_field(descriptor.field, raster, request, cancel)
                    dimensions: dict[str, Any] = {
                        "experiment": job.experiment,
                        "realization": str(job.request["realization"]),
                        "model_resolution": str(job.request["resolution"]),
                    }
                    if descriptor.level is not None:
                        dimensions["level"] = descriptor.level
                    try:
                        yield RasterSlice(
                            variable=descriptor.variable,
                            source=raster,
                            time=descriptor.timestamp,
                            dimensions=dimensions,
                            attrs={
                                **descriptor.attrs,
                                "source": "DestinE Climate DT Generation 2 IFS-FESOM",
                                "interpolation": self.option(request, "interpolation"),
                                "target_resolution_degrees": target_resolution(request),
                            },
                        )
                    finally:
                        raster.unlink(missing_ok=True)
            finally:
                grib.unlink(missing_ok=True)
