"""Small immutable objects shared by the Python API and user interfaces."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from geodaten.common.errors import ConfigurationError


class OutputFormat(str, Enum):
    """Supported final output families."""

    TIF = "tif"
    NETCDF = "netcdf"


@dataclass(frozen=True)
class BBox:
    """A WGS84 bounding box in west, south, east, north order.

    ``west > east`` represents a box crossing the antimeridian.
    """

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        values = tuple(map(float, (self.west, self.south, self.east, self.north)))
        if not all(math.isfinite(value) for value in values):
            raise ConfigurationError("Bounding-box coordinates must be finite numbers")
        west, south, east, north = values
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ConfigurationError("Longitude must be between -180 and 180 degrees")
        if not (-90 <= south < north <= 90):
            raise ConfigurationError("Latitude must satisfy -90 <= SOUTH < NORTH <= 90")
        if west == east:
            raise ConfigurationError("WEST and EAST must differ; use -180 180 for the globe")
        object.__setattr__(self, "west", west)
        object.__setattr__(self, "south", south)
        object.__setattr__(self, "east", east)
        object.__setattr__(self, "north", north)

    @classmethod
    def from_values(cls, values: Sequence[float]) -> BBox:
        if len(values) != 4:
            raise ConfigurationError("A bounding box requires WEST SOUTH EAST NORTH")
        return cls(*map(float, values))

    @property
    def crosses_antimeridian(self) -> bool:
        return self.west > self.east

    @property
    def width(self) -> float:
        return self.east - self.west if not self.crosses_antimeridian else 360 - self.west + self.east

    def segments(self) -> tuple[BBox, ...]:
        if not self.crosses_antimeridian:
            return (self,)
        return (
            BBox(self.west, self.south, 180.0, self.north),
            BBox(-180.0, self.south, self.east, self.north),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.west, self.south, self.east, self.north


@dataclass(frozen=True)
class TimeRange:
    """Provider-resolved inclusive temporal bounds."""

    start: str = "earliest"
    end: str = "latest"

    def __post_init__(self) -> None:
        start = str(self.start).strip()
        end = str(self.end).strip()
        if not start or not end:
            raise ConfigurationError("Temporal bounds cannot be empty")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class VariableSpec:
    id: str
    title: str
    units: str | None = None
    standard_name: str | None = None
    description: str = ""


@dataclass(frozen=True)
class CredentialField:
    id: str
    title: str
    environment: tuple[str, ...] = ()
    secret: bool = True
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class OptionSpec:
    id: str
    title: str
    kind: str = "text"
    default: Any = None
    choices: tuple[Any, ...] = ()
    required: bool = False
    multiple: bool = False
    description: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    title: str
    summary: str
    variables: tuple[VariableSpec, ...]
    temporal: bool = True
    credentials: tuple[CredentialField, ...] = ()
    options: tuple[OptionSpec, ...] = ()
    source_url: str = ""
    citation: str = ""
    license: str = ""
    group: str = ""

    @property
    def variable_ids(self) -> tuple[str, ...]:
        return tuple(variable.id for variable in self.variables)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class DownloadRequest:
    """One complete dataset request, independent of its user interface."""

    dataset: str
    bbox: BBox
    variables: tuple[str, ...]
    output_format: OutputFormat
    output_dir: Path
    time: TimeRange | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    credentials: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)
    cache_dir: Path | None = None
    overwrite: bool = False
    workers: int = 4
    allow_partial: bool = False

    def __post_init__(self) -> None:
        dataset = self.dataset.strip().lower()
        if not dataset:
            raise ConfigurationError("Dataset identifier cannot be empty")
        variables = tuple(dict.fromkeys(value.strip() for value in self.variables if value.strip()))
        if not variables:
            raise ConfigurationError("Select at least one variable")
        if self.workers < 1:
            raise ConfigurationError("workers must be at least one")
        output_format = (
            self.output_format
            if isinstance(self.output_format, OutputFormat)
            else OutputFormat(str(self.output_format).lower())
        )
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "output_format", output_format)
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser() if self.cache_dir else None)
        object.__setattr__(self, "options", dict(self.options))
        object.__setattr__(self, "credentials", dict(self.credentials))

    def public_dict(self) -> dict[str, Any]:
        """Return a stable request representation that never contains secrets."""

        return {
            "dataset": self.dataset,
            "bbox": list(self.bbox.as_tuple()),
            "time": _json_value(self.time.__dict__) if self.time else None,
            "variables": list(self.variables),
            "format": self.output_format.value,
            "output_dir": str(self.output_dir),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "options": _json_value(self.options),
            "overwrite": self.overwrite,
            "workers": self.workers,
            "allow_partial": self.allow_partial,
        }

    @property
    def fingerprint(self) -> str:
        payload = dict(self.public_dict())
        payload.pop("output_dir", None)
        payload.pop("cache_dir", None)
        payload.pop("overwrite", None)
        payload.pop("workers", None)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class DownloadPlan:
    dataset: str
    period_count: int | None
    slice_count: int | None
    estimated_bytes: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "period_count": self.period_count,
            "slice_count": self.slice_count,
            "estimated_bytes": self.estimated_bytes,
            "details": _json_value(self.details),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RasterSlice:
    """A local or GDAL-readable, single-variable raster for one logical slice."""

    variable: str
    source: str | Path
    time: Any = None
    band: int = 1
    dimensions: Mapping[str, Any] = field(default_factory=dict)
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DownloadResult:
    request: DownloadRequest
    outputs: tuple[Path, ...]
    manifest: Path
    skipped: int = 0
