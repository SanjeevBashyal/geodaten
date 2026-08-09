"""GIMMS LAI4g v1.2 half-monthly LAI and QC subsets from Zenodo."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import requests

from geodaten.common.errors import ConfigurationError, DownloadError
from geodaten.common.gdal import translate_to_cog, validate_raster
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

ZENODO_RECORD_ID = 8281930
ZENODO_API = "https://zenodo.org/api/records/{record_id}"
FIRST_YEAR = 1982
CONSOLIDATED_QC_LAST_YEAR = 2003
FILL_VALUE = 65535
LAI_SCALE_FACTOR = 0.001

VARIANTS: dict[str, dict[str, Any]] = {
    "AVHRR_MODIS_consolidated": {
        "last_year": 2020,
        "archive_ranges": ((1982, 1990), (1991, 2000), (2001, 2010), (2011, 2020)),
        "member_prefix": "GIMMS_LAI4g_V1.2_",
    },
    "AVHRR_solely": {
        "last_year": 2015,
        "archive_ranges": ((1982, 1990), (1991, 2000), (2001, 2010), (2011, 2015)),
        "member_prefix": "GIMMS_LAI4g(AVHRR)_V1.2_",
    },
}


@dataclass(frozen=True, order=True)
class HalfMonth:
    year: int
    month: int
    half: int

    def __post_init__(self) -> None:
        try:
            date(self.year, self.month, 1)
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid half-month year/month: {self.year}-{self.month}"
            ) from exc
        if self.half not in (1, 2):
            raise ConfigurationError("Half-month index must be 1 or 2")

    @property
    def code(self) -> str:
        return f"{self.year:04d}{self.month:02d}{self.half:02d}"

    @property
    def nominal_date(self) -> date:
        return date(self.year, self.month, 1 if self.half == 1 else 16)

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-H{self.half}"


def parse_time_bound(value: str, *, lower: bool, last_year: int) -> HalfMonth:
    text = str(value).strip().lower()
    if text == "earliest":
        return HalfMonth(FIRST_YEAR, 1, 1)
    if text == "latest":
        return HalfMonth(last_year, 12, 2)
    parts = text.split("-")
    try:
        if len(parts) == 1:
            year = int(parts[0])
            return HalfMonth(year, 1 if lower else 12, 1 if lower else 2)
        if len(parts) == 2:
            year, month = map(int, parts)
            return HalfMonth(year, month, 1 if lower else 2)
        if len(parts) == 3:
            year, month, day = map(int, parts)
            date(year, month, day)
            return HalfMonth(year, month, 1 if day <= 15 else 2)
    except (ValueError, ConfigurationError) as exc:
        raise ConfigurationError(f"Invalid GIMMS temporal bound: {value!r}") from exc
    raise ConfigurationError(
        f"Invalid GIMMS temporal bound {value!r}; use earliest, latest, YYYY, "
        "YYYY-MM, or YYYY-MM-DD"
    )


def selected_periods(start: str, end: str, *, last_year: int) -> tuple[HalfMonth, ...]:
    first = parse_time_bound(start, lower=True, last_year=last_year)
    last = parse_time_bound(end, lower=False, last_year=last_year)
    coverage_first = HalfMonth(FIRST_YEAR, 1, 1)
    coverage_last = HalfMonth(last_year, 12, 2)
    if first > last:
        raise ConfigurationError(f"Start period {first.label} is later than {last.label}")
    if first < coverage_first or last > coverage_last:
        raise ConfigurationError(
            f"Requested {first.label} through {last.label}, but this variant covers "
            f"{coverage_first.label} through {coverage_last.label}"
        )
    return tuple(
        HalfMonth(year, month, half)
        for year in range(first.year, last.year + 1)
        for month in range(1, 13)
        for half in (1, 2)
        if first <= HalfMonth(year, month, half) <= last
    )


def source_filename(variant: str, period: HalfMonth) -> str:
    return f"{VARIANTS[variant]['member_prefix']}{period.code}.tif"


def archive_filename(variant: str, period: HalfMonth) -> str:
    for first, last in VARIANTS[variant]["archive_ranges"]:
        if first <= period.year <= last:
            return f"GIMMS_LAI4g_{variant}_{first}_{last}.zip"
    raise ConfigurationError(f"No GIMMS archive contains {period.label}")


def _record_files(session: requests.Session) -> tuple[Mapping[str, Any], dict[str, str]]:
    try:
        response = session.get(ZENODO_API.format(record_id=ZENODO_RECORD_ID), timeout=60)
        response.raise_for_status()
        record = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise DownloadError(f"Could not read Zenodo record {ZENODO_RECORD_ID}") from exc
    if str(record.get("metadata", {}).get("version", "")).upper() != "V1.2":
        raise DownloadError(f"Zenodo record {ZENODO_RECORD_ID} is not GIMMS LAI4g V1.2")
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
            raise DownloadError(f"Duplicate member {basename!r} in GIMMS archive")
        lookup[basename] = info.filename
    return lookup


def _extract_member(
    remote: Any,
    member: str,
    destination: Path,
    *,
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
        if copied != info.file_size or not validate_raster(temporary, bands=2):
            raise DownloadError(f"Extracted GIMMS source failed validation: {destination.name}")
    return destination


class GimmsLai4gProvider(Provider):
    spec = DatasetSpec(
        id="gimms_lai4g",
        title="GIMMS LAI4g v1.2",
        summary=(
            "Global half-monthly leaf area index and quality-control fields at "
            "1/12-degree resolution."
        ),
        group="LAI",
        variables=(
            VariableSpec(
                "lai",
                "Leaf area index",
                units="1",
                standard_name="leaf_area_index",
                description="Raw uint16 values with a 0.001 scale factor.",
            ),
            VariableSpec(
                "qc",
                "Quality-control flags",
                units="1",
                description=(
                    "Dataset quality-control band; the consolidated variant provides "
                    "QC through 2003."
                ),
            ),
        ),
        temporal=True,
        options=(
            OptionSpec(
                "variant",
                "Product variant",
                kind="choice",
                default="AVHRR_MODIS_consolidated",
                choices=tuple(VARIANTS),
                description=(
                    "Consolidated AVHRR/MODIS coverage through 2020 or "
                    "AVHRR-only coverage through 2015."
                ),
            ),
        ),
        source_url=f"https://zenodo.org/records/{ZENODO_RECORD_ID}",
        citation="Cao et al., GIMMS LAI4g, version 1.2.",
        license="CC BY 4.0",
    )

    def _periods(self, request: DownloadRequest) -> tuple[HalfMonth, ...]:
        variant = str(self.option(request, "variant"))
        assert request.time is not None
        return selected_periods(
            request.time.start,
            request.time.end,
            last_year=int(VARIANTS[variant]["last_year"]),
        )

    @staticmethod
    def _variables_for_period(
        request: DownloadRequest, variant: str, period: HalfMonth
    ) -> tuple[str, ...]:
        return tuple(
            variable
            for variable in request.variables
            if not (
                variable == "qc"
                and variant == "AVHRR_MODIS_consolidated"
                and period.year > CONSOLIDATED_QC_LAST_YEAR
            )
        )

    def _unsupported_qc_periods(
        self, request: DownloadRequest, periods: tuple[HalfMonth, ...]
    ) -> tuple[HalfMonth, ...]:
        variant = str(self.option(request, "variant"))
        if "qc" not in request.variables or variant != "AVHRR_MODIS_consolidated":
            return ()
        return tuple(
            period for period in periods if period.year > CONSOLIDATED_QC_LAST_YEAR
        )

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        if request.bbox.crosses_antimeridian:
            raise ConfigurationError("GIMMS LAI4g subsets cannot cross the antimeridian")
        periods = self._periods(request)
        unsupported = self._unsupported_qc_periods(request, periods)
        if unsupported and not request.allow_partial:
            raise ConfigurationError(
                "The consolidated GIMMS LAI4g QC band is available only through 2003; "
                "use allow_partial to skip unsupported QC slices"
            )

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        variant = str(self.option(request, "variant"))
        periods = self._periods(request)
        supported = {
            period: self._variables_for_period(request, variant, period)
            for period in periods
        }
        active_periods = tuple(period for period, variables in supported.items() if variables)
        unsupported = self._unsupported_qc_periods(request, periods)
        archives = tuple(
            dict.fromkeys(archive_filename(variant, period) for period in active_periods)
        )
        warning = f"Skipping {len(unsupported)} consolidated QC slice(s) after 2003"
        if "lai" in request.variables:
            warning += "; requested LAI slices are unchanged"
        warnings = (warning + ".",) if unsupported else ()
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=len(periods),
            slice_count=sum(len(variables) for variables in supported.values()),
            details={
                "variant": variant,
                "first_period": periods[0].label,
                "last_period": periods[-1].label,
                "archives": archives,
                "source_member_count": len(active_periods),
                "skipped_qc_slices": len(unsupported),
                "consolidated_qc_last_period": "2003-12-H2",
                "native_resolution_degrees": 1.0 / 12.0,
                "zenodo_record_id": ZENODO_RECORD_ID,
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
        variant = str(self.option(request, "variant"))
        periods = self._periods(request)
        variables_by_period = {
            period: self._variables_for_period(request, variant, period)
            for period in periods
        }
        active_periods = tuple(
            period for period, variables in variables_by_period.items() if variables
        )
        unsupported = self._unsupported_qc_periods(request, periods)
        if unsupported:
            progress(
                ProgressEvent(
                    "warning",
                    f"Skipping {len(unsupported)} consolidated QC slice(s) after 2003",
                )
            )
        if not active_periods:
            return

        require_module("remotezip", "zenodo")
        from remotezip import RemoteZip

        grouped: dict[str, list[HalfMonth]] = {}
        for period in active_periods:
            grouped.setdefault(archive_filename(variant, period), []).append(period)
        source_dir = workspace / "gimms_lai4g" / "sources"
        slice_dir = workspace / "gimms_lai4g" / "slices"
        source_dir.mkdir(parents=True, exist_ok=True)
        slice_dir.mkdir(parents=True, exist_ok=True)
        variable_bands = {"lai": 1, "qc": 2}

        session = make_session()
        try:
            _record, files = _record_files(session)
            period_index = 0
            for archive, archive_periods in grouped.items():
                cancel.check()
                url = files.get(archive)
                if not url:
                    raise DownloadError(f"Official GIMMS record does not contain {archive}")
                progress(ProgressEvent("download", f"Opening GIMMS archive {archive}"))
                try:
                    with RemoteZip(url, session=session) as remote:
                        lookup = _member_lookup(remote)
                        for period in archive_periods:
                            period_index += 1
                            cancel.check()
                            filename = source_filename(variant, period)
                            member = lookup.get(filename)
                            if member is None:
                                raise DownloadError(
                                    f"GIMMS archive {archive} does not contain {filename}"
                                )
                            source = source_dir / filename
                            progress(
                                ProgressEvent(
                                    "download",
                                    f"Reading GIMMS {period.label}",
                                    current=period_index,
                                    total=len(active_periods),
                                )
                            )
                            try:
                                _extract_member(
                                    remote,
                                    member,
                                    source,
                                    progress=progress,
                                    cancel=cancel,
                                )
                                for variable in variables_by_period[period]:
                                    cancel.check()
                                    local = slice_dir / (
                                        f"{safe_component(variable)}_{period.code}.tif"
                                    )
                                    translate_to_cog(
                                        source,
                                        local,
                                        bbox=request.bbox,
                                        band=variable_bands[variable],
                                        metadata={
                                            "dataset": "GIMMS_LAI4g",
                                            "version": "V1.2",
                                            "variant": variant,
                                            "period": period.label,
                                        },
                                        nodata=FILL_VALUE,
                                        cancel=cancel,
                                    )
                                    attrs: dict[str, Any] = {
                                        "units": "1",
                                        "_FillValue": FILL_VALUE,
                                        "product_variant": variant,
                                        "source": (
                                            f"GIMMS LAI4g V1.2 Zenodo record {ZENODO_RECORD_ID}"
                                        ),
                                    }
                                    if variable == "lai":
                                        attrs.update(
                                            {
                                                "long_name": "leaf area index",
                                                "standard_name": "leaf_area_index",
                                                "scale_factor": LAI_SCALE_FACTOR,
                                            }
                                        )
                                    else:
                                        attrs["long_name"] = "quality-control flags"
                                    try:
                                        yield RasterSlice(
                                            variable=variable,
                                            source=local,
                                            time=period.nominal_date,
                                            attrs=attrs,
                                        )
                                    finally:
                                        local.unlink(missing_ok=True)
                            finally:
                                source.unlink(missing_ok=True)
                except DownloadError:
                    raise
                except Exception as exc:
                    raise DownloadError(f"Could not read GIMMS archive {archive}") from exc
        finally:
            session.close()
