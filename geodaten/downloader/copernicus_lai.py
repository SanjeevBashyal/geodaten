"""Copernicus Global Land LAI 300 m Version 2 COG subsets."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any
from urllib.parse import urlparse

import numpy as np

from geodaten.common import gdal
from geodaten.common.auth import child_environment, require_credentials
from geodaten.common.errors import ConfigurationError, DownloadError
from geodaten.common.io import make_session, safe_component
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


CATALOG_URL = (
    "https://s3.waw3-1.cloudferro.com/swift/v1/CatalogueCSV/"
    "bio-geophysical/vegetation_properties/lai_global_300m_10daily_v2/"
    "lai_global_300m_10daily_v2_cog.csv"
)
DEFAULT_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
AMS_ENDPOINT = "https://eodata.ams.dataspace.copernicus.eu"
_CATALOG_CACHE_SECONDS = 300.0

VARIABLE_CODES: dict[str, str] = {
    "lai": "LAI",
    "length_before": "LBEFORE",
    "length_after": "LAFTER",
    "nobs": "NOBS",
    "qflag": "QFLAG",
    "rmse": "RMSE",
}
CODE_VARIABLES = {code: variable for variable, code in VARIABLE_CODES.items()}


def _packing(maximum_dn: int, *, scale_factor: float = 1.0) -> dict[str, Any]:
    """CF packing metadata for the product's unsigned-byte digital numbers."""

    return {
        "scale_factor": scale_factor,
        "add_offset": 0.0,
        "_FillValue": 255,
        "missing_value": 255,
        "valid_range": np.asarray((0, maximum_dn), dtype="uint8"),
    }


VARIABLE_ATTRIBUTES: dict[str, dict[str, Any]] = {
    "lai": {
        "long_name": "leaf area index",
        "standard_name": "leaf_area_index",
        "units": "m2 m-2",
        **_packing(210, scale_factor=1.0 / 30.0),
    },
    "length_before": {
        "long_name": "length of the compositing period before the nominal date",
        "units": "days",
        **_packing(210),
    },
    "length_after": {
        "long_name": "length of the compositing period after the nominal date",
        "units": "days",
        **_packing(60),
    },
    "nobs": {
        "long_name": "number of available valid instantaneous LAI estimates",
        "units": "1",
        **_packing(60),
    },
    "qflag": {
        "long_name": "quality flags",
        "units": "1",
        "_FillValue": 255,
        "missing_value": 255,
        "flag_masks": np.asarray((1, 2, 16, 96, 96, 96, 96, 128), dtype="uint8"),
        "flag_values": np.asarray((1, 2, 16, 0, 32, 64, 96, 128), dtype="uint8"),
        "flag_meanings": (
            "water evergreen_broadleaf_forest previous_dekadal_or_missing "
            "second_degree_polynomial_fit linear_fit interpolation_nearest_15_days "
            "nearest_5_days_or_missing instantaneous_evergreen_broadleaf_classification"
        ),
    },
    "rmse": {
        "long_name": "leaf area index root mean square error",
        "units": "m2 m-2",
        **_packing(210, scale_factor=1.0 / 30.0),
    },
}

_VERSION_RE = re.compile(r"_V(\d+)\.(\d+)\.(\d+)(?:_|$)", re.IGNORECASE)
_REVISION_RE = re.compile(r"-RT(\d+)_", re.IGNORECASE)
_BAND_RE = re.compile(
    r"^c_gls_LAI300-(LAI|LBEFORE|LAFTER|NOBS|QFLAG|RMSE)(?:-RT\d+)?_",
    re.IGNORECASE,
)


def _parse_datetime(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


@dataclass(frozen=True)
class Product:
    product_id: str
    name: str
    content_length: int
    nominal_date: date
    modification_date: datetime
    s3_path: str

    @property
    def sensor(self) -> str:
        upper = self.name.upper()
        if "_OLCI_" in upper:
            return "OLCI"
        if "_PROBAV_" in upper:
            return "PROBAV"
        return "UNKNOWN"

    @property
    def version(self) -> tuple[int, int, int]:
        match = _VERSION_RE.search(self.name)
        return tuple(map(int, match.groups())) if match else (0, 0, 0)

    @property
    def revision(self) -> int | None:
        match = _REVISION_RE.search(self.name)
        return int(match.group(1)) if match else None


def parse_catalog(text: str) -> list[Product]:
    """Parse the official semicolon-delimited catalogue and validate its schema."""

    reader = csv.DictReader(text.lstrip("\ufeff").splitlines(), delimiter=";")
    required = {
        "id",
        "name",
        "content_length",
        "nominal_date",
        "modification_date",
        "s3_path",
    }
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise DownloadError(f"Unexpected Copernicus catalogue columns: {reader.fieldnames}")

    products: list[Product] = []
    for row in reader:
        try:
            nominal = datetime.fromisoformat(
                str(row["nominal_date"]).replace("Z", "+00:00")
            ).date()
            size = int(row["content_length"] or 0)
            s3_path = str(row["s3_path"]).strip()
            parse_s3_path(s3_path)
        except (TypeError, ValueError) as exc:
            raise DownloadError(
                f"Invalid Copernicus catalogue row for {row.get('name', '<unknown>')}"
            ) from exc
        products.append(
            Product(
                product_id=str(row["id"]).strip(),
                name=str(row["name"]).strip(),
                content_length=max(0, size),
                nominal_date=nominal,
                modification_date=_parse_datetime(str(row["modification_date"])),
                s3_path=s3_path,
            )
        )
    return products


def read_catalog(
    catalog_file: Path | None = None,
    *,
    url: str = CATALOG_URL,
    session: Any = None,
) -> list[Product]:
    """Read the public COG catalogue; dataset payload credentials are not used."""

    if catalog_file is not None:
        try:
            return parse_catalog(Path(catalog_file).read_text(encoding="utf-8-sig"))
        except OSError as exc:
            raise DownloadError(f"Could not read Copernicus catalogue file {catalog_file}") from exc

    own_session = session is None
    active = session or make_session()
    try:
        response = active.get(url, timeout=(30.0, 120.0))
        response.raise_for_status()
        return parse_catalog(response.text)
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError("Could not download the public Copernicus COG catalogue") from exc
    finally:
        if own_session:
            active.close()


def product_score(product: Product) -> tuple[Any, ...]:
    """Rank revisions without encoding a sensor transition date.

    A product without an RT marker is treated as consolidated.  Otherwise the
    highest available RT revision wins.  Version and catalogue modification
    time make the rule continue to work when the archive is reprocessed.
    """

    consolidated = product.revision is None
    revision = product.revision if product.revision is not None else -1
    return (
        product.version,
        consolidated,
        revision,
        product.modification_date,
        product.name,
    )


def select_products(products: Iterable[Product], start: date, end: date) -> list[Product]:
    """Choose one best/consolidated product for each nominal date."""

    grouped: dict[date, list[Product]] = defaultdict(list)
    for product in products:
        if start <= product.nominal_date <= end:
            grouped[product.nominal_date].append(product)
    return [max(grouped[value], key=product_score) for value in sorted(grouped)]


def _parse_boundary(value: str, *, end: bool, available: Sequence[date]) -> date:
    normalized = str(value).strip().lower()
    if normalized == "earliest":
        return min(available)
    if normalized == "latest":
        return max(available)
    if re.fullmatch(r"\d{4}", normalized):
        year = int(normalized)
        return date(year, 12, 31) if end else date(year, 1, 1)
    try:
        return date.fromisoformat(normalized)
    except ValueError as exc:
        raise ConfigurationError(
            f"Invalid temporal boundary {value!r}; use YYYY, YYYY-MM-DD, earliest, or latest"
        ) from exc


def resolve_date_window(
    products: Sequence[Product], request: DownloadRequest
) -> tuple[date, date]:
    if not products:
        raise DownloadError("The public Copernicus LAI catalogue is empty")
    if request.time is None:
        raise ConfigurationError("Copernicus LAI requires a temporal range")
    available = tuple(product.nominal_date for product in products)
    start = _parse_boundary(request.time.start, end=False, available=available)
    end = _parse_boundary(request.time.end, end=True, available=available)
    if start > end:
        raise ConfigurationError("Start date must be on or before end date")
    return start, end


def parse_s3_path(value: str) -> tuple[str, str]:
    parsed = urlparse(str(value))
    if parsed.scheme.lower() != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid S3 path: {value}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/") + "/"


def object_variable(key: str) -> str | None:
    filename = PurePosixPath(str(key)).name
    if not filename.lower().endswith((".tif", ".tiff")):
        return None
    match = _BAND_RE.match(filename)
    return CODE_VARIABLES.get(match.group(1).upper()) if match else None


def object_matches(key: str, variables: Sequence[str] | str) -> bool:
    """Return whether an S3 object is a requested data-layer COG."""

    requested = {variables} if isinstance(variables, str) else set(variables)
    return object_variable(key) in requested


def _make_s3_client(access_key: str, secret_key: str, endpoint: str, workers: int) -> Any:
    require_module("boto3", "cdse")
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="default",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 10, "mode": "standard"},
            connect_timeout=60,
            read_timeout=300,
            max_pool_connections=max(1, workers),
        ),
    )


def iter_product_objects(client: Any, bucket: str, prefix: str) -> Iterator[dict[str, Any]]:
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        yield from page.get("Contents", ())


def _gdal_environment(credentials: dict[str, str], endpoint: str) -> dict[str, str]:
    host = urlparse(endpoint).netloc
    return child_environment(
        {
            "AWS_ACCESS_KEY_ID": credentials["access_key"],
            "AWS_SECRET_ACCESS_KEY": credentials["secret_key"],
            "AWS_S3_ENDPOINT": host,
            "AWS_REGION": "default",
            "AWS_HTTPS": "YES",
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "AWS_NO_SIGN_REQUEST": "NO",
            "GDAL_HTTP_TCP_KEEPALIVE": "YES",
        }
    )


class CopernicusLaiProvider(Provider):
    spec = DatasetSpec(
        id="copernicus_lai",
        title="Copernicus Global Land LAI 300 m Version 2",
        summary=(
            "Ten-daily global leaf area index and supporting quality layers. "
            "Only requested windows are read from the official global COG assets."
        ),
        group="LAI",
        variables=tuple(
            VariableSpec(
                id=identifier,
                title=str(VARIABLE_ATTRIBUTES[identifier]["long_name"]),
                units=str(VARIABLE_ATTRIBUTES[identifier]["units"]),
                standard_name=VARIABLE_ATTRIBUTES[identifier].get("standard_name"),
            )
            for identifier in VARIABLE_CODES
        ),
        credentials=(
            CredentialField(
                "access_key",
                "CDSE S3 access key",
                ("CDSE_S3_ACCESS_KEY",),
                description="A user-generated CDSE S3 key; never stored by geodaten.",
            ),
            CredentialField(
                "secret_key",
                "CDSE S3 secret key",
                ("CDSE_S3_SECRET_KEY",),
                description="The secret paired with the user's CDSE S3 access key.",
            ),
        ),
        options=(
            OptionSpec(
                "endpoint",
                "CDSE S3 endpoint",
                default=DEFAULT_ENDPOINT,
                choices=(DEFAULT_ENDPOINT, AMS_ENDPOINT),
                description="Official global or Amsterdam CDSE EODATA endpoint.",
            ),
        ),
        source_url=CATALOG_URL,
        citation="Copernicus Global Land Service LAI 300 m Version 2",
        license="Copernicus free, full and open data policy",
    )

    def __init__(self) -> None:
        self._resolved: dict[
            str, tuple[float, tuple[tuple[Product, ...], date, date]]
        ] = {}

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        if request.bbox.crosses_antimeridian:
            raise ConfigurationError(
                "Copernicus LAI currently requires a bbox that does not cross the antimeridian"
            )
        if request.bbox.south < -60 or request.bbox.north > 80:
            raise ConfigurationError("Copernicus LAI Version 2 coverage is limited to 60°S–80°N")

    def _selection(self, request: DownloadRequest) -> tuple[tuple[Product, ...], date, date]:
        key = request.fingerprint
        cached = self._resolved.get(key)
        if cached is not None and monotonic() - cached[0] <= _CATALOG_CACHE_SECONDS:
            return cached[1]
        products = read_catalog()
        start, end = resolve_date_window(products, request)
        selected = tuple(select_products(products, start, end))
        if not selected:
            raise DownloadError("No Copernicus LAI products matched the requested time range")
        resolved = selected, start, end
        self._resolved.clear()
        self._resolved[key] = (monotonic(), resolved)
        return resolved

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        selected, start, end = self._selection(request)
        revisions: dict[str, int] = defaultdict(int)
        sensors: dict[str, int] = defaultdict(int)
        for product in selected:
            revision = "final" if product.revision is None else f"RT{product.revision}"
            revisions[revision] += 1
            sensors[product.sensor] += 1
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=len(selected),
            slice_count=len(selected) * len(request.variables),
            # Catalogue size is for every global layer. Range-request transfer
            # depends on the spatial window and is intentionally not guessed.
            estimated_bytes=None,
            details={
                "start": start.isoformat(),
                "end": end.isoformat(),
                "catalogue": CATALOG_URL,
                "catalogue_product_bytes": sum(item.content_length for item in selected),
                "selection": "highest current version and consolidated/RT revision per date",
                "revisions": dict(sorted(revisions.items())),
                "sensors": dict(sorted(sensors.items())),
                "native_grid": "EPSG:4326, 1/336 degree",
                "payload_source": "COG",
            },
            warnings=("Actual transfer size depends on GDAL COG range requests.",),
        )

    def slices(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Iterator[RasterSlice]:
        self.validate(request)
        gdal.require_commands("gdal_translate", "gdalinfo")
        credentials = require_credentials(self.spec.credentials, request.credentials)
        endpoint = str(self.option(request, "endpoint"))
        client = _make_s3_client(
            credentials["access_key"], credentials["secret_key"], endpoint, request.workers
        )
        environment = _gdal_environment(credentials, endpoint)
        products, _, _ = self._selection(request)
        self._resolved.pop(request.fingerprint, None)
        total = len(products) * len(request.variables)
        current = 0
        produced = 0

        for product in products:
            cancel.check()
            progress(ProgressEvent("discover", f"Listing COG layers for {product.nominal_date}"))
            bucket, prefix = parse_s3_path(product.s3_path)
            try:
                objects = list(iter_product_objects(client, bucket, prefix))
            except Exception as exc:
                raise DownloadError(f"Could not list CDSE product {product.name}") from exc
            if not objects:
                if request.allow_partial:
                    progress(ProgressEvent("warning", f"No COG objects for {product.nominal_date}"))
                    continue
                raise DownloadError(f"No COG objects found for Copernicus product {product.name}")

            matches: dict[str, list[str]] = defaultdict(list)
            for item in objects:
                key = str(item.get("Key", ""))
                variable = object_variable(key)
                if variable in request.variables:
                    matches[variable].append(key)

            for variable in request.variables:
                cancel.check()
                keys = sorted(set(matches.get(variable, ())))
                if not keys:
                    if request.allow_partial:
                        current += 1
                        progress(
                            ProgressEvent(
                                "warning",
                                f"Missing {variable} for {product.nominal_date}",
                                current=current,
                                total=total,
                            )
                        )
                        continue
                    raise DownloadError(
                        f"Copernicus product {product.name} has no COG for variable {variable}"
                    )
                if len(keys) > 1:
                    raise DownloadError(
                        f"Copernicus product {product.name} has multiple COGs for variable {variable}"
                    )

                key = keys[0]
                destination = (
                    workspace
                    / "slices"
                    / f"copernicus_lai_{product.nominal_date.isoformat()}_{safe_component(variable)}.tif"
                )
                current += 1
                progress(
                    ProgressEvent(
                        "download",
                        f"Reading {variable} COG window for {product.nominal_date}",
                        current=current,
                        total=total,
                    )
                )
                gdal.translate_to_cog(
                    f"/vsis3/{bucket}/{key}",
                    destination,
                    bbox=request.bbox,
                    metadata={
                        "dataset": self.spec.id,
                        "variable": variable,
                        "nominal_date": product.nominal_date.isoformat(),
                        "source_product": product.name,
                    },
                    nodata=VARIABLE_ATTRIBUTES[variable].get("_FillValue"),
                    scale=VARIABLE_ATTRIBUTES[variable].get("scale_factor"),
                    offset=VARIABLE_ATTRIBUTES[variable].get("add_offset"),
                    env=environment,
                    cancel=cancel,
                )
                produced += 1
                try:
                    # Range-read windows are request scratch data. The common
                    # writer consumes one synchronously, then this frees it.
                    yield RasterSlice(
                        variable=variable,
                        source=destination,
                        time=product.nominal_date,
                        attrs={
                            **VARIABLE_ATTRIBUTES[variable],
                            "source": "Copernicus Land Monitoring Service LAI 300 m Version 2 COG",
                            "references": (
                                "https://land.copernicus.eu/en/technical-library/"
                                "product-user-manual-leaf-area-index-300m-version-2"
                            ),
                        },
                    )
                finally:
                    destination.unlink(missing_ok=True)
        if not produced:
            raise DownloadError("No Copernicus LAI raster slices were produced")
