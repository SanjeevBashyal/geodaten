"""GLiM v1.1 global lithological map polygons cropped to the requested extent."""

from __future__ import annotations

import math
import os
import shutil
import uuid
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from geodaten.common.errors import ConfigurationError, DownloadError, OutputError
from geodaten.common.gdal import require_commands, validate_vector, vector_to_shapefile
from geodaten.common.io import (
    available_bytes,
    cache_root,
    copy_stream,
    make_session,
    stream_download,
)
from geodaten.common.models import (
    VECTOR_FORMATS,
    BBox,
    DatasetSpec,
    DownloadPlan,
    DownloadRequest,
    OptionSpec,
    VariableSpec,
    VectorSlice,
)
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent
from geodaten.downloader.base import Provider

SOURCE_PAGE = (
    "https://www.geo.uni-hamburg.de/en/geologie/forschung/aquatische-geochemie/glim.html"
)
#: Publisher-hosted download of the single ``LiMW_GIS 2015.gdb`` release.
ARCHIVE_URL = (
    "https://www.dropbox.com/scl/fi/5v00i8op7a9brmn4qeg8b/LiMW_GIS-2015.gdb.zip"
    "?rlkey=89q1aj8zfpmiv82y41xe6ttmp&e=1&dl=0"
)
ARCHIVE_NAME = "LiMW_GIS_2015.gdb.zip"
GEODATABASE_NAME = "limw_gis_2015.gdb"
#: Compressed and extracted sizes of the 2015 release, used for space checks only.
ARCHIVE_BYTES = 1_138_691_332
GEODATABASE_BYTES = 1_290_000_000

SOURCE_LAYER = "GLiM_export"
SOURCE_CRS = "ESRI:54012"
TARGET_CRS = "EPSG:4326"

# The published attribute table. ``Shape_Length``/``Shape_Area`` are unbounded
# doubles in the file geodatabase, and the ESRI Shapefile driver would otherwise
# default them to a DBF width that cannot hold the largest continental polygons.
_FIELDS: tuple[tuple[str, str | None, str], ...] = (
    ("IDENTITY_", None, "Source polygon identifier of the GLiM compilation"),
    ("Litho", None, "Full hierarchical GLiM lithology code"),
    ("xx", None, "Primary GLiM lithology class code"),
    ("Shape_Length", "numeric(24,6)", "Perimeter of the original polygon in Eckert IV metres"),
    ("Shape_Area", "numeric(24,6)", "Area of the original polygon in Eckert IV square metres"),
)
FIELD_NAMES: tuple[str, ...] = tuple(name for name, _cast, _description in _FIELDS)
#: Shapefile DBF names are limited to ten characters, so GDAL launders these two.
LAUNDERED_FIELDS = {"Shape_Length": "Shape_Leng"}
#: Geometry-derived fields describe the uncropped source polygon, not the output.
SOURCE_GEOMETRY_FIELDS = ("Shape_Length", "Shape_Area")

#: Primary lithology classes of the ``xx`` field, in GLiM v1.1 legend order.
CLASSES: tuple[tuple[str, str], ...] = (
    ("nd", "no data"),
    ("su", "unconsolidated sediments"),
    ("ss", "siliciclastic sedimentary rocks"),
    ("sm", "mixed sedimentary rocks"),
    ("sc", "carbonate sedimentary rocks"),
    ("py", "pyroclastics"),
    ("ev", "evaporites"),
    ("mt", "metamorphic rocks"),
    ("pa", "acid plutonic rocks"),
    ("pi", "intermediate plutonic rocks"),
    ("pb", "basic plutonic rocks"),
    ("va", "acid volcanic rocks"),
    ("vi", "intermediate volcanic rocks"),
    ("vb", "basic volcanic rocks"),
    ("ig", "ice and glaciers"),
    ("wb", "water bodies"),
)

# An ESRI Shapefile cannot exceed two gigabytes per component file; GLiM averages
# roughly 1.6 kB of geometry per polygon, so only near-global extents come close.
_LARGE_EXTENT_DEGREES_SQUARED = 20_000.0


def archive_url() -> str:
    """Return the publisher link rewritten to serve the file rather than a page."""

    parsed = urlsplit(ARCHIVE_URL)
    query = [(key, value) for key, value in parse_qsl(parsed.query) if key != "dl"]
    query.append(("dl", "1"))
    return urlunsplit(parsed._replace(query=urlencode(query)))


def selected_fields(request: DownloadRequest) -> tuple[str, ...]:
    """Resolve the attribute columns to carry into the shapefile."""

    value: Any = request.options.get("fields")
    if value is None:
        return FIELD_NAMES
    names = (
        (value,)
        if isinstance(value, str)
        else tuple(dict.fromkeys(str(item) for item in value))
    )
    unknown = tuple(name for name in names if name not in FIELD_NAMES)
    if unknown:
        raise ConfigurationError(
            "Unknown GLiM field(s): "
            + ", ".join(unknown)
            + "; choose from: "
            + ", ".join(FIELD_NAMES)
        )
    if not names:
        raise ConfigurationError("Select at least one GLiM attribute field")
    # Keep the published column order regardless of how the option was written.
    return tuple(name for name in FIELD_NAMES if name in names)


def source_sql(fields: tuple[str, ...], *, layer: str = SOURCE_LAYER) -> str:
    """Return the OGR SQL that selects one attribute set without losing precision.

    Columns are aliased to the names the shapefile driver would launder them to.
    Appending a second antimeridian segment matches columns by name, so an
    unaliased long name would silently arrive as NULL in that segment.
    """

    casts = {name: cast for name, cast, _description in _FIELDS}
    columns = ", ".join(
        (
            f'CAST("{name}" AS {casts[name]}) AS "{LAUNDERED_FIELDS.get(name, name)}"'
            if casts[name]
            else f'"{name}" AS "{LAUNDERED_FIELDS.get(name, name)}"'
        )
        for name in fields
    )
    return f'SELECT {columns} FROM "{layer}"'


def prefilter_bounds(
    segment: BBox, *, source_crs: str = SOURCE_CRS
) -> tuple[float, float, float, float]:
    """Return the requested extent as a bounding box in the source projection.

    GLiM is stored in an equal-area projection whose meridians and parallels are
    curved, so the corners alone understate the extent; the boundary is densified
    before it is enveloped, and the result is only ever a superset of the request.
    """

    from pyproj import Transformer

    transformer = Transformer.from_crs(TARGET_CRS, source_crs, always_xy=True)
    west, south, east, north = segment.as_tuple()
    minimum_x, minimum_y, maximum_x, maximum_y = transformer.transform_bounds(
        west, south, east, north, densify_pts=201
    )
    corners = (minimum_x, minimum_y, maximum_x, maximum_y)
    if not all(math.isfinite(value) for value in corners):
        raise ConfigurationError(
            f"The requested extent cannot be projected to the GLiM source CRS {source_crs}"
        )
    # Absorb the residual densification error without widening the selection.
    margin = 1e-6 * max(maximum_x - minimum_x, maximum_y - minimum_y)
    return (minimum_x - margin, minimum_y - margin, maximum_x + margin, maximum_y + margin)


def extent_area(bbox: BBox) -> float:
    """Return the extent's area in square degrees, antimeridian segments included."""

    return sum(
        (segment.east - segment.west) * (segment.north - segment.south)
        for segment in bbox.segments()
    )


def _geodatabase_prefix(archive: zipfile.ZipFile) -> str:
    prefixes = {
        name.replace("\\", "/").split("/", 1)[0]
        for name in archive.namelist()
        if "/" in name.replace("\\", "/")
    }
    geodatabases = sorted(name for name in prefixes if name.lower().endswith(".gdb"))
    if len(geodatabases) != 1:
        raise DownloadError(
            "The GLiM archive does not contain exactly one file geodatabase directory"
        )
    return geodatabases[0] + "/"


def extract_geodatabase(
    archive_path: Path,
    destination: Path,
    *,
    progress: ProgressCallback,
    cancel: CancellationToken,
) -> Path:
    """Unpack the file geodatabase of a GLiM archive into ``destination``."""

    destination = Path(destination)
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.part"
    try:
        with zipfile.ZipFile(archive_path) as archive:
            prefix = _geodatabase_prefix(archive)
            members = tuple(
                info
                for info in archive.infolist()
                if not info.is_dir() and info.filename.replace("\\", "/").startswith(prefix)
            )
            if not members:
                raise DownloadError("The GLiM archive contains no geodatabase members")
            staging.mkdir(parents=True)
            root = staging.resolve()
            total = sum(info.file_size for info in members)
            copied_total = 0
            for index, info in enumerate(members, start=1):
                cancel.check()
                relative = info.filename.replace("\\", "/")[len(prefix) :]
                target = (staging / relative).resolve()
                if not relative or not target.is_relative_to(root):
                    raise DownloadError(f"Refusing unsafe GLiM archive member: {info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as sink:
                    copied = copy_stream(source, sink, cancel=cancel)
                if copied != info.file_size:
                    raise DownloadError(f"Incomplete GLiM archive member: {info.filename}")
                copied_total += copied
                progress(
                    ProgressEvent(
                        "transform",
                        f"Extracting {destination.name}",
                        current=index,
                        total=len(members),
                        bytes_done=copied_total,
                        bytes_total=total,
                    )
                )
        # ``os.replace`` refuses a non-empty destination directory, so a stale
        # geodatabase is removed only once a complete replacement exists.
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DownloadError(f"Invalid GLiM archive {Path(archive_path).name}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def _require_space(path: Path, needed: int) -> None:
    free = available_bytes(path)
    if free < needed:
        raise DownloadError(
            f"GLiM needs about {needed / 2**30:.1f} GiB below {path}, "
            f"but only {free / 2**30:.1f} GiB is free"
        )


class GlimProvider(Provider):
    spec = DatasetSpec(
        id="glim",
        title="GLiM v1.1 global lithological map",
        summary=(
            "Global surface lithology polygons compiled by the University of Hamburg, "
            "cropped to the requested extent and written as an ESRI Shapefile with the "
            "published attribute table."
        ),
        group="Geology",
        variables=(
            VariableSpec(
                "lithology",
                "Surface lithology polygons",
                units="1",
                description=(
                    "One polygon per mapped lithological unit. The primary class code "
                    "'xx' takes the values: "
                    + ", ".join(f"{code} ({label})" for code, label in CLASSES)
                    + "."
                ),
            ),
        ),
        temporal=False,
        formats=VECTOR_FORMATS,
        options=(
            OptionSpec(
                id="fields",
                title="Attribute fields",
                kind="choice",
                default=None,
                choices=FIELD_NAMES,
                multiple=True,
                description=(
                    "Columns copied into the shapefile attribute table. Every published "
                    "column is kept by default."
                ),
            ),
            OptionSpec(
                id="clip",
                title="Clip polygons to the extent",
                kind="bool",
                default=True,
                description=(
                    "Cut polygons exactly at the requested extent. Set to false to keep "
                    "whole polygons instead; selection then uses the extent enveloped in "
                    "the source projection, so a polygon just outside it may be included."
                ),
            ),
            OptionSpec(
                id="local_source",
                title="Local GLiM source",
                kind="path",
                default=None,
                description=(
                    "An already downloaded 'LiMW_GIS 2015.gdb' directory or its .zip "
                    "archive, used instead of transferring about 1.1 GiB again. Any "
                    "other GDAL-readable source carrying the same layer also works."
                ),
            ),
            OptionSpec(
                id="cache_source",
                title="Keep the source geodatabase",
                kind="bool",
                default=True,
                description=(
                    "Store the extracted geodatabase (about 1.2 GiB) below the cache "
                    "directory so later extents reuse it. Set to false to discard it "
                    "with the request scratch data."
                ),
            ),
        ),
        source_url=SOURCE_PAGE,
        citation=(
            "Hartmann, J. and Moosdorf, N. (2012), The new global lithological map "
            "database GLiM, Geochemistry, Geophysics, Geosystems 13, Q12004."
        ),
        license="CC BY 3.0",
    )

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        fields = selected_fields(request)
        clip = bool(self.option(request, "clip"))
        if request.bbox.crosses_antimeridian and not clip:
            raise ConfigurationError(
                "An antimeridian-crossing GLiM extent requires clip=true; unclipped "
                "polygons reaching across 180 degrees would be written twice"
            )
        local = self.option(request, "local_source")
        if local is not None and not Path(local).expanduser().exists():
            raise ConfigurationError(f"local_source does not exist: {local}")
        source_sql(fields)
        for segment in request.bbox.segments():
            prefilter_bounds(segment)

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        fields = selected_fields(request)
        clip = bool(self.option(request, "clip"))
        local = self.option(request, "local_source")
        cached = self._cached_geodatabase(request)
        staged = local is not None or cached is not None
        warnings: list[str] = []
        if any(field in fields for field in SOURCE_GEOMETRY_FIELDS):
            warnings.append(
                "Shape_Length and Shape_Area describe the original uncropped polygon in "
                "the native Eckert IV projection, not the written geometry"
            )
        if any(field in fields for field in LAUNDERED_FIELDS):
            warnings.append(
                "Shapefile column names are limited to ten characters: "
                + ", ".join(f"{name} becomes {short}" for name, short in LAUNDERED_FIELDS.items())
            )
        if not staged:
            warnings.append(
                f"The complete {ARCHIVE_BYTES / 2**30:.1f} GiB source archive is transferred "
                "once before the extent is cropped"
            )
        if extent_area(request.bbox) >= _LARGE_EXTENT_DEGREES_SQUARED:
            warnings.append(
                "An ESRI Shapefile component file cannot exceed two gigabytes; this extent "
                "may be too large for a single shapefile"
            )
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=1,
            slice_count=1,
            estimated_bytes=0 if staged else ARCHIVE_BYTES,
            details={
                "source_layer": SOURCE_LAYER,
                "source_crs": SOURCE_CRS,
                "target_crs": TARGET_CRS,
                "geometry_type": "MultiPolygon",
                "fields": fields,
                "shapefile_field_names": tuple(
                    LAUNDERED_FIELDS.get(name, name) for name in fields
                ),
                "clip": clip,
                "release": "LiMW_GIS 2015 (GLiM v1.1)",
                "staged_source": str(local or cached) if staged else None,
                "primary_classes": {code: label for code, label in CLASSES},
            },
            warnings=tuple(warnings),
        )

    def manifest_metadata(self, request: DownloadRequest, plan: DownloadPlan) -> dict[str, Any]:
        metadata = dict(super().manifest_metadata(request, plan))
        metadata.update(
            {
                "source_layer": SOURCE_LAYER,
                "source_crs": SOURCE_CRS,
                "target_crs": TARGET_CRS,
                "release": "LiMW_GIS 2015 (GLiM v1.1)",
                "primary_classes": {code: label for code, label in CLASSES},
            }
        )
        return metadata

    def _cached_geodatabase(self, request: DownloadRequest) -> Path | None:
        if not bool(self.option(request, "cache_source")):
            return None
        candidate = cache_root(request) / self.spec.id / GEODATABASE_NAME
        return candidate if validate_vector(candidate, layer=SOURCE_LAYER) else None

    def _source_geodatabase(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Path:
        local = self.option(request, "local_source")
        if local is not None:
            return self._local_geodatabase(Path(local).expanduser(), workspace, progress, cancel)

        cached = self._cached_geodatabase(request)
        if cached is not None:
            progress(ProgressEvent("download", f"Reusing staged GLiM geodatabase {cached}"))
            return cached

        keep = bool(self.option(request, "cache_source"))
        root = (cache_root(request) if keep else workspace) / self.spec.id
        root.mkdir(parents=True, exist_ok=True)
        destination = root / GEODATABASE_NAME
        _require_space(root, ARCHIVE_BYTES + GEODATABASE_BYTES)
        archive = workspace / ARCHIVE_NAME
        session = make_session()
        try:
            progress(ProgressEvent("download", f"Downloading GLiM archive {ARCHIVE_NAME}"))
            stream_download(
                archive_url(),
                archive,
                session=session,
                progress=progress,
                cancel=cancel,
            )
            extract_geodatabase(archive, destination, progress=progress, cancel=cancel)
        finally:
            session.close()
            archive.unlink(missing_ok=True)
        if not validate_vector(destination, layer=SOURCE_LAYER, cancel=cancel):
            raise DownloadError(
                f"The downloaded GLiM geodatabase has no {SOURCE_LAYER!r} layer"
            )
        return destination

    def _local_geodatabase(
        self,
        source: Path,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Path:
        if source.is_file() and zipfile.is_zipfile(source):
            destination = workspace / self.spec.id / GEODATABASE_NAME
            destination.parent.mkdir(parents=True, exist_ok=True)
            progress(ProgressEvent("transform", f"Unpacking local GLiM archive {source.name}"))
            extract_geodatabase(source, destination, progress=progress, cancel=cancel)
        else:
            destination = source
        if not validate_vector(destination, layer=SOURCE_LAYER, cancel=cancel):
            raise ConfigurationError(
                f"local_source has no {SOURCE_LAYER!r} layer: {source}. A file geodatabase "
                "directory must keep its '.gdb' suffix for GDAL to recognise it."
            )
        return destination

    def slices(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Iterator[VectorSlice]:
        self.validate(request)
        # Fail before transferring about 1.1 GiB on a machine without OGR.
        require_commands("ogr2ogr", "ogrinfo")
        fields = selected_fields(request)
        clip = bool(self.option(request, "clip"))
        geodatabase = self._source_geodatabase(request, workspace, progress, cancel)

        slice_dir = workspace / self.spec.id / "slices"
        slice_dir.mkdir(parents=True, exist_ok=True)
        local = slice_dir / "lithology.shp"
        segments = request.bbox.segments()
        for index, segment in enumerate(segments):
            cancel.check()
            progress(
                ProgressEvent(
                    "transform",
                    "Cropping GLiM lithology polygons",
                    current=index + 1,
                    total=len(segments),
                )
            )
            vector_to_shapefile(
                geodatabase,
                local,
                sql=source_sql(fields),
                layer_name=local.stem,
                source_bounds=prefilter_bounds(segment),
                clip_bbox=segment if clip else None,
                target_crs=TARGET_CRS,
                geometry_type="MULTIPOLYGON",
                append=index > 0,
                cancel=cancel,
            )
        if not validate_vector(local, minimum_features=0, cancel=cancel):
            raise OutputError("The cropped GLiM layer could not be read back")
        if not validate_vector(local, cancel=cancel):
            raise DownloadError(
                "No GLiM lithology polygons intersect the requested extent"
            )
        try:
            yield VectorSlice(
                variable="lithology",
                source=local,
                attrs={
                    "long_name": "GLiM surface lithology",
                    "source": "GLiM v1.1 (LiMW_GIS 2015)",
                    "source_layer": SOURCE_LAYER,
                    "source_crs": SOURCE_CRS,
                    "clipped": clip,
                },
            )
        finally:
            for path in slice_dir.glob(local.stem + ".*"):
                path.unlink(missing_ok=True)


__all__ = [
    "ARCHIVE_URL",
    "CLASSES",
    "FIELD_NAMES",
    "GlimProvider",
    "SOURCE_LAYER",
    "archive_url",
    "extent_area",
    "extract_geodatabase",
    "prefilter_bounds",
    "selected_fields",
    "source_sql",
]
