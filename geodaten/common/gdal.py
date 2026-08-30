"""Thin, checked wrappers around GDAL's streaming command-line tools."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from geodaten.common.errors import DependencyError, OutputError
from geodaten.common.io import atomic_path
from geodaten.common.models import BBox
from geodaten.common.progress import CancellationToken

#: Files an ESRI Shapefile layer is spread across; only ``.shp``/``.shx``/``.dbf``
#: are mandatory, the rest are written whenever GDAL has something to record.
SHAPEFILE_SIDECARS: tuple[str, ...] = (".shp", ".shx", ".dbf", ".prj", ".cpg")
_REQUIRED_SIDECARS: tuple[str, ...] = (".shp", ".shx", ".dbf")

_LAYER_NAME = re.compile(r"^Layer name:\s*(?P<name>.+?)\s*$")
_FEATURE_COUNT = re.compile(r"^Feature Count:\s*(?P<count>-?\d+)\s*$")


def require_commands(*commands: str) -> None:
    missing = [command for command in commands if shutil.which(command) is None]
    if missing:
        raise DependencyError("Missing GDAL command(s): " + ", ".join(missing))


def run(
    command: Sequence[str | os.PathLike[str]],
    *,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
    cancel: CancellationToken | None = None,
) -> str:
    args = [str(value) for value in command]
    if cancel:
        cancel.check()
    try:
        process = subprocess.Popen(
            args,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            # Keep failures available to CLI/TUI/GUI callers. GDAL writes no
            # unbounded progress stream unless explicitly asked for one.
            stderr=subprocess.PIPE,
            env=dict(env) if env else None,
        )
    except FileNotFoundError as exc:
        raise DependencyError(f"Required program is not installed: {args[0]}") from exc
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.25 if cancel else None)
                break
            except subprocess.TimeoutExpired:
                assert cancel is not None
                cancel.check()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        raise
    if process.returncode:
        detail = (stderr or stdout or "").strip()
        raise OutputError(
            f"GDAL command failed ({process.returncode}): {Path(args[0]).name}"
            + (f"\n{detail}" if detail else "")
        )
    return stdout or ""


def info(
    source: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    cancel: CancellationToken | None = None,
) -> dict[str, Any]:
    require_commands("gdalinfo")
    try:
        return json.loads(
            run(("gdalinfo", "-json", str(source)), env=env, capture=True, cancel=cancel)
        )
    except json.JSONDecodeError as exc:
        raise OutputError(f"GDAL returned invalid metadata for {source}") from exc


def validate_raster(path: Path, *, bands: int | None = None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        metadata = info(path)
        if min(metadata.get("size", (0, 0))) <= 0:
            return False
        if bands is not None and len(metadata.get("bands", ())) != bands:
            return False
        return bool(metadata.get("coordinateSystem") or metadata.get("geoTransform"))
    except (OSError, OutputError, TypeError, ValueError):
        return False


def vector_layers(
    source: str | Path,
    *,
    layer: str | None = None,
    env: Mapping[str, str] | None = None,
    cancel: CancellationToken | None = None,
) -> dict[str, int | None]:
    """Return ``{layer name: feature count}`` for a GDAL-readable vector source.

    ``ogrinfo`` summary text is parsed instead of ``-json`` because the JSON
    output only exists from GDAL 3.7 onwards, while the wrappers here still
    support the older releases handled by :func:`warp_to_cog`.
    """

    require_commands("ogrinfo")
    command = ["ogrinfo", "-ro", "-so", "-nomd", str(source)]
    command.append(layer if layer else "-al")
    output = run(command, env=env, capture=True, cancel=cancel)
    layers: dict[str, int | None] = {}
    current: str | None = None
    for line in output.splitlines():
        name = _LAYER_NAME.match(line)
        if name:
            current = name.group("name")
            layers.setdefault(current, None)
            continue
        count = _FEATURE_COUNT.match(line)
        if count and current is not None:
            value = int(count.group("count"))
            layers[current] = value if value >= 0 else None
    return layers


def shapefile_members(
    path: str | Path, *, extensions: Sequence[str] = SHAPEFILE_SIDECARS
) -> tuple[Path, ...]:
    """Return every sidecar path of one shapefile layer, present or not.

    ``Path.with_suffix`` cannot be used because a layer name may legitimately
    contain a dot, which would silently rewrite the wrong part of the name.
    """

    path = Path(path)
    return tuple(path.parent / (path.stem + extension) for extension in extensions)


def validate_vector(
    source: Path,
    *,
    layer: str | None = None,
    minimum_features: int = 1,
    env: Mapping[str, str] | None = None,
    cancel: CancellationToken | None = None,
) -> bool:
    """Return true only for a readable vector layer holding enough features.

    An empty layer is structurally valid but is never a usable crop, so it is
    rejected by default; a driver that cannot count cheaply reports ``None`` and
    is accepted. A missing GDAL program is raised rather than reported as
    invalid, so that a broken installation is not mistaken for stale output.
    """

    source = Path(source)
    if source.suffix.lower() == ".shp":
        if any(
            not path.is_file() or path.stat().st_size == 0
            for path in shapefile_members(source, extensions=_REQUIRED_SIDECARS)
        ):
            return False
    elif not source.exists():
        return False
    try:
        layers = vector_layers(source, layer=layer, env=env, cancel=cancel)
    except (OSError, OutputError, ValueError):
        return False
    if layer is not None and layer not in layers:
        return False
    counts = [layers[layer]] if layer is not None else list(layers.values())
    if not counts:
        return False
    return all(count is None or count >= minimum_features for count in counts)


def vector_to_shapefile(
    source: str | Path,
    destination: Path,
    *,
    layer: str | None = None,
    sql: str | None = None,
    layer_name: str | None = None,
    source_bounds: Sequence[float] | None = None,
    clip_bbox: BBox | None = None,
    target_crs: str | None = None,
    geometry_type: str | None = None,
    append: bool = False,
    env: Mapping[str, str] | None = None,
    cancel: CancellationToken | None = None,
) -> Path:
    """Convert one feature layer to an ESRI Shapefile, optionally cropping it.

    ``source_bounds`` is an inexpensive prefilter expressed in the *source* CRS so
    that a spatial index can be used, while ``clip_bbox`` performs the exact crop
    in the destination CRS after reprojection. Callers that need an extent
    crossing the antimeridian pass one non-crossing segment per call and set
    ``append`` for every call after the first.
    """

    require_commands("ogr2ogr", "ogrinfo")
    if clip_bbox is not None and clip_bbox.crosses_antimeridian:
        raise OutputError("A single vector clip cannot cross the antimeridian")
    if clip_bbox is not None and target_crs is None:
        # ogr2ogr clips in the destination CRS, so a WGS84 extent applied to a
        # projected source would silently select nothing.
        raise OutputError("Clipping to a WGS84 extent requires an explicit target CRS")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command: list[str] = ["ogr2ogr", "-f", "ESRI Shapefile"]
    command.extend(("-append",) if append else ("-overwrite",))
    if target_crs is not None:
        command.extend(("-t_srs", target_crs))
    if source_bounds is not None:
        if len(source_bounds) != 4:
            raise OutputError("A source prefilter needs MINX MINY MAXX MAXY")
        if source_bounds[0] >= source_bounds[2] or source_bounds[1] >= source_bounds[3]:
            # ogr2ogr accepts an inverted rectangle and quietly selects its
            # complement instead of reporting the mistake.
            raise OutputError("A source prefilter needs MINX < MAXX and MINY < MAXY")
        command.append("-spat")
        command.extend(f"{float(value):.10g}" for value in source_bounds)
    if clip_bbox is not None:
        west, south, east, north = clip_bbox.as_tuple()
        command.append("-clipdst")
        command.extend(f"{value:.10g}" for value in (west, south, east, north))
    if geometry_type is not None:
        command.extend(("-nlt", geometry_type))
    if layer_name is not None:
        command.extend(("-nln", layer_name))
    if not append:
        command.extend(("-lco", "ENCODING=UTF-8"))
    if sql is not None:
        command.extend(("-sql", sql))
    command.extend((str(destination), str(source)))
    if sql is None and layer is not None:
        command.append(layer)
    run(command, env=env, cancel=cancel)
    # A crop may legitimately select nothing from one segment, so only the layer's
    # structure is checked here; callers decide whether an empty result is usable.
    if not validate_vector(destination, minimum_features=0, env=env, cancel=cancel):
        raise OutputError(f"Generated shapefile failed validation: {destination}")
    return destination


def build_vrt(
    sources: Sequence[str | Path],
    destination: Path,
    *,
    band: int | None = None,
    separate: bool = False,
    env: Mapping[str, str] | None = None,
    cancel: CancellationToken | None = None,
) -> Path:
    """Assemble sources into one VRT.

    ``separate`` stacks each source as its own band instead of mosaicking them
    side by side; every source must already share one grid.
    """

    require_commands("gdalbuildvrt")
    if not sources:
        raise OutputError("Cannot build a mosaic without source rasters")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_list = destination.with_suffix(".sources.txt")
    source_list.write_text("".join(f"{source}\n" for source in sources), encoding="utf-8")
    command: list[str] = ["gdalbuildvrt", "-overwrite", "-resolution", "highest"]
    if separate:
        command.append("-separate")
    if band is not None:
        command.extend(("-b", str(band)))
    command.extend(("-input_file_list", str(source_list), str(destination)))
    try:
        run(command, env=env, cancel=cancel)
    finally:
        source_list.unlink(missing_ok=True)
    return destination


def translate_to_cog(
    source: str | Path,
    destination: Path,
    *,
    bbox: BBox | None = None,
    band: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    nodata: Any = None,
    scale: Any = None,
    offset: Any = None,
    env: Mapping[str, str] | None = None,
    cancel: CancellationToken | None = None,
) -> Path:
    if bbox is not None:
        return warp_to_cog(
            source,
            destination,
            bbox=bbox,
            band=band,
            metadata=metadata,
            nodata=nodata,
            scale=scale,
            offset=offset,
            env=env,
            cancel=cancel,
        )
    require_commands("gdal_translate", "gdalinfo")
    with atomic_path(destination, suffix=".part.tif") as temporary:
        command: list[str] = ["gdal_translate", "-of", "COG"]
        if band is not None:
            command.extend(("-b", str(band)))
        if nodata is not None:
            command.extend(("-a_nodata", str(nodata)))
        if scale is not None:
            command.extend(("-a_scale", str(scale)))
        if offset is not None:
            command.extend(("-a_offset", str(offset)))
        command.extend(
            (
                "-co",
                "COMPRESS=DEFLATE",
                "-co",
                "BIGTIFF=IF_SAFER",
                "-co",
                "OVERVIEW_RESAMPLING=NEAREST",
            )
        )
        for key, value in (metadata or {}).items():
            if value is not None:
                command.extend(("-mo", f"{str(key).upper()}={str(value).replace(chr(10), ' ')}"))
        command.extend((str(source), str(temporary)))
        run(command, env=env, cancel=cancel)
        if not validate_raster(temporary, bands=1 if band is not None else None):
            raise OutputError(f"Generated raster failed validation: {destination}")
    return destination


def warp_to_cog(
    source: str | Path | Sequence[str | Path],
    destination: Path,
    *,
    bbox: BBox,
    band: int | None = None,
    shape: tuple[int, int] | None = None,
    metadata: Mapping[str, Any] | None = None,
    nodata: Any = None,
    scale: Any = None,
    offset: Any = None,
    env: Mapping[str, str] | None = None,
    cancel: CancellationToken | None = None,
) -> Path:
    """Crop/reproject one or more sources to the exact WGS84 extent via a lazy VRT.

    This avoids pre-GDAL-3.11 ``gdal_translate -projwin_srs`` corner-only
    transforms for nonlinear source projections. Several sources are mosaicked
    by ``gdalwarp`` itself, which unlike ``gdalbuildvrt`` reprojects each one, so
    tiles spanning different projections merge instead of being dropped.

    ``shape`` forces the output ``(width, height)``, which pins separate calls to
    one identical grid so their results can later be stacked as bands.
    """

    require_commands("gdalwarp", "gdal_translate", "gdalinfo")
    if bbox.crosses_antimeridian:
        raise OutputError("A single GDAL crop cannot cross the antimeridian")
    sources: tuple[str | Path, ...] = (
        (source,) if isinstance(source, (str, Path)) else tuple(source)
    )
    if not sources:
        raise OutputError("Cannot warp without a source raster")
    with tempfile.TemporaryDirectory(prefix="geodaten_warp_") as temporary:
        temporary_root = Path(temporary)
        warp_sources: list[str | Path] = list(sources)
        if band is not None:
            # gdalwarp only gained -srcband/-b in GDAL 3.7. Select the band
            # through a lazy VRT first so older supported installations keep
            # the same bounded, streaming behavior.
            warp_sources = []
            for index, item in enumerate(sources):
                selected = temporary_root / f"selected-band-{index}.vrt"
                run(
                    (
                        "gdal_translate",
                        "-q",
                        "-of",
                        "VRT",
                        "-b",
                        str(band),
                        str(item),
                        str(selected),
                    ),
                    env=env,
                    cancel=cancel,
                )
                warp_sources.append(selected)
        # gdalwarp can only build a warped VRT around a single source, so a
        # mosaic must land in a real raster first or every source but the first
        # is silently dropped.
        mosaicking = len(warp_sources) > 1
        vrt = temporary_root / ("warped.tif" if mosaicking else "warped.vrt")
        command: list[str] = [
            "gdalwarp",
            "-overwrite",
            "-of",
            "GTiff" if mosaicking else "VRT",
            "-t_srs",
            "EPSG:4326",
            "-te",
            f"{bbox.west:g}",
            f"{bbox.south:g}",
            f"{bbox.east:g}",
            f"{bbox.north:g}",
            "-te_srs",
            "EPSG:4326",
            "-r",
            "near",
        ]
        if nodata is not None:
            command.extend(("-dstnodata", str(nodata)))
        if shape is not None:
            command.extend(("-ts", str(shape[0]), str(shape[1])))
        if mosaicking:
            command.extend(("-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER"))
        command.extend(str(item) for item in warp_sources)
        command.append(str(vrt))
        run(command, env=env, cancel=cancel)
        return translate_to_cog(
            vrt,
            destination,
            metadata=metadata,
            nodata=nodata,
            scale=scale,
            offset=offset,
            env=env,
            cancel=cancel,
        )


def mosaic_to_cog(
    sources: Sequence[str | Path],
    destination: Path,
    *,
    bbox: BBox,
    band: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    nodata: Any = None,
    env: Mapping[str, str] | None = None,
    cancel: CancellationToken | None = None,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="geodaten_vrt_") as temporary:
        vrt = Path(temporary) / "mosaic.vrt"
        build_vrt(sources, vrt, band=band, env=env, cancel=cancel)
        return warp_to_cog(
            vrt,
            destination,
            bbox=bbox,
            metadata=metadata,
            nodata=nodata,
            env=env,
            cancel=cancel,
        )
