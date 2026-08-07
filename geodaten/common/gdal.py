"""Thin, checked wrappers around GDAL's streaming command-line tools."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from geodaten.common.errors import DependencyError, OutputError
from geodaten.common.io import atomic_path
from geodaten.common.models import BBox
from geodaten.common.progress import CancellationToken


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


def build_vrt(
    sources: Sequence[str | Path],
    destination: Path,
    *,
    band: int | None = None,
    env: Mapping[str, str] | None = None,
    cancel: CancellationToken | None = None,
) -> Path:
    require_commands("gdalbuildvrt")
    if not sources:
        raise OutputError("Cannot build a mosaic without source rasters")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_list = destination.with_suffix(".sources.txt")
    source_list.write_text("".join(f"{source}\n" for source in sources), encoding="utf-8")
    command: list[str] = ["gdalbuildvrt", "-overwrite", "-resolution", "highest"]
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
    source: str | Path,
    destination: Path,
    *,
    bbox: BBox,
    band: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    nodata: Any = None,
    scale: Any = None,
    offset: Any = None,
    env: Mapping[str, str] | None = None,
    cancel: CancellationToken | None = None,
) -> Path:
    """Crop/reproject a source to the exact WGS84 extent through a lazy VRT.

    This avoids pre-GDAL-3.11 ``gdal_translate -projwin_srs`` corner-only
    transforms for nonlinear source projections.
    """

    require_commands("gdalwarp", "gdal_translate", "gdalinfo")
    if bbox.crosses_antimeridian:
        raise OutputError("A single GDAL crop cannot cross the antimeridian")
    with tempfile.TemporaryDirectory(prefix="geodaten_warp_") as temporary:
        temporary_root = Path(temporary)
        warp_source: str | Path = source
        if band is not None:
            # gdalwarp only gained -srcband/-b in GDAL 3.7. Select the band
            # through a lazy VRT first so older supported installations keep
            # the same bounded, streaming behavior.
            selected = temporary_root / "selected-band.vrt"
            run(
                (
                    "gdal_translate",
                    "-q",
                    "-of",
                    "VRT",
                    "-b",
                    str(band),
                    str(source),
                    str(selected),
                ),
                env=env,
                cancel=cancel,
            )
            warp_source = selected
        vrt = temporary_root / "warped.vrt"
        command: list[str] = [
            "gdalwarp",
            "-overwrite",
            "-of",
            "VRT",
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
        command.extend((str(warp_source), str(vrt)))
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
