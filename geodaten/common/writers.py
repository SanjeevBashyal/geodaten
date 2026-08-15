"""Memory-bounded writers for per-slice COGs, one request-level NetCDF, and shapefiles."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
from rasterio.windows import Window

from geodaten.common.errors import OutputError
from geodaten.common.gdal import (
    shapefile_members,
    translate_to_cog,
    validate_raster,
    validate_vector,
    vector_layers,
    vector_to_shapefile,
)
from geodaten.common.io import safe_component
from geodaten.common.models import BBox, DownloadRequest, RasterSlice, VectorSlice
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent, null_progress


def output_root(request: DownloadRequest) -> Path:
    return request.output_dir / request.dataset / request.fingerprint


def _uses_latest(request: DownloadRequest) -> bool:
    return request.time is not None and any(
        str(value).strip().lower() == "latest"
        for value in (request.time.start, request.time.end)
    )


def _identifier(value: Any, fallback: str = "variable") -> str:
    name = re.sub(r"\W+", "_", str(value).strip(), flags=re.UNICODE).strip("_").lower()
    if not name:
        name = fallback
    if name[0].isdigit():
        name = "v_" + name
    return name


def _attribute(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (tuple, list)):
        return np.asarray(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _bounds_wkt(bbox: BBox) -> str:
    rings = []
    for segment in bbox.segments():
        west, south, east, north = segment.as_tuple()
        points = (
            (west, south),
            (east, south),
            (east, north),
            (west, north),
            (west, south),
        )
        rings.append("(" + ", ".join(f"{x:g} {y:g}" for x, y in points) + ")")
    if len(rings) == 1:
        return "POLYGON(" + rings[0] + ")"
    return "MULTIPOLYGON(" + ", ".join(f"({ring})" for ring in rings) + ")"


class TiffWriter:
    def __init__(
        self,
        request: DownloadRequest,
        *,
        progress: ProgressCallback = null_progress,
        cancel: CancellationToken | None = None,
    ) -> None:
        self.request = request
        self.root = output_root(request)
        self.progress = progress
        self.cancel = cancel
        self.outputs: list[Path] = []
        self.skipped = 0

    def resume_complete_request(self, expected_count: int | None) -> bool:
        """Reuse only a complete, self-identifying request-level TIFF set."""

        if (
            self.request.overwrite
            or _uses_latest(self.request)
            or expected_count is None
            or expected_count < 1
            or not self.root.is_dir()
        ):
            return False
        candidates = sorted(
            path
            for path in self.root.iterdir()
            if path.suffix.lower() in {".tif", ".tiff"}
        )
        if len(candidates) != expected_count:
            return False
        for path in candidates:
            if self.cancel:
                self.cancel.check()
            if path.is_symlink() or not path.is_file() or not validate_raster(path, bands=1):
                return False
            try:
                with rasterio.open(path) as source:
                    tags = {key.upper(): str(value) for key, value in source.tags().items()}
            except (OSError, rasterio.errors.RasterioError):
                return False
            if (
                tags.get("DATASET") != self.request.dataset
                or tags.get("REQUEST_FINGERPRINT") != self.request.fingerprint
                or tags.get("OUTPUT_NAME") != path.name
                or tags.get("VARIABLE") not in self.request.variables
            ):
                return False

        self.outputs.extend(candidates)
        self.skipped += len(candidates)
        self.progress(
            ProgressEvent(
                "output",
                f"Reusing complete TIFF request ({len(candidates)} files)",
                current=len(candidates),
                total=len(candidates),
            )
        )
        return True

    def _path(self, item: RasterSlice) -> Path:
        pieces = [safe_component(self.request.dataset), safe_component(item.variable)]
        if item.time is not None:
            label = item.time.isoformat() if hasattr(item.time, "isoformat") else str(item.time)
            pieces.append(safe_component(label))
        pieces.extend(
            f"{safe_component(key)}-{safe_component(value)}"
            for key, value in sorted(item.dimensions.items())
        )
        return self.root / ("_".join(pieces) + ".tif")

    def write(self, item: RasterSlice) -> Path:
        if self.cancel:
            self.cancel.check()
        destination = self._path(item)
        if (
            destination.exists()
            and not self.request.overwrite
            and not _uses_latest(self.request)
            and validate_raster(destination, bands=1)
        ):
            self.outputs.append(destination)
            self.skipped += 1
            self.progress(ProgressEvent("output", f"Already complete: {destination.name}"))
            return destination
        self.progress(ProgressEvent("output", f"Writing {destination.name}"))
        metadata = {
            "dataset": self.request.dataset,
            "request_fingerprint": self.request.fingerprint,
            "output_name": destination.name,
            "variable": item.variable,
            "time": item.time.isoformat() if hasattr(item.time, "isoformat") else item.time,
            **item.dimensions,
            **{key: value for key, value in item.attrs.items() if np.isscalar(value)},
        }
        translate_to_cog(
            item.source,
            destination,
            band=item.band,
            metadata=metadata,
            nodata=item.attrs.get("_FillValue"),
            scale=item.attrs.get("scale_factor"),
            offset=item.attrs.get("add_offset"),
            cancel=self.cancel,
        )
        self.outputs.append(destination)
        return destination

    def close(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(self.outputs))

    def abort(self) -> None:
        return None


class NetCDFWriter:
    """Append raster slices block-by-block to one CF-1.13 NetCDF4 file."""

    def __init__(
        self,
        request: DownloadRequest,
        *,
        progress: ProgressCallback = null_progress,
        cancel: CancellationToken | None = None,
    ) -> None:
        self.request = request
        self.root = output_root(request)
        self.target = self.root / f"{safe_component(request.dataset)}_{request.fingerprint}.nc"
        self.partial = self.target.with_name(
            f".{self.target.name}.{uuid.uuid4().hex}.part"
        )
        self.progress = progress
        self.cancel = cancel
        self.dataset: Any = None
        self.outputs: list[Path] = []
        self.skipped = 0
        self._grid: tuple[Any, ...] | None = None
        self._variables: dict[str, tuple[str, ...]] = {}
        self._coordinate_indices: dict[str, dict[str, int]] = {}
        self._time_calendar: str | None = None
        self.complete = (
            self._is_complete(self.target)
            and not request.overwrite
            and not _uses_latest(request)
        )
        if self.complete:
            self.outputs.append(self.target)
            self.skipped = 1

    @staticmethod
    def _is_complete(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            from netCDF4 import Dataset

            with Dataset(path) as dataset:
                spatial = any(
                    name not in dataset.dimensions
                    and name != "crs"
                    and {"y", "x"}.issubset(variable.dimensions)
                    for name, variable in dataset.variables.items()
                )
                return str(getattr(dataset, "Conventions", "")).startswith("CF-") and spatial
        except Exception:
            return False

    def _open(self, source: rasterio.io.DatasetReader) -> None:
        from netCDF4 import Dataset

        if source.crs is None:
            raise OutputError("Raster slices require a CRS for CF NetCDF output")
        self.root.mkdir(parents=True, exist_ok=True)
        self.partial.unlink(missing_ok=True)
        dataset = Dataset(self.partial, "w", format="NETCDF4")
        dataset.Conventions = "CF-1.13"
        dataset.title = f"{self.request.dataset} spatial and temporal subset"
        dataset.source = self.request.dataset
        dataset.history = f"{datetime.now(UTC).isoformat(timespec='seconds')} created by geodaten"
        dataset.geospatial_bounds = _bounds_wkt(self.request.bbox)
        dataset.geospatial_bounds_crs = "EPSG:4326"
        dataset.geospatial_lat_min = self.request.bbox.south
        dataset.geospatial_lat_max = self.request.bbox.north
        dataset.request_fingerprint = self.request.fingerprint
        dataset.createDimension("y", source.height)
        dataset.createDimension("x", source.width)

        transform = source.transform
        if not math.isclose(transform.b, 0.0, abs_tol=1e-12) or not math.isclose(
            transform.d, 0.0, abs_tol=1e-12
        ):
            dataset.close()
            raise OutputError("Rotated rasters cannot be represented by the current NetCDF writer")
        x = transform.c + (np.arange(source.width, dtype="float64") + 0.5) * transform.a
        y = transform.f + (np.arange(source.height, dtype="float64") + 0.5) * transform.e
        x_var = dataset.createVariable("x", "f8", ("x",))
        y_var = dataset.createVariable("y", "f8", ("y",))
        x_var[:] = x
        y_var[:] = y
        if source.crs and source.crs.is_geographic:
            x_var.standard_name = "longitude"
            x_var.long_name = "longitude"
            x_var.units = "degrees_east"
            x_var.axis = "X"
            y_var.standard_name = "latitude"
            y_var.long_name = "latitude"
            y_var.units = "degrees_north"
            y_var.axis = "Y"
        else:
            x_var.standard_name = "projection_x_coordinate"
            x_var.long_name = "x coordinate of projection"
            x_var.units = "m"
            x_var.axis = "X"
            y_var.standard_name = "projection_y_coordinate"
            y_var.long_name = "y coordinate of projection"
            y_var.units = "m"
            y_var.axis = "Y"

        crs_var = dataset.createVariable("crs", "i4")
        crs_var.spatial_ref = source.crs.to_wkt()
        crs_var.crs_wkt = source.crs.to_wkt()
        try:
            from pyproj import CRS

            cf_crs = CRS.from_user_input(source.crs).to_cf()
            if "grid_mapping_name" not in cf_crs:
                raise ValueError("CRS has no CF grid mapping")
            for key, value in cf_crs.items():
                converted = _attribute(value)
                if converted is not None:
                    setattr(crs_var, key, converted)
        except Exception as exc:
            dataset.close()
            raise OutputError(f"Could not encode raster CRS as a CF grid mapping: {exc}") from exc
        self.dataset = dataset
        self._grid = (
            source.width,
            source.height,
            tuple(source.transform),
            source.crs.to_wkt() if source.crs else None,
        )

    def _assert_grid(self, source: rasterio.io.DatasetReader) -> None:
        signature = (
            source.width,
            source.height,
            tuple(source.transform),
            source.crs.to_wkt() if source.crs else None,
        )
        if signature != self._grid:
            raise OutputError("All slices in one NetCDF request must use the same spatial grid")

    def _time_index(self, value: Any, calendar: str) -> int:
        from netCDF4 import date2num

        name = "time"
        if name not in self.dataset.dimensions:
            self.dataset.createDimension(name, None)
            variable = self.dataset.createVariable(name, "f8", (name,))
            variable.standard_name = "time"
            variable.long_name = "time"
            variable.axis = "T"
            variable.units = "seconds since 1970-01-01 00:00:00 UTC"
            variable.calendar = calendar
            self._time_calendar = calendar
            self._coordinate_indices[name] = {}
        if calendar != self._time_calendar:
            raise OutputError("A single NetCDF output cannot mix calendars")

        if isinstance(value, np.datetime64):
            value = value.astype("datetime64[us]").astype(datetime)
        if isinstance(value, str):
            text = value.strip().replace("Z", "+00:00")
            try:
                value = datetime.fromisoformat(text)
            except ValueError:
                try:
                    value = date.fromisoformat(text)
                except ValueError as exc:
                    raise OutputError(f"Time value is not ISO-8601: {value}") from exc
        if isinstance(value, date) and not isinstance(value, datetime):
            value = datetime(value.year, value.month, value.day)
        key = repr(value)
        indices = self._coordinate_indices[name]
        if key in indices:
            return indices[key]
        index = len(indices)
        variable = self.dataset.variables[name]
        variable[index] = date2num(value, variable.units, calendar=calendar)
        indices[key] = index
        return index

    def _dimension_index(self, name: str, value: Any) -> int:
        name = _identifier(name, "dimension")
        key = json.dumps(_attribute(value), sort_keys=True, default=str)
        if name not in self.dataset.dimensions:
            self.dataset.createDimension(name, None)
            dtype: Any = "f8" if isinstance(value, (int, float, np.number)) else str
            self.dataset.createVariable(name, dtype, (name,))
            self._coordinate_indices[name] = {}
        indices = self._coordinate_indices[name]
        if key in indices:
            return indices[key]
        index = len(indices)
        self.dataset.variables[name][index] = value
        indices[key] = index
        return index

    def _create_data_variable(
        self,
        item: RasterSlice,
        source: rasterio.io.DatasetReader,
        dimensions: tuple[str, ...],
    ) -> Any:
        name = _identifier(item.variable)
        if name in self.dataset.variables:
            if self._variables.get(name) != dimensions:
                raise OutputError(f"Dimensions changed for variable {name}")
            return self.dataset.variables[name]

        dtype = np.dtype(source.dtypes[item.band - 1])
        nodata = item.attrs.get("_FillValue", source.nodatavals[item.band - 1])
        chunks = tuple(1 for _ in dimensions[:-2]) + (
            min(512, source.height),
            min(512, source.width),
        )
        kwargs: dict[str, Any] = {
            "zlib": True,
            "complevel": 4,
            "shuffle": dtype.kind != "O",
            "chunksizes": chunks,
        }
        if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
            kwargs["fill_value"] = nodata
        variable = self.dataset.createVariable(name, dtype, dimensions, **kwargs)
        variable.set_auto_maskandscale(False)
        variable.grid_mapping = "crs"
        variable.coordinates = "y x"
        variable.long_name = str(item.attrs.get("long_name", item.variable))

        attrs = dict(item.attrs)
        if "scale_factor" not in attrs and source.scales[item.band - 1] != 1.0:
            attrs["scale_factor"] = source.scales[item.band - 1]
        if "add_offset" not in attrs and source.offsets[item.band - 1] != 0.0:
            attrs["add_offset"] = source.offsets[item.band - 1]
        structural = {
            "_FillValue",
            "actual_max",
            "actual_min",
            "actual_range",
            "ancillary_variables",
            "bounds",
            "calendar",
            "cell_measures",
            "coordinates",
            "dimension_attrs",
            "grid_mapping",
            "long_name",
        }
        for key, value in attrs.items():
            if key in structural:
                continue
            converted = _attribute(value)
            if converted is not None:
                setattr(variable, key, converted)
        self._variables[name] = dimensions
        return variable

    def write(self, item: RasterSlice) -> Path:
        if self.complete:
            return self.target
        if self.cancel:
            self.cancel.check()
        self.progress(ProgressEvent("output", f"Appending {item.variable} to NetCDF"))
        try:
            source = rasterio.open(item.source)
        except Exception as exc:
            raise OutputError(f"Could not open raster slice {item.source}: {exc}") from exc

        with source:
            if item.band < 1 or item.band > source.count:
                raise OutputError(f"Band {item.band} does not exist in {item.source}")
            if self.dataset is None:
                self._open(source)
            else:
                self._assert_grid(source)

            prefix_dimensions: list[str] = []
            prefix_indices: list[int] = []
            if item.time is not None:
                calendar = str(item.attrs.get("calendar", "proleptic_gregorian"))
                prefix_dimensions.append("time")
                prefix_indices.append(self._time_index(item.time, calendar))
            for name, value in sorted(item.dimensions.items()):
                normalized = _identifier(name, "dimension")
                prefix_dimensions.append(normalized)
                prefix_indices.append(self._dimension_index(normalized, value))

            dimensions = tuple(prefix_dimensions) + ("y", "x")
            variable = self._create_data_variable(item, source, dimensions)
            block = 512
            total_blocks = math.ceil(source.height / block) * math.ceil(source.width / block)
            current = 0
            for row in range(0, source.height, block):
                for column in range(0, source.width, block):
                    if self.cancel:
                        self.cancel.check()
                    height = min(block, source.height - row)
                    width = min(block, source.width - column)
                    window = Window(column, row, width, height)
                    data = source.read(item.band, window=window, masked=False)
                    index = tuple(prefix_indices) + (
                        slice(row, row + height),
                        slice(column, column + width),
                    )
                    variable[index] = data
                    current += 1
                    self.progress(
                        ProgressEvent(
                            "output",
                            f"Appending {item.variable}",
                            current=current,
                            total=total_blocks,
                        )
                    )
        return self.target

    def close(self) -> tuple[Path, ...]:
        if self.complete:
            return tuple(self.outputs)
        if self.dataset is None:
            raise OutputError("Provider produced no raster slices")
        self.dataset.date_created = datetime.now(UTC).isoformat(timespec="seconds")
        self.dataset.close()
        self.dataset = None
        self.target.parent.mkdir(parents=True, exist_ok=True)
        if not self._is_complete(self.partial):
            self.partial.unlink(missing_ok=True)
            raise OutputError("Generated NetCDF failed validation")
        os.replace(self.partial, self.target)
        self.outputs.append(self.target)
        return tuple(self.outputs)

    def abort(self) -> None:
        if self.dataset is not None:
            try:
                self.dataset.close()
            finally:
                self.dataset = None
        self.partial.unlink(missing_ok=True)


class ShapefileWriter:
    """Install one cropped feature layer per variable as an ESRI Shapefile set."""

    def __init__(
        self,
        request: DownloadRequest,
        *,
        progress: ProgressCallback = null_progress,
        cancel: CancellationToken | None = None,
    ) -> None:
        self.request = request
        self.root = output_root(request)
        self.progress = progress
        self.cancel = cancel
        self.outputs: list[Path] = []
        self.skipped = 0
        self._staging: Path | None = None
        self._installing: Path | None = None
        self._written: set[Path] = set()

    def _layer_path(self, variable: str) -> Path:
        pieces = [safe_component(self.request.dataset), safe_component(variable)]
        return self.root / ("_".join(pieces) + ".shp")

    @staticmethod
    def _members(target: Path) -> tuple[Path, ...]:
        return tuple(path for path in shapefile_members(target) if path.is_file())

    def resume_complete_request(self, expected_count: int | None) -> bool:
        """Reuse only a complete, self-identifying shapefile set for this request.

        One vector slice becomes several files, so completeness is decided by the
        exact set of layers the request asks for rather than by a file count.
        """

        if (
            self.request.overwrite
            or _uses_latest(self.request)
            or expected_count is None
            or expected_count != len(self.request.variables)
            or not self.root.is_dir()
        ):
            return False
        expected = {
            self._layer_path(variable): variable for variable in self.request.variables
        }
        if set(self.root.glob("*.shp")) != set(expected):
            return False
        reused: list[Path] = []
        for path in expected:
            if self.cancel:
                self.cancel.check()
            if path.is_symlink() or not validate_vector(path, cancel=self.cancel):
                return False
            reused.extend(self._members(path))

        self.outputs.extend(reused)
        self.skipped += len(expected)
        self.progress(
            ProgressEvent(
                "output",
                f"Reusing complete shapefile request ({len(expected)} layer(s))",
                current=len(expected),
                total=len(expected),
            )
        )
        return True

    def write(self, item: VectorSlice) -> Path:
        if self.cancel:
            self.cancel.check()
        destination = self._layer_path(item.variable)
        if destination in self._written:
            raise OutputError(
                f"Variable {item.variable!r} was written twice; one shapefile request "
                "must produce one layer per variable"
            )
        self._written.add(destination)
        if (
            destination.exists()
            and not self.request.overwrite
            and not _uses_latest(self.request)
            and validate_vector(destination, cancel=self.cancel)
        ):
            self.outputs.extend(self._members(destination))
            self.skipped += 1
            self.progress(ProgressEvent("output", f"Already complete: {destination.name}"))
            return destination

        self.progress(ProgressEvent("output", f"Writing {destination.name}"))
        self.root.mkdir(parents=True, exist_ok=True)
        # A shapefile layer is several files, so it is built in full beside the
        # target and only then moved in, keeping a failed run from leaving a
        # half-replaced set behind.
        staging = self.root / f".{destination.stem}.{uuid.uuid4().hex}.part"
        self._staging = staging
        self._installing = destination
        try:
            staging.mkdir(parents=True)
            staged = vector_to_shapefile(
                item.source,
                staging / destination.name,
                layer=item.layer,
                layer_name=destination.stem,
                cancel=self.cancel,
            )
            if not validate_vector(staged, cancel=self.cancel):
                raise OutputError(f"Generated shapefile failed validation: {destination.name}")
            # Move in every sidecar the new layer has and drop any left over from
            # an earlier run, so the installed set never mixes two conversions.
            for path in shapefile_members(staged):
                if self.cancel:
                    self.cancel.check()
                target = destination.parent / (destination.stem + path.suffix)
                if path.is_file():
                    os.replace(path, target)
                else:
                    target.unlink(missing_ok=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            self._staging = None
        self._installing = None

        written = self._members(destination)
        self.outputs.extend(written)
        counts = vector_layers(destination, cancel=self.cancel)
        features = next(iter(counts.values()), None) if counts else None
        self.progress(
            ProgressEvent(
                "output",
                f"Wrote {destination.name}"
                + (f" ({features} features)" if features is not None else ""),
            )
        )
        return destination

    def close(self) -> tuple[Path, ...]:
        if not self.outputs:
            raise OutputError("Provider produced no vector layers")
        return tuple(dict.fromkeys(self.outputs))

    def abort(self) -> None:
        if self._staging is not None:
            shutil.rmtree(self._staging, ignore_errors=True)
            self._staging = None
        if self._installing is not None:
            # An interrupted install can leave new and previous members side by
            # side; no output at all is safer than a layer that mixes both.
            for path in shapefile_members(self._installing):
                path.unlink(missing_ok=True)
            self._installing = None


Writer = TiffWriter | NetCDFWriter | ShapefileWriter


def make_writer(
    request: DownloadRequest,
    *,
    progress: ProgressCallback = null_progress,
    cancel: CancellationToken | None = None,
) -> Writer:
    if request.output_format.value == "tif":
        return TiffWriter(request, progress=progress, cancel=cancel)
    if request.output_format.value == "shp":
        return ShapefileWriter(request, progress=progress, cancel=cancel)
    return NetCDFWriter(request, progress=progress, cancel=cancel)
