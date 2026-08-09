"""Windowed access to the public SoilGrids 250 m VRT mosaics."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from geodaten.common.errors import ConfigurationError
from geodaten.common.gdal import translate_to_cog
from geodaten.common.io import safe_component
from geodaten.common.models import (
    DatasetSpec,
    DownloadPlan,
    DownloadRequest,
    OptionSpec,
    RasterSlice,
    VariableSpec,
)
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent
from geodaten.downloader.base import Provider

BASE_URL = "https://files.isric.org/soilgrids/latest/data"
DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm")
OCS_DEPTHS = ("0-30cm",)
STATISTICS = ("mean", "Q0.05", "Q0.5", "Q0.95", "uncertainty")

# Scale factors convert the compact integer maps to SoilGrids' conventional
# units without materialising floating-point rasters.
_VARIABLES: tuple[tuple[str, str, str, float], ...] = (
    ("bdod", "Bulk density", "kg dm-3", 0.01),
    ("cec", "Cation exchange capacity", "cmol kg-1", 0.1),
    ("cfvo", "Coarse fragments", "%", 0.1),
    ("clay", "Clay content", "%", 0.1),
    ("nitrogen", "Total nitrogen", "g kg-1", 0.01),
    ("ocd", "Organic carbon density", "kg m-3", 0.1),
    ("ocs", "Organic carbon stock", "kg m-2", 0.1),
    ("phh2o", "Soil pH in water", "1", 0.1),
    ("sand", "Sand content", "%", 0.1),
    ("silt", "Silt content", "%", 0.1),
    ("soc", "Soil organic carbon", "g kg-1", 0.1),
)
_VARIABLE_METADATA = {
    identifier: {"title": title, "units": units, "scale_factor": scale}
    for identifier, title, units, scale in _VARIABLES
}


def remote_vrt(variable: str, depth: str, statistic: str) -> str:
    """Return a GDAL range-readable path for one official remote VRT."""

    layer = f"{variable}_{depth}_{statistic}"
    url = f"{BASE_URL}/{variable}/{layer}.vrt"
    return "/vsicurl?max_retry=5&retry_delay=2&list_dir=no&url=" + url


def _selected_values(
    request: DownloadRequest, option: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    value: Any = request.options.get(option)
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(dict.fromkeys(str(item) for item in value))


def selected_layers(request: DownloadRequest) -> tuple[tuple[str, str, str], ...]:
    """Resolve variable-specific default depths into concrete source layers."""

    statistics = _selected_values(request, "statistics", ("mean",))
    explicit_depths = request.options.get("depths") is not None
    requested_depths = _selected_values(request, "depths", DEPTHS)
    layers: list[tuple[str, str, str]] = []
    for variable in request.variables:
        available = OCS_DEPTHS if variable == "ocs" else DEPTHS
        depths = requested_depths if explicit_depths else available
        invalid = tuple(depth for depth in depths if depth not in available)
        if invalid:
            raise ConfigurationError(
                f"SoilGrids variable {variable!r} does not provide depth(s): "
                + ", ".join(invalid)
            )
        layers.extend((variable, depth, statistic) for statistic in statistics for depth in depths)
    return tuple(layers)


class SoilGridsProvider(Provider):
    spec = DatasetSpec(
        id="soilgrids",
        title="SoilGrids 250 m",
        summary=(
            "Global ISRIC soil-property predictions at native 250 m resolution, "
            "read directly from public VRT mosaics."
        ),
        group="Soil",
        variables=tuple(
            VariableSpec(
                identifier,
                title,
                units=units,
                description=(
                    f"Packed SoilGrids integer values with scale factor {scale:g}."
                ),
            )
            for identifier, title, units, scale in _VARIABLES
        ),
        temporal=False,
        options=(
            OptionSpec(
                id="depths",
                title="Soil depths",
                kind="choice",
                default=None,
                choices=DEPTHS + OCS_DEPTHS,
                multiple=True,
                description=(
                    "Depth intervals to download. By default every depth available for each "
                    "selected property is used; ocs is available only for 0-30cm."
                ),
            ),
            OptionSpec(
                id="statistics",
                title="Prediction statistics",
                kind="choice",
                default=("mean",),
                choices=STATISTICS,
                multiple=True,
                description="Mean, quantiles, and/or relative uncertainty.",
            ),
        ),
        source_url=BASE_URL + "/",
        citation="Poggio et al. (2021), SoilGrids 2.0, SOIL 7, 217-240.",
        license="CC BY 4.0",
    )

    def validate(self, request: DownloadRequest) -> None:
        super().validate(request)
        if request.bbox.crosses_antimeridian:
            raise ConfigurationError("SoilGrids subsets cannot cross the antimeridian")
        if not _selected_values(request, "statistics", ("mean",)):
            raise ConfigurationError("Select at least one SoilGrids statistic")
        if request.options.get("depths") is not None and not _selected_values(
            request, "depths", DEPTHS
        ):
            raise ConfigurationError("Select at least one SoilGrids depth")
        selected_layers(request)

    def plan(self, request: DownloadRequest) -> DownloadPlan:
        self.validate(request)
        layers = selected_layers(request)
        return DownloadPlan(
            dataset=self.spec.id,
            period_count=1,
            slice_count=len(layers),
            details={
                "layers": tuple("_".join(layer) for layer in layers),
                "native_resolution_m": 250,
                "native_crs": "Interrupted Goode Homolosine",
                "access": "anonymous HTTP range reads",
            },
        )

    def slices(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Iterator[RasterSlice]:
        self.validate(request)
        layers = selected_layers(request)
        slice_dir = workspace / "soilgrids"
        slice_dir.mkdir(parents=True, exist_ok=True)
        for index, (variable, depth, statistic) in enumerate(layers, start=1):
            cancel.check()
            layer = f"{variable}_{depth}_{statistic}"
            progress(
                ProgressEvent(
                    "download",
                    f"Reading SoilGrids layer {layer}",
                    current=index,
                    total=len(layers),
                )
            )
            local = slice_dir / f"{safe_component(layer)}.tif"
            translate_to_cog(
                remote_vrt(variable, depth, statistic),
                local,
                bbox=request.bbox,
                metadata={"source_layer": layer},
                cancel=cancel,
            )
            metadata = _VARIABLE_METADATA[variable]
            uncertainty = statistic == "uncertainty"
            attrs: dict[str, Any] = {
                "long_name": (
                    f"{metadata['title']} relative uncertainty"
                    if uncertainty
                    else metadata["title"]
                ),
                "units": "1" if uncertainty else metadata["units"],
                "scale_factor": 0.1 if uncertainty else metadata["scale_factor"],
                "source": f"SoilGrids {layer}",
            }
            try:
                yield RasterSlice(
                    # Uncertainty is a dimensionless ratio with a different
                    # packing scale, so it must not share a NetCDF variable
                    # with the property's prediction statistics.
                    variable=f"{variable}_uncertainty" if uncertainty else variable,
                    source=local,
                    dimensions=(
                        {"depth": depth}
                        if uncertainty
                        else {"depth": depth, "statistic": statistic}
                    ),
                    attrs=attrs,
                )
            finally:
                local.unlink(missing_ok=True)
