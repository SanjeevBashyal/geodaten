"""Lazy built-in provider registry."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module

from geodaten.common.errors import ConfigurationError
from geodaten.common.models import DatasetSpec
from geodaten.downloader.base import Provider

_BUILTINS: dict[str, tuple[str, str]] = {
    "soilgrids": ("geodaten.downloader.soilgrids", "SoilGridsProvider"),
    "srtm": ("geodaten.downloader.srtm", "SrtmProvider"),
    "glc_fcs30d": ("geodaten.downloader.glc_fcs30d", "GlcFcs30dProvider"),
    "gimms_lai4g": ("geodaten.downloader.gimms_lai4g", "GimmsLai4gProvider"),
    "modis_lai": ("geodaten.downloader.modis_lai", "ModisLaiProvider"),
    "copernicus_lai": ("geodaten.downloader.copernicus_lai", "CopernicusLaiProvider"),
    "nex_gddp_cmip6": ("geodaten.downloader.nex_gddp_cmip6", "NexGddpProvider"),
    "ifs_fesom": ("geodaten.downloader.ifs_fesom", "IfsFesomProvider"),
}


@lru_cache(maxsize=None)
def get_provider(identifier: str) -> Provider:
    normalized = identifier.strip().lower()
    try:
        module_name, class_name = _BUILTINS[normalized]
    except KeyError as exc:
        raise ConfigurationError(
            f"Unknown dataset {identifier!r}; choose from: {', '.join(_BUILTINS)}"
        ) from exc
    provider_class = getattr(import_module(module_name), class_name)
    return provider_class()


def provider_specs() -> tuple[DatasetSpec, ...]:
    return tuple(get_provider(identifier).spec for identifier in _BUILTINS)


def provider_ids() -> tuple[str, ...]:
    return tuple(_BUILTINS)
