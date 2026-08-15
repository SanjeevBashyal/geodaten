"""Minimal provider contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Mapping

from geodaten.common.errors import ConfigurationError, DependencyError
from geodaten.common.models import (
    DatasetSpec,
    DownloadPlan,
    DownloadRequest,
    RasterSlice,
    VectorSlice,
)
from geodaten.common.progress import CancellationToken, ProgressCallback


class Provider(ABC):
    spec: DatasetSpec

    def validate(self, request: DownloadRequest) -> None:
        if request.dataset != self.spec.id:
            raise ConfigurationError(
                f"Provider {self.spec.id!r} cannot handle dataset {request.dataset!r}"
            )
        if request.output_format not in self.spec.formats:
            raise ConfigurationError(
                f"Dataset {self.spec.id} cannot be written as "
                f"{request.output_format.value!r}; choose from: "
                + ", ".join(self.spec.format_ids)
            )
        unknown_variables = sorted(set(request.variables) - set(self.spec.variable_ids))
        if unknown_variables:
            raise ConfigurationError(
                f"Unknown {self.spec.id} variable(s): {', '.join(unknown_variables)}"
            )
        if self.spec.temporal and request.time is None:
            raise ConfigurationError(f"Dataset {self.spec.id} requires a temporal range")
        if not self.spec.temporal and request.time is not None:
            raise ConfigurationError(f"Dataset {self.spec.id} is static and does not accept time")
        option_specs = {option.id: option for option in self.spec.options}
        unknown_options = sorted(set(request.options) - set(option_specs))
        if unknown_options:
            raise ConfigurationError(
                f"Unknown {self.spec.id} option(s): {', '.join(unknown_options)}"
            )
        for option in self.spec.options:
            value = request.options.get(option.id, option.default)
            if option.required and (value is None or value == "" or value == ()):
                raise ConfigurationError(f"Option {option.id!r} is required")
            if option.choices and value is not None:
                values = value if option.multiple and isinstance(value, (tuple, list)) else (value,)
                invalid = [item for item in values if item not in option.choices]
                if invalid:
                    raise ConfigurationError(
                        f"Invalid {option.id}: {invalid}; choose from {option.choices}"
                    )

    def option(self, request: DownloadRequest, name: str) -> Any:
        if name in request.options:
            return request.options[name]
        for option in self.spec.options:
            if option.id == name:
                return option.default
        raise KeyError(name)

    @abstractmethod
    def plan(self, request: DownloadRequest) -> DownloadPlan:
        """Validate/discover a request without downloading dataset payloads."""

    @abstractmethod
    def slices(
        self,
        request: DownloadRequest,
        workspace: Path,
        progress: ProgressCallback,
        cancel: CancellationToken,
    ) -> Iterator[RasterSlice | VectorSlice]:
        """Yield one local, cropped payload at a time.

        Raster providers yield a single-variable raster per logical slice; vector
        providers yield one cropped feature layer per variable.
        """

    def manifest_metadata(
        self, request: DownloadRequest, plan: DownloadPlan
    ) -> Mapping[str, Any]:
        return {
            "id": self.spec.id,
            "title": self.spec.title,
            "source_url": self.spec.source_url,
            "citation": self.spec.citation,
            "license": self.spec.license,
        }


def require_module(module: str, extra: str) -> None:
    try:
        available = find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    if not available:
        raise DependencyError(
            f"Optional dependency {module!r} is required; install geodaten[{extra}]"
        )
