"""Small authenticated helpers shared by ECMWF ARCO providers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from geodaten.common.errors import AuthenticationError, DownloadError
from geodaten.downloader.base import require_module


def resolve_cds_api_key(
    supplied: Mapping[str, str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    rc_path: Path | None = None,
) -> str:
    """Resolve a caller-owned CDS token without persisting or displaying it."""

    supplied = supplied or {}
    environ = os.environ if environ is None else environ
    value = supplied.get("cds_api_key") or environ.get("CDSAPI_KEY")
    if value is not None and str(value).strip():
        return str(value).strip()

    path = rc_path or (Path.home() / ".cdsapirc")
    if path.is_file():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                key, separator, candidate = line.partition(":")
                if separator and key.strip().lower() == "key" and candidate.strip():
                    return candidate.strip()
        except (OSError, UnicodeError) as exc:
            raise AuthenticationError(f"CDS credential file is unreadable: {path}") from exc

    raise AuthenticationError(
        "Provide your CDS API key with request credentials, CDSAPI_KEY, or ~/.cdsapirc"
    )


def open_arco_dataset(url: str, api_key: str) -> Any:
    """Open one consolidated HTTP Zarr store using the user's bearer token."""

    for module in ("xarray", "zarr", "fsspec"):
        require_module(module, "arco")
    import xarray as xr

    try:
        return xr.open_zarr(
            url,
            consolidated=True,
            chunks=None,
            storage_options={"headers": {"Authorization": f"Bearer {api_key}"}},
        )
    except Exception as exc:
        message = str(exc).lower()
        if any(marker in message for marker in ("401", "403", "unauthorized", "forbidden")):
            raise AuthenticationError(
                "The ERA5-Land ARCO service rejected the supplied CDS API key; "
                "check the token and accept the dataset licence in the CDS"
            ) from exc
        raise DownloadError(
            f"Could not open ERA5-Land ARCO store ({type(exc).__name__})"
        ) from exc
