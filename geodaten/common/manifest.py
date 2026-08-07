"""Write credential-free, resumable request provenance."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from geodaten._version import __version__
from geodaten.common.io import redacted_url, write_json_atomic
from geodaten.common.models import DownloadPlan, DownloadRequest

_SENSITIVE_KEY = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|credential|authorization|cookie)",
    re.IGNORECASE,
)


def _sanitized(value: Any, secrets: tuple[str, ...], *, key: str = "") -> Any:
    if key and _SENSITIVE_KEY.search(key):
        return "***"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitized(item, secrets, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (tuple, list, set)):
        return [_sanitized(item, secrets) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        text = value
        for secret in secrets:
            text = text.replace(secret, "***")
        if text.lower().startswith(("http://", "https://", "s3://")):
            text = redacted_url(text)
        return text
    return value


def write_manifest(
    path: Path,
    request: DownloadRequest,
    plan: DownloadPlan,
    outputs: Sequence[Path],
    *,
    provider_metadata: Mapping[str, Any] | None = None,
) -> Path:
    payload = {
        "schema": 1,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "software": {"name": "geodaten", "version": __version__},
        "request_fingerprint": request.fingerprint,
        "request": request.public_dict(),
        "plan": plan.public_dict(),
        "provider": dict(provider_metadata or {}),
        "outputs": [
            {"path": str(output), "size": output.stat().st_size}
            for output in outputs
            if output.is_file()
        ],
    }
    secrets = tuple(str(value) for value in request.credentials.values() if str(value))
    return write_json_atomic(path, _sanitized(payload, secrets))
