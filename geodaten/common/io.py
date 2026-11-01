"""Streaming HTTP and atomic filesystem helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping
from urllib.parse import urlsplit, urlunsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from geodaten.common.errors import DownloadError
from geodaten.common.models import DownloadRequest
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent, null_progress


def cache_root(request: DownloadRequest) -> Path:
    """Return the staging root shared by the service and source-caching providers.

    A provider that reuses one large monolithic source across requests writes it
    below this directory so that only request scratch data is discarded.
    """

    return request.cache_dir or (request.output_dir / ".geodaten-cache")


def safe_component(value: Any, *, fallback: str = "value") -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-._")
    return text or fallback


def redacted_url(value: str) -> str:
    """Remove credentials and query/fragment tokens from a URL."""

    try:
        parsed = urlsplit(str(value))
        if parsed.scheme.lower() not in {"http", "https", "s3"} or not parsed.hostname:
            return str(value)
        hostname = parsed.hostname
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname + (f":{parsed.port}" if parsed.port else "")
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "<redacted-url>"


def make_session(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    retry = Retry(
        total=max(0, retries),
        connect=max(0, retries),
        read=max(0, retries),
        status=max(0, retries),
        backoff_factor=max(0.0, backoff),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers["User-Agent"] = "geodaten/0.1"
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def remote_exists(session: Any, url: str, label: str) -> bool:
    """Check one public HTTP asset without downloading its body."""

    response: Any = None
    try:
        response = session.head(url, allow_redirects=True, timeout=(30, 90))
        if response.status_code in {405, 501}:
            response.close()
            response = session.get(
                url,
                headers={"Range": "bytes=0-0"},
                stream=True,
                allow_redirects=True,
                timeout=(30, 90),
            )
        with response:
            if response.status_code == 404:
                return False
            response.raise_for_status()
            return True
    except requests.RequestException as exc:
        raise DownloadError(f"Could not check availability of {label}") from exc


@contextmanager
def atomic_path(final: Path, *, suffix: str = ".part") -> Iterator[Path]:
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(f".{final.name}.{uuid.uuid4().hex}{suffix}")
    try:
        yield temporary
        os.replace(temporary, final)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def stream_download(
    url: str,
    destination: Path,
    *,
    session: requests.Session | None = None,
    headers: Mapping[str, str] | None = None,
    expected_size: int | None = None,
    timeout: tuple[float, float] = (30.0, 300.0),
    chunk_size: int = 8 * 1024 * 1024,
    progress: ProgressCallback = null_progress,
    cancel: CancellationToken | None = None,
) -> Path:
    own_session = session is None
    active = session or make_session()
    try:
        with active.get(url, headers=dict(headers or {}), stream=True, timeout=timeout) as response:
            response.raise_for_status()
            declared = int(response.headers.get("Content-Length", "0") or 0) or expected_size
            with atomic_path(destination) as temporary, temporary.open("wb") as target:
                copied = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if cancel:
                        cancel.check()
                    if not chunk:
                        continue
                    target.write(chunk)
                    copied += len(chunk)
                    progress(
                        ProgressEvent(
                            "download",
                            destination.name,
                            bytes_done=copied,
                            bytes_total=declared,
                        )
                    )
                if expected_size is not None and copied != expected_size:
                    raise DownloadError(
                        f"Size mismatch for {destination.name}: expected {expected_size}, received {copied}"
                    )
        return destination
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f"HTTP {status}" if status is not None else type(exc).__name__
        raise DownloadError(f"Could not download {redacted_url(url)} ({detail})") from exc
    finally:
        if own_session:
            active.close()


def copy_stream(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    chunk_size: int = 8 * 1024 * 1024,
    cancel: CancellationToken | None = None,
) -> int:
    copied = 0
    while True:
        if cancel:
            cancel.check()
        chunk = source.read(chunk_size)
        if not chunk:
            return copied
        destination.write(chunk)
        copied += len(chunk)


def checksum(path: Path, algorithm: str = "sha256", chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def available_bytes(path: Path) -> int:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
