"""Shared plan and execution pipeline used by every interface."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from geodaten.common.manifest import write_manifest
from geodaten.common.models import DownloadPlan, DownloadRequest, DownloadResult
from geodaten.common.progress import CancellationToken, ProgressCallback, ProgressEvent, null_progress
from geodaten.common.writers import NetCDFWriter, TiffWriter, make_writer, output_root
from geodaten.downloader.registry import get_provider


def _runtime_warnings_from_manifest(path: Path) -> tuple[str, ...]:
    try:
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            return ()
        payload = json.loads(path.read_text(encoding="utf-8"))
        warnings = payload.get("provider", {}).get("runtime_warnings", ())
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        return ()
    if not isinstance(warnings, (list, tuple)):
        return ()
    return tuple(value for value in warnings if isinstance(value, str))


class DownloadService:
    def plan(self, request: DownloadRequest) -> DownloadPlan:
        provider = get_provider(request.dataset)
        provider.validate(request)
        return provider.plan(request)

    def download(
        self,
        request: DownloadRequest,
        *,
        progress: ProgressCallback = null_progress,
        cancel: CancellationToken | None = None,
    ) -> DownloadResult:
        token = cancel or CancellationToken()
        runtime_warnings: list[str] = []
        seen_warnings: set[str] = set()
        secrets = tuple(
            str(value) for value in request.credentials.values() if value is not None and str(value)
        )

        def report(event: ProgressEvent) -> None:
            message = event.message
            for secret in secrets:
                message = message.replace(secret, "***")
            if message != event.message:
                event = replace(event, message=message)
            if event.stage == "warning" and event.message not in seen_warnings:
                seen_warnings.add(event.message)
                runtime_warnings.append(event.message)
            progress(event)

        provider = get_provider(request.dataset)
        provider.validate(request)
        plan = provider.plan(request)
        report(ProgressEvent("plan", f"Plan ready for {request.dataset}"))
        writer = make_writer(request, progress=report, cancel=token)

        cache_root = request.cache_dir or (request.output_dir / ".geodaten-cache")
        try:
            tiff_complete = isinstance(writer, TiffWriter) and writer.resume_complete_request(
                plan.slice_count
            )
            if not tiff_complete and (
                not isinstance(writer, NetCDFWriter) or not writer.complete
            ):
                cache_root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=f"{request.dataset}_", dir=cache_root
                ) as temporary:
                    workspace = Path(temporary)
                    for item in provider.slices(request, workspace, report, token):
                        token.check()
                        writer.write(item)
            outputs = writer.close()
        except BaseException:
            writer.abort()
            raise

        root = output_root(request)
        root.mkdir(parents=True, exist_ok=True)
        reused_complete_request = tiff_complete or (
            isinstance(writer, NetCDFWriter) and writer.complete
        )
        if reused_complete_request:
            for warning in _runtime_warnings_from_manifest(root / "manifest.json"):
                if warning not in seen_warnings:
                    seen_warnings.add(warning)
                    runtime_warnings.append(warning)
        provider_metadata = dict(provider.manifest_metadata(request, plan))
        provider_metadata["runtime_warnings"] = tuple(runtime_warnings)
        manifest = write_manifest(
            root / "manifest.json",
            request,
            plan,
            outputs,
            provider_metadata=provider_metadata,
        )
        report(ProgressEvent("complete", f"Completed {len(outputs)} output file(s)"))
        return DownloadResult(
            request=request,
            outputs=tuple(outputs),
            manifest=manifest,
            skipped=writer.skipped,
        )
