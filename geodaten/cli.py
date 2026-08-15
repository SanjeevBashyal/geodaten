"""Small argparse interface over :class:`geodaten.DownloadService`."""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import sys
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Sequence

from geodaten._version import __version__
from geodaten.common.auth import resolve_credentials
from geodaten.common.errors import GeodatenError
from geodaten.common.models import BBox, DownloadRequest, OptionSpec, OutputFormat, TimeRange
from geodaten.common.progress import ProgressEvent
from geodaten.downloader.registry import get_provider, provider_ids, provider_specs
from geodaten.downloader.service import DownloadService


def _coerce(value: str, option: OptionSpec) -> Any:
    if option.kind in {"int", "integer"}:
        return int(value)
    if option.kind in {"float", "number"}:
        return float(value)
    if option.kind == "bool":
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{option.id} expects true or false")
    if option.kind == "path":
        return Path(value).expanduser()
    return value


def parse_provider_options(values: Sequence[str], specs: tuple[OptionSpec, ...]) -> dict[str, Any]:
    raw: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Provider option must be KEY=VALUE: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key or key in raw:
            raise ValueError(f"Duplicate or empty provider option: {key!r}")
        raw[key] = value.strip()

    known = {option.id: option for option in specs}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ValueError("Unknown provider option(s): " + ", ".join(unknown))
    parsed: dict[str, Any] = {}
    for option in specs:
        if option.id not in raw:
            continue
        value = raw[option.id]
        if option.multiple:
            parsed[option.id] = tuple(
                _coerce(part.strip(), option) for part in value.split(",") if part.strip()
            )
        else:
            parsed[option.id] = _coerce(value, option)
    return parsed


def _add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", choices=provider_ids())
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="WGS84 extent; WEST > EAST denotes an antimeridian crossing",
    )
    parser.add_argument("--start", help="inclusive start date/year or earliest")
    parser.add_argument("--end", help="inclusive end date/year or latest")
    parser.add_argument(
        "-v",
        "--variable",
        action="append",
        default=[],
        help="variable identifier; repeat for more than one",
    )
    parser.add_argument(
        "--format",
        choices=tuple(item.value for item in OutputFormat),
        default=None,
        help="output family; defaults to the first format the dataset supports",
    )
    parser.add_argument("--output", type=Path, default=Path("geodaten-output"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument(
        "-O",
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="provider-specific option; comma-separate multi-value options",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "provider concurrency hint (currently used mainly for connection-pool sizing; "
            "parallelism is not guaranteed)"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--prompt-credentials",
        action="store_true",
        help="prompt for credentials missing from the environment/config files",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geodaten", description=__doc__)
    parser.add_argument("--version", action="version", version=f"geodaten {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("datasets", help="list available datasets")
    info = subparsers.add_parser("info", help="show variables, options, and authentication")
    info.add_argument("dataset", choices=provider_ids())
    for name, help_text in (
        ("plan", "validate and summarize without downloading payloads"),
        ("download", "download and write the requested output"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        _add_request_arguments(command)
    subparsers.add_parser("doctor", help="check runtime programs and optional dependencies")
    subparsers.add_parser("tui", help="open the Textual terminal interface")
    subparsers.add_parser("gui", help="open the QtPy graphical interface")
    return parser


def request_from_args(args: argparse.Namespace) -> DownloadRequest:
    provider = get_provider(args.dataset)
    spec = provider.spec
    options = parse_provider_options(args.option, spec.options)
    variables = tuple(args.variable) or (spec.variables[0].id,)
    temporal = (
        TimeRange(args.start or "earliest", args.end or "latest") if spec.temporal else None
    )
    if not spec.temporal and (args.start or args.end):
        raise ValueError(f"{spec.id} is static; omit --start and --end")
    credentials = resolve_credentials(spec.credentials)
    if args.prompt_credentials:
        for field in spec.credentials:
            if credentials.get(field.id):
                continue
            prompt = f"{field.title}: "
            value = getpass.getpass(prompt) if field.secret else input(prompt)
            if value:
                credentials[field.id] = value
    return DownloadRequest(
        dataset=spec.id,
        bbox=BBox.from_values(args.bbox),
        time=temporal,
        variables=variables,
        output_format=OutputFormat(args.format) if args.format else spec.formats[0],
        output_dir=args.output,
        cache_dir=args.cache,
        options=options,
        credentials=credentials,
        workers=args.workers,
        overwrite=args.overwrite,
        allow_partial=args.allow_partial,
    )


class ConsoleProgress:
    def __init__(self) -> None:
        self._last: tuple[str, str, int | None] | None = None

    def __call__(self, event: ProgressEvent) -> None:
        percent: int | None = None
        if event.total and event.current is not None:
            percent = int(event.current * 100 / event.total) // 10 * 10
        elif event.bytes_total and event.bytes_done is not None:
            percent = int(event.bytes_done * 100 / event.bytes_total) // 10 * 10
        key = (event.stage, event.message, percent)
        if key == self._last:
            return
        self._last = key
        suffix = f" ({percent}%)" if percent is not None else ""
        print(f"[{event.stage}] {event.message}{suffix}", flush=True)


def _doctor() -> int:
    def module_available(name: str) -> bool:
        try:
            return find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    qt_binding = any(
        module_available(name) for name in ("PySide6", "PyQt6", "PySide2", "PyQt5")
    )
    checks = {
        "gdalinfo": shutil.which("gdalinfo") is not None,
        "gdal_translate": shutil.which("gdal_translate") is not None,
        "gdalbuildvrt": shutil.which("gdalbuildvrt") is not None,
        "gdalwarp": shutil.which("gdalwarp") is not None,
        "ogrinfo": shutil.which("ogrinfo") is not None,
        "ogr2ogr": shutil.which("ogr2ogr") is not None,
        "requests": module_available("requests"),
        "rasterio": module_available("rasterio"),
        "pyproj": module_available("pyproj"),
        "netCDF4": module_available("netCDF4"),
        "remotezip [zenodo]": module_available("remotezip"),
        "earthaccess [earthdata]": module_available("earthaccess"),
        "boto3 [cdse]": module_available("boto3"),
        "earthkit + Polytope [destine]": all(
            module_available(name)
            for name in ("earthkit.data", "earthkit.regrid", "polytope")
        ),
        "textual [tui]": module_available("textual"),
        "qtpy + Qt binding [gui]": module_available("qtpy") and qt_binding,
    }
    for name, available in checks.items():
        print(f"{'ok' if available else 'missing':7} {name}")
    required = (
        "gdalinfo",
        "gdal_translate",
        "gdalbuildvrt",
        "gdalwarp",
        "ogrinfo",
        "ogr2ogr",
        "requests",
        "rasterio",
        "pyproj",
        "netCDF4",
    )
    return 0 if all(checks[name] for name in required) else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "datasets":
            for spec in provider_specs():
                auth = "user credentials" if spec.credentials else "public"
                print(f"{spec.id:20} {auth:16} {spec.title}")
            return 0
        if args.command == "info":
            print(json.dumps(asdict(get_provider(args.dataset).spec), indent=2, default=str))
            return 0
        if args.command == "doctor":
            return _doctor()
        if args.command == "tui":
            try:
                from geodaten.tui.app import main as tui_main
            except ImportError as exc:
                raise GeodatenError(
                    "The terminal interface is unavailable; install geodaten[tui]"
                ) from exc

            return tui_main()
        if args.command == "gui":
            try:
                from geodaten.gui.app import main as gui_main
            except ImportError as exc:
                raise GeodatenError(
                    "The graphical interface is unavailable; install geodaten[gui-pyside6]"
                ) from exc

            return gui_main()

        request = request_from_args(args)
        service = DownloadService()
        if args.command == "plan":
            print(json.dumps(service.plan(request).public_dict(), indent=2, default=str))
            return 0
        result = service.download(request, progress=ConsoleProgress())
        for output in result.outputs:
            print(output)
        print(f"Manifest: {result.manifest}")
        return 0
    except (GeodatenError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


def tui_entry() -> int:
    return main(("tui",))


def gui_entry() -> int:
    return main(("gui",))


if __name__ == "__main__":
    raise SystemExit(main())
