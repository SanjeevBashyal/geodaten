# Agent guidance for geodaten

## Scope and intent

`geodaten` provides one Python service behind a CLI, Textual TUI, and QtPy GUI. Keep changes small, modular, and reusable. A provider must download only the requested spatial and temporal subset and must never embed developer credentials.

## Architecture

- Put shared immutable contracts, authentication, I/O, GDAL helpers, writers, progress, and errors in `geodaten/common/`.
- Put one source adapter per module in `geodaten/downloader/`; register built-ins lazily in `geodaten/downloader/registry.py`.
- Keep CLI, TUI, and GUI thin. They must derive dataset groups, variables, formats, options, temporal availability, and credential fields from `DatasetSpec` rather than duplicate provider metadata.
- Keep the Qt Designer form in `geodaten/gui/download.ui` and controller behavior in `geodaten/gui/app.py`. Use QtPy, not a binding-specific API.

## Provider contract

Each provider declares a complete `DatasetSpec` and implements `validate()`, `plan()`, and `slices()`:

- `validate()` rejects unsupported variables, formats, options, extents, dates, or combinations with actionable messages.
- `plan()` may discover metadata but must not download the requested payload. Report exact logical slice counts when practical, estimates, source choices, and all partial-coverage warnings.
- `slices()` yields a cropped `RasterSlice`, bounded multitime `RasterBlock`, or `VectorSlice` and checks `CancellationToken` around network and expensive work.
- Credentials come only from the request, documented environment variables, or the upstream service's standard user configuration. Never write secrets to logs, exceptions, URLs, filenames, fingerprints, manifests, tests, or fixtures.
- Use `allow_partial` only for explicitly reported omissions. Never silently clip an explicit request.

## Output invariants

- Raster datasets support `tif` and `netcdf`: produce one single-band COG per variable/time/dimension combination, or one request-level compressed CF-1.13 NetCDF file.
- Vector datasets declare `shp`: crop features to the requested WGS84 extent and preserve the source attribute table and complete shapefile sidecar set.
- A logical raster slice must cover the complete requested extent on one consistent grid. NetCDF requests cannot mix grids.
- Stage temporary data below the request workspace, validate before publication, and use atomic installation where shared helpers provide it.

## Large-data behavior

- Stream HTTP bodies and process bounded chunks, windows, months, days, or variables. Use `RasterBlock` when a source can retrieve several times efficiently, but do not materialize a full multi-variable temporal cube.
- Reuse validated immutable source assets when useful, keep discovery caches bounded, and consume request-specific cached state after execution.
- Avoid unbounded concurrency. `workers` is only a provider hint unless an adapter explicitly implements bounded parallel work.
- A request containing literal `latest` must refresh discovery; explicit reproducible requests may reuse complete validated outputs.

## Development workflow

- Preserve unrelated user changes in a dirty worktree. Use focused patches and do not use destructive Git commands.
- Add mocked tests for metadata parsing, strict/partial coverage, credential resolution, planning, and slice invariants. Tests must not require live services or real accounts.
- Update the registry, optional dependency extras, `geodaten doctor`, README dataset table, and dynamic-interface expectations when adding a provider.
- Run `python3 -m pytest -q`, `python3 -m compileall -q geodaten`, `git diff --check`, and an isolated package build when packaging changes.
