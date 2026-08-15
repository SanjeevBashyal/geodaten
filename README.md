# geodaten

`geodaten` downloads spatial and temporal subsets of global geospatial datasets through one Python API, CLI, Textual TUI, or QtPy GUI.

The package streams source assets, stages one logical slice at a time, and writes only these final formats:

- `tif`: one single-band Cloud Optimized GeoTIFF for each requested variable/time combination, mosaicked over the complete requested extent.
- `netcdf`: exactly one compressed NetCDF4 file for the complete request, written block by block and marked `CF-1.13`.
- `shp`: one ESRI Shapefile layer for each requested variable of a vector dataset, cropped to the extent in EPSG:4326 with its published attribute table.

Each dataset declares the formats it can produce, so a raster source never offers `shp` and a vector source never offers `tif` or `netcdf`. `geodaten info DATASET` lists them, and `--format` defaults to the first one.

Credentials are always supplied by the user through environment variables, an SDK's own configuration, the Python request, or a masked UI prompt. They are never written to filenames, logs, or manifests.

## Supported datasets

| ID | Dataset family | Formats | Access |
| --- | --- | --- | --- |
| `soilgrids` | ISRIC SoilGrids | `tif`, `netcdf` | Public |
| `srtm` | NASA SRTMGL1 v3 | `tif`, `netcdf` | NASA Earthdata account |
| `glc_fcs30d` | GLC_FCS30D v2 land cover | `tif`, `netcdf` | Public Zenodo archive |
| `glim` | GLiM v1.1 global lithological map | `shp` | Public download |
| `gimms_lai4g` | GIMMS LAI4g v1.2 | `tif`, `netcdf` | Public Zenodo archive |
| `modis_lai` | MODIS LAI/FPAR C6.1 | `tif`, `netcdf` | NASA Earthdata account |
| `copernicus_lai` | Copernicus Global Land LAI 300 m v2 | `tif`, `netcdf` | Copernicus Data Space account |
| `nex_gddp_cmip6` | NASA NEX-GDDP-CMIP6 | `tif`, `netcdf` | Public cloud archive |
| `ifs_fesom` | DestinE IFS-FESOM climate simulations | `tif`, `netcdf` | DestinE account |

Run `geodaten info DATASET` for its variables, provider options, credential environment names, source, citation, and license.

## Installation

Python 3.12 or newer and the GDAL command-line programs (`gdalinfo`, `gdal_translate`, `gdalbuildvrt`, and `gdalwarp`) are required. Install the core package from this checkout with:

```bash
python -m pip install -e .
```

Install only the adapters and interfaces you use:

```bash
python -m pip install -e ".[zenodo]"       # GLC_FCS30D and GIMMS
python -m pip install -e ".[earthdata]"    # SRTM and MODIS
python -m pip install -e ".[cdse]"         # Copernicus LAI
python -m pip install -e ".[destine]"      # IFS-FESOM
python -m pip install -e ".[tui]"
python -m pip install -e ".[gui-pyside6]"
```

For every optional adapter and interface, use `python -m pip install -e ".[all,gui-pyside6]"`. Check the current machine with `geodaten doctor`.

## CLI

Discover a dataset before starting a large request:

```bash
geodaten datasets
geodaten info srtm
geodaten plan srtm --bbox 5 47 15 55 -v elevation --format tif
```

Then run the same request as a download:

```bash
geodaten download srtm \
  --bbox 5 47 15 55 \
  -v elevation \
  --format tif \
  --output ./data
```

A static vector dataset is requested the same way, and its extent crop is written as a shapefile:

```bash
geodaten download glim \
  --bbox 79 26 90 31 \
  -v lithology \
  --output ./data
```

Temporal datasets additionally accept inclusive `--start` and `--end` values. Repeat `-v` for multiple variables and pass adapter-specific settings as `-O KEY=VALUE`; comma-separate values for a multi-value option. `WEST > EAST` represents an antimeridian-crossing bounding box where the selected source supports such a crop; `plan` reports a clear error otherwise.

Credentials are normally read from the environment or the source SDK's user configuration. Add `--prompt-credentials` to enter missing values without echoing secrets:

```bash
geodaten download modis_lai \
  --bbox 5 47 15 55 \
  --start 2024-01-01 --end 2024-01-31 \
  -v lai --prompt-credentials
```

Use `--allow-partial` only when an incomplete source coverage is acceptable. Existing validated outputs are reused unless `--overwrite` is set. A temporal request containing literal `latest` deliberately refreshes discovery and outputs; use explicit dates for reproducible resume. `--workers` is a provider concurrency hint, currently used mainly to size connection pools; a provider may still run safely in sequence.

## Python API

```python
from pathlib import Path

from geodaten import BBox, DownloadRequest, DownloadService, OutputFormat

request = DownloadRequest(
    dataset="srtm",
    bbox=BBox(west=5, south=47, east=15, north=55),
    variables=("elevation",),
    output_format=OutputFormat.TIF,
    output_dir=Path("data"),
)

service = DownloadService()
plan = service.plan(request)
result = service.download(request)
print(result.outputs, result.manifest)
```

For a credentialed adapter, pass only your own values in `credentials={...}`. The accepted keys are listed by `geodaten info DATASET`. The exported `CancellationToken` and `ProgressCallback` can be supplied to `download()` by long-running applications.

## TUI and GUI

```bash
geodaten-tui
geodaten-gui
```

Both interfaces collect the dataset, variables, WGS84 extent, dates, output format/directory, a provider worker hint, provider options, and user credentials. The hint does not guarantee parallel execution. The GUI controller uses QtPy and loads the packaged Qt Designer file at `geodaten/gui/download.ui`; the `gui-pyside6` extra supplies the default Qt binding.

## Output layout and large requests

Each request is stored below `OUTPUT/DATASET/FINGERPRINT/` with a credential-free `manifest.json`. The fingerprint makes retries deterministic while excluding credentials, worker hints, cache locations, and output paths. Files are first written to a temporary path and atomically moved into place only after validation.

Raster transfers and conversions use streaming I/O and GDAL virtual rasters. NetCDF data are compressed and appended in bounded raster windows, so memory use is based on a processing block rather than the complete spatial/temporal cube. `--cache` selects a staging location; adapters such as MODIS reuse validated source assets, while request scratch data and incomplete NetCDF files are deliberately removed. A retry reuses an exact, validated complete TIFF set without restaging provider data. Without `--overwrite`, a partial TIFF retry may repeat provider staging but never overwrites already validated completed slices. Source services may still impose request-size, rate, date-range, or account limits, so use `plan` before large jobs.

A `shp` request writes the whole `.shp`/`.shx`/`.dbf`/`.prj`/`.cpg` set for each layer, all of which are listed in the manifest; the set is built beside the target and moved in together, and a retry reuses a validated complete set without restaging. Vector sources that ship as one large archive are cropped locally, so the source is cached below `--cache` and reused by later extents; `geodaten info glim` documents the options that relocate, reuse, or discard it.

## Development

```bash
python -m pytest -q
python -m build
```

Tests mock remote catalogs and authentication. Live downloads are intentionally not required, because users must provide and authorize their own service accounts.
