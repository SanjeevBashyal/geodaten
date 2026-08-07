from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from geodaten.common.errors import CancelledError, ConfigurationError
from geodaten.common.models import BBox, DownloadRequest, OutputFormat, TimeRange
from geodaten.common.progress import CancellationToken
from geodaten.downloader import copernicus_lai as copernicus
from geodaten.downloader import modis_lai as modis


def request(
    dataset: str,
    variables: tuple[str, ...],
    *,
    time: TimeRange = TimeRange("2025-01-01", "2025-01-10"),
    options: dict[str, object] | None = None,
    credentials: dict[str, str] | None = None,
    output_format: OutputFormat = OutputFormat.TIF,
) -> DownloadRequest:
    return DownloadRequest(
        dataset=dataset,
        bbox=BBox(79, 26, 80, 27),
        variables=variables,
        time=time,
        output_format=output_format,
        output_dir=Path("outputs"),
        options=options or {},
        credentials=credentials or {},
    )


class FakeGranule(dict):
    def __init__(self, name: str, beginning: str, ending: str, *, size: float = 1.0):
        super().__init__(
            meta={"native-id": name.removesuffix(".hdf")},
            umm={
                "TemporalExtent": {
                    "RangeDateTime": {
                        "BeginningDateTime": beginning,
                        "EndingDateTime": ending,
                    }
                }
            },
            size=size,
        )
        self.name = name

    def data_links(self, access: str | None = None) -> list[str]:
        del access
        return [f"https://example.test/data/{self.name}"]


def product(
    day: date,
    *,
    sensor: str = "OLCI",
    revision: int | None = 6,
    version: str = "2.0.1",
    modified: str = "2026-01-01T00:00:00Z",
    suffix: str = "",
) -> copernicus.Product:
    rt = "" if revision is None else f"-RT{revision}"
    name = f"c_gls_LAI300{rt}_{day:%Y%m%d}0000_GLOBE_{sensor}_V{version}_cog{suffix}"
    return copernicus.Product(
        product_id=name,
        name=name,
        content_length=100,
        nominal_date=day,
        modification_date=copernicus._parse_datetime(modified),
        s3_path=f"s3://eodata/products/{name}",
    )


def test_modis_spec_exposes_all_collection_61_products_and_user_auth() -> None:
    spec = modis.ModisLaiProvider.spec
    option = {item.id: item for item in spec.options}["product"]
    assert option.default == "MCD15A3H"
    assert set(option.choices) == {"MCD15A3H", "MCD15A2H", "MOD15A2H", "MYD15A2H"}
    assert {field.id for field in spec.credentials} == {
        "earthdata_username",
        "earthdata_password",
        "earthdata_token",
    }
    assert all(field.required is False for field in spec.credentials)
    assert {variable.id for variable in spec.variables} == set(modis.LAYERS)


def test_modis_selects_exact_nominal_dates_and_deduplicates_year_boundary() -> None:
    before_request = FakeGranule(
        "MCD15A3H.A2024361.h25v06.061.1.hdf",
        "2024-12-26T00:00:00Z",
        "2025-01-02T23:59:59Z",
    )
    current = FakeGranule(
        "MCD15A3H.A2025001.h25v06.061.1.hdf",
        "2025-01-01T00:00:00Z",
        "2025-01-04T23:59:59Z",
    )
    selected = modis._groups_for_window(
        [before_request, current],
        window_start=date(2025, 1, 1),
        window_end=date(2025, 1, 1),
    )
    assert [period for period, _ in selected] == [date(2025, 1, 1)]

    year_end = FakeGranule(
        "MCD15A3H.A2024365.h25v06.061.1.hdf",
        "2024-12-30T00:00:00Z",
        "2025-01-02T23:59:59Z",
    )
    december = modis._groups_for_window(
        [year_end], window_start=date(2024, 12, 30), window_end=date(2024, 12, 31)
    )
    january = modis._groups_for_window(
        [year_end, current], window_start=date(2025, 1, 1), window_end=date(2025, 1, 1)
    )
    assert [period for period, _ in december] == [date(2024, 12, 30)]
    assert [period for period, _ in january] == [date(2025, 1, 1)]


def test_modis_hdf_sds_matching_is_exact() -> None:
    metadata = {
        "metadata": {
            "SUBDATASETS": {
                "SUBDATASET_1_NAME": 'HDF4_EOS:EOS_GRID:"tile.hdf":grid:Lai_500m',
                "SUBDATASET_2_NAME": 'HDF4_EOS:EOS_GRID:"tile.hdf":grid:FparLai_QC',
                "SUBDATASET_2_DESC": "ignored",
            }
        }
    }
    assert modis.find_subdataset(metadata, "Lai_500m").endswith(":Lai_500m")
    with pytest.raises(Exception, match="Expected one"):
        modis.find_subdataset(metadata, "Fpar_500m")


def test_modis_cache_validation_propagates_cancellation(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "tile.hdf"
    source.write_bytes(b"hdf")

    def cancelled(*_args, **_kwargs):
        raise CancelledError("cancelled")

    monkeypatch.setattr(modis.gdal, "info", cancelled)
    with pytest.raises(CancelledError):
        modis.ModisLaiProvider._valid_hdf(source, ("lai",), CancellationToken())


def test_modis_plan_counts_composites_without_downloading(monkeypatch) -> None:
    before_request = FakeGranule(
        "MCD15A3H.A2024361.h25v06.061.1.hdf",
        "2024-12-26T00:00:00Z",
        "2025-01-02T23:59:59Z",
        size=9.0,
    )
    granules = [before_request] + [
        FakeGranule(
            f"MCD15A3H.A2025001.h2{tile}v06.061.1.hdf",
            "2025-01-01T00:00:00Z",
            "2025-01-04T23:59:59Z",
            size=2.5,
        )
        for tile in (4, 5)
    ]
    monkeypatch.setattr(modis, "_search_data", lambda *_args, **_kwargs: granules)
    item = request(
        "modis_lai",
        ("lai", "fpar_lai_qc"),
        time=TimeRange("2025", "2025"),
    )
    plan = modis.ModisLaiProvider().plan(item)
    assert plan.period_count == 1
    assert plan.slice_count == 2
    assert plan.details["granules"] == 2
    assert plan.details["product"] == "MCD15A3H"
    assert plan.estimated_bytes == 5 * 1024 * 1024


def test_modis_iter_groups_deduplicates_cross_year_cmr_results(monkeypatch) -> None:
    year_end = FakeGranule(
        "MCD15A3H.A2024365.h25v06.061.1.hdf",
        "2024-12-30T00:00:00Z",
        "2025-01-02T23:59:59Z",
    )
    new_year = FakeGranule(
        "MCD15A3H.A2025001.h25v06.061.1.hdf",
        "2025-01-01T00:00:00Z",
        "2025-01-04T23:59:59Z",
    )

    def fake_search(_product, start, _end, _bbox):
        return [year_end] if start.year == 2024 else [year_end, new_year]

    monkeypatch.setattr(modis, "_search_data", fake_search)
    item = request(
        "modis_lai",
        ("lai",),
        time=TimeRange("2024-12-30", "2025-01-01"),
    )
    groups = list(
        modis.ModisLaiProvider()._iter_groups(
            item, date(2024, 12, 30), date(2025, 1, 1)
        )
    )
    assert [period for period, _ in groups] == [date(2024, 12, 30), date(2025, 1, 1)]


def test_modis_slice_extracts_only_requested_sds(monkeypatch, tmp_path: Path) -> None:
    provider = modis.ModisLaiProvider()
    item = request("modis_lai", ("lai",), time=TimeRange("2025-01-01", "2025-01-01"))
    granule = FakeGranule(
        "MCD15A3H.A2025001.h25v06.061.1.hdf",
        "2025-01-01T00:00:00Z",
        "2025-01-04T23:59:59Z",
    )
    hdf = tmp_path / granule.name
    hdf.write_bytes(b"hdf")
    metadata = {
        "metadata": {
            "SUBDATASETS": {
                "SUBDATASET_1_NAME": f'HDF4_EOS:EOS_GRID:"{hdf}":grid:Lai_500m'
            }
        }
    }

    @contextmanager
    def fake_session(_request):
        yield object()

    monkeypatch.setattr(provider, "_earthdata_session", fake_session)
    discovery_calls = 0

    def fake_groups(*_args, **_kwargs):
        nonlocal discovery_calls
        discovery_calls += 1
        return iter(((date(2025, 1, 1), [granule]),))

    monkeypatch.setattr(provider, "_iter_groups", fake_groups)
    monkeypatch.setattr(provider, "_cached_granules", lambda *_args: [hdf])
    monkeypatch.setattr(modis.gdal, "require_commands", lambda *_args: None)
    info_tokens: list[CancellationToken] = []

    def fake_info(_path, *, cancel=None, **_kwargs):
        info_tokens.append(cancel)
        return metadata

    monkeypatch.setattr(modis.gdal, "info", fake_info)
    captured: list[list[str]] = []
    mosaic_tokens: list[CancellationToken] = []

    def fake_mosaic(sources, destination, *, cancel=None, **_kwargs):
        captured.append(list(sources))
        mosaic_tokens.append(cancel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"tif")
        return destination

    monkeypatch.setattr(modis.gdal, "mosaic_to_cog", fake_mosaic)
    token = CancellationToken()
    assert provider.plan(item).slice_count == 1
    slices = list(provider.slices(item, tmp_path / "work", lambda _event: None, token))
    assert len(slices) == 1
    assert slices[0].variable == "lai"
    assert captured[0][0].endswith(":Lai_500m")
    assert info_tokens == [token]
    assert mosaic_tokens == [token]
    assert discovery_calls == 1


def test_copernicus_catalog_parser_and_dynamic_revision_selection() -> None:
    text = "\n".join(
        (
            "id;name;content_length;nominal_date;modification_date;s3_path",
            "old;c_gls_LAI300-RT1_202507100000_GLOBE_OLCI_V2.0.1_cog;10;"
            "2025-07-10T00:00:00.000;2025-07-15T00:00:00.000;"
            "s3://eodata/products/old",
            "new;c_gls_LAI300-RT6_202507100000_GLOBE_OLCI_V2.0.1_cog;20;"
            "2025-07-10T00:00:00.000;2025-08-01T00:00:00.000;"
            "s3://eodata/products/new",
        )
    )
    parsed = copernicus.parse_catalog(text)
    selected = copernicus.select_products(parsed, date(2025, 7, 1), date(2025, 7, 31))
    assert len(parsed) == 2
    assert [item.revision for item in selected] == [6]

    # Sensor names do not participate in the ranking; the live catalogue
    # itself determines which product/revision is current for each date.
    day = date(2019, 1, 10)
    candidates = [
        product(day, sensor="PROBAV", revision=2),
        product(day, sensor="OLCI", revision=6),
    ]
    assert copernicus.select_products(candidates, day, day)[0].sensor == "OLCI"


def test_copernicus_consolidated_and_new_version_rank_first() -> None:
    day = date(2024, 4, 10)
    assert copernicus.product_score(product(day, revision=None)) > copernicus.product_score(
        product(day, revision=6)
    )
    assert copernicus.product_score(
        product(day, revision=0, version="2.1.0")
    ) > copernicus.product_score(product(day, revision=None, version="2.0.1"))


@pytest.mark.parametrize(
    ("filename", "variable"),
    (
        ("c_gls_LAI300-LAI-RT6_202501100000_GLOBE_OLCI_V2.0.1.tiff", "lai"),
        ("c_gls_LAI300-LBEFORE_201901100000_GLOBE_PROBAV_V2.0.1.tif", "length_before"),
        ("c_gls_LAI300-LAFTER-RT2_202501100000_GLOBE_OLCI_V2.0.1.tif", "length_after"),
        ("c_gls_LAI300-NOBS-RT0_202501100000_GLOBE_OLCI_V2.0.1.tif", "nobs"),
        ("c_gls_LAI300-QFLAG-RT1_202501100000_GLOBE_OLCI_V2.0.1.tif", "qflag"),
        ("c_gls_LAI300-RMSE-RT6_202501100000_GLOBE_OLCI_V2.0.1.tif", "rmse"),
    ),
)
def test_copernicus_object_matching(filename: str, variable: str) -> None:
    key = f"prefix/product/{filename}"
    assert copernicus.object_variable(key) == variable
    assert copernicus.object_matches(key, variable)
    assert not copernicus.object_matches(key, tuple(set(copernicus.VARIABLE_CODES) - {variable}))


def test_copernicus_object_matching_rejects_ancillary_files() -> None:
    assert copernicus.object_variable("c_gls_LAI300-LAI-RT6_20250110.xml") is None
    assert copernicus.object_variable("quicklook-LAI.tif") is None
    assert copernicus.parse_s3_path("s3://eodata/a/product") == ("eodata", "a/product/")
    with pytest.raises(ValueError, match="Invalid S3"):
        copernicus.parse_s3_path("https://example.test/not-s3")


def test_copernicus_options_and_credentials_are_restricted_to_official_inputs() -> None:
    spec = copernicus.CopernicusLaiProvider.spec
    endpoint = {item.id: item for item in spec.options}["endpoint"]
    assert endpoint.default == copernicus.DEFAULT_ENDPOINT
    assert endpoint.choices == (copernicus.DEFAULT_ENDPOINT, copernicus.AMS_ENDPOINT)
    fields = {field.id: field for field in spec.credentials}
    assert fields["access_key"].environment == ("CDSE_S3_ACCESS_KEY",)
    assert fields["secret_key"].environment == ("CDSE_S3_SECRET_KEY",)
    assert all(field.required and field.secret for field in fields.values())


@pytest.mark.parametrize(
    ("variable", "maximum_dn", "scale_factor"),
    (
        ("lai", 210, 1.0 / 30.0),
        ("rmse", 210, 1.0 / 30.0),
        ("length_before", 210, 1.0),
        ("length_after", 60, 1.0),
        ("nobs", 60, 1.0),
    ),
)
def test_copernicus_packing_matches_version_2_product_manual(
    variable: str, maximum_dn: int, scale_factor: float
) -> None:
    attrs = copernicus.VARIABLE_ATTRIBUTES[variable]
    assert attrs["_FillValue"] == attrs["missing_value"] == 255
    assert attrs["valid_range"].dtype.name == "uint8"
    assert attrs["valid_range"].tolist() == [0, maximum_dn]
    assert attrs["scale_factor"] == pytest.approx(scale_factor)
    assert attrs["add_offset"] == 0.0


def test_copernicus_qflag_metadata_describes_documented_byte_flags() -> None:
    attrs = copernicus.VARIABLE_ATTRIBUTES["qflag"]
    assert attrs["_FillValue"] == attrs["missing_value"] == 255
    assert attrs["flag_masks"].dtype.name == "uint8"
    assert attrs["flag_masks"].tolist() == [1, 2, 16, 96, 96, 96, 96, 128]
    assert attrs["flag_values"].tolist() == [1, 2, 16, 0, 32, 64, 96, 128]
    assert len(attrs["flag_meanings"].split()) == len(attrs["flag_values"])


def test_copernicus_plan_uses_public_cog_catalog_without_credentials(monkeypatch) -> None:
    day = date(2025, 1, 10)
    candidates = [product(day, revision=0), product(day, revision=6)]
    monkeypatch.setattr(copernicus, "read_catalog", lambda: candidates)
    item = request("copernicus_lai", ("lai", "qflag"), credentials={})
    plan = copernicus.CopernicusLaiProvider().plan(item)
    assert plan.period_count == 1
    assert plan.slice_count == 2
    assert plan.details["payload_source"] == "COG"
    assert plan.details["revisions"] == {"RT6": 1}


def test_copernicus_slices_range_read_cogs_for_netcdf(monkeypatch, tmp_path: Path) -> None:
    day = date(2025, 1, 10)
    selected = product(day, revision=6)
    catalog_calls = 0

    def fake_catalog():
        nonlocal catalog_calls
        catalog_calls += 1
        return [selected]

    monkeypatch.setattr(copernicus, "read_catalog", fake_catalog)
    monkeypatch.setattr(copernicus, "_make_s3_client", lambda *_args: object())
    keys = [
        "products/item/c_gls_LAI300-LAI-RT6_202501100000_GLOBE_OLCI_V2.0.1.tiff",
        "products/item/c_gls_LAI300-QFLAG-RT6_202501100000_GLOBE_OLCI_V2.0.1.tiff",
        "products/item/metadata.xml",
    ]
    monkeypatch.setattr(
        copernicus,
        "iter_product_objects",
        lambda *_args: iter({"Key": key, "Size": 100} for key in keys),
    )
    monkeypatch.setattr(copernicus.gdal, "require_commands", lambda *_args: None)
    calls: list[tuple[str, Path, dict[str, str], dict[str, object]]] = []

    def fake_translate(source, destination, *, env, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"cropped")
        calls.append((source, destination, env, kwargs))
        return destination

    monkeypatch.setattr(copernicus.gdal, "translate_to_cog", fake_translate)
    secret = "user-secret-never-log"
    item = request(
        "copernicus_lai",
        ("lai", "qflag"),
        credentials={"access_key": "user-access", "secret_key": secret},
        output_format=OutputFormat.NETCDF,
    )
    events: list[str] = []
    token = CancellationToken()
    provider = copernicus.CopernicusLaiProvider()
    assert provider.plan(item).slice_count == 2
    slices = list(
        provider.slices(
            item,
            tmp_path,
            lambda event: events.append(event.message),
            token,
        )
    )
    assert [item.variable for item in slices] == ["lai", "qflag"]
    assert all(source.startswith("/vsis3/eodata/") for source, _, _, _ in calls)
    assert all(path.suffix == ".tif" for _, path, _, _ in calls)
    assert all(env["AWS_SECRET_ACCESS_KEY"] == secret for _, _, env, _ in calls)
    assert all(kwargs["cancel"] is token for _, _, _, kwargs in calls)
    assert calls[0][3]["nodata"] == 255
    assert calls[0][3]["scale"] == pytest.approx(1.0 / 30.0)
    assert calls[0][3]["offset"] == 0.0
    assert secret not in " ".join(events)
    assert slices[0].attrs["scale_factor"] == pytest.approx(1.0 / 30.0)
    assert slices[0].attrs["valid_range"].tolist() == [0, 210]
    varying = {"source_product", "sensor", "product_version", "processing_revision"}
    assert all(varying.isdisjoint(raster.attrs) for raster in slices)
    assert all("Copernicus Land Monitoring Service" in raster.attrs["source"] for raster in slices)
    assert catalog_calls == 1


def test_vegetation_providers_reject_antimeridian_crop() -> None:
    bbox = BBox(170, -10, -170, 10)
    for provider, dataset in (
        (modis.ModisLaiProvider(), "modis_lai"),
        (copernicus.CopernicusLaiProvider(), "copernicus_lai"),
    ):
        item = DownloadRequest(
            dataset=dataset,
            bbox=bbox,
            variables=("lai",),
            time=TimeRange("2025-01-01", "2025-01-10"),
            output_format=OutputFormat.TIF,
            output_dir=Path("outputs"),
        )
        with pytest.raises(ConfigurationError, match="antimeridian"):
            provider.validate(item)


@pytest.mark.parametrize(
    "provider", (modis.ModisLaiProvider(), copernicus.CopernicusLaiProvider())
)
def test_vegetation_provider_specs_have_complete_metadata(provider: object) -> None:
    spec = provider.spec
    assert spec.id and spec.title and spec.summary
    assert spec.variables and all(variable.id and variable.title for variable in spec.variables)
    assert spec.source_url.startswith("https://")
    assert spec.citation and spec.license
