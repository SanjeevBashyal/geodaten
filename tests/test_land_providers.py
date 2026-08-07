from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from pathlib import Path
import sys
from types import ModuleType

import pytest
import requests

from geodaten.common.errors import AuthenticationError, ConfigurationError, DownloadError
from geodaten.common.models import BBox, DownloadRequest, OutputFormat, TimeRange
from geodaten.common.progress import CancellationToken
import geodaten.downloader.gimms_lai4g as gimms_module
import geodaten.downloader.glc_fcs30d as glc_module
import geodaten.downloader.srtm as srtm_module
from geodaten.downloader.gimms_lai4g import (
    GimmsLai4gProvider,
    HalfMonth,
    archive_filename as gimms_archive,
    parse_time_bound,
    selected_periods,
    source_filename as gimms_source,
)
from geodaten.downloader.glc_fcs30d import (
    GlcFcs30dProvider,
    archive_name,
    intersecting_tiles as glc_tiles,
    selected_years,
    source_spec,
)
from geodaten.downloader.soilgrids import (
    DEPTHS,
    SoilGridsProvider,
    remote_vrt,
    selected_layers,
)
from geodaten.downloader.srtm import (
    SrtmProvider,
    intersecting_tiles as srtm_tiles,
    tile_label as srtm_label,
    tile_url,
)


def request(
    dataset: str,
    variables: tuple[str, ...],
    *,
    bbox: BBox = BBox(79, 26, 80, 27),
    time: TimeRange | None = None,
    options: dict[str, object] | None = None,
    credentials: dict[str, str] | None = None,
    allow_partial: bool = False,
) -> DownloadRequest:
    return DownloadRequest(
        dataset=dataset,
        bbox=bbox,
        variables=variables,
        time=time,
        output_format=OutputFormat.TIF,
        output_dir=Path("outputs"),
        options=options or {},
        credentials=credentials or {},
        allow_partial=allow_partial,
    )


def test_soilgrids_resolves_variable_specific_default_depths() -> None:
    item = request("soilgrids", ("sand", "ocs"))
    layers = selected_layers(item)

    assert [layer for layer in layers if layer[0] == "sand"] == [
        ("sand", depth, "mean") for depth in DEPTHS
    ]
    assert [layer for layer in layers if layer[0] == "ocs"] == [
        ("ocs", "0-30cm", "mean")
    ]
    assert SoilGridsProvider().plan(item).slice_count == 7


def test_soilgrids_expands_depth_and_statistic_options() -> None:
    item = request(
        "soilgrids",
        ("clay",),
        options={"depths": ("0-5cm", "5-15cm"), "statistics": ("mean", "Q0.5")},
    )
    assert selected_layers(item) == (
        ("clay", "0-5cm", "mean"),
        ("clay", "5-15cm", "mean"),
        ("clay", "0-5cm", "Q0.5"),
        ("clay", "5-15cm", "Q0.5"),
    )
    SoilGridsProvider().validate(item)


def test_soilgrids_rejects_unavailable_ocs_depth() -> None:
    item = request("soilgrids", ("ocs",), options={"depths": ("0-5cm",)})
    with pytest.raises(ConfigurationError, match="does not provide"):
        SoilGridsProvider().validate(item)


def test_soilgrids_vrt_uses_public_range_read_source() -> None:
    source = remote_vrt("soc", "0-5cm", "Q0.5")
    assert source.startswith("/vsicurl?")
    assert source.endswith(
        "https://files.isric.org/soilgrids/latest/data/soc/soc_0-5cm_Q0.5.vrt"
    )


def test_srtm_tile_math_honours_exact_degree_boundaries() -> None:
    assert srtm_tiles(BBox(79, 26, 80, 27)) == ("N26E079",)
    assert srtm_label(-1, -81) == "S01W081"
    assert srtm_tiles(BBox(-80.2, -1, -78, 1)) == (
        "S01W081",
        "S01W080",
        "S01W079",
        "N00W081",
        "N00W080",
        "N00W079",
    )


def test_srtm_plan_and_url_are_computed_without_network() -> None:
    item = request("srtm", ("elevation",), bbox=BBox(7.2, 50.1, 9.0, 51.0))
    plan = SrtmProvider().plan(item)
    assert plan.details["tiles"] == ("N50E007", "N50E008")
    assert plan.details["tile_count"] == 2
    assert tile_url("N50E007").endswith(
        "/N50E007.SRTMGL1.hgt/N50E007.SRTMGL1.hgt.zip"
    )


def test_srtm_spec_exposes_only_user_owned_auth_inputs() -> None:
    fields = {field.id: field for field in SrtmProvider.spec.credentials}
    assert set(fields) == {
        "earthdata_username",
        "earthdata_password",
        "earthdata_token",
    }
    assert fields["earthdata_password"].secret and fields["earthdata_token"].secret
    assert fields["earthdata_username"].environment == ("EARTHDATA_USERNAME",)
    assert all(field.required is False for field in fields.values())


def test_srtm_token_takes_precedence_over_incomplete_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    provider = SrtmProvider()
    provider.validate(
        request(
            "srtm",
            ("elevation",),
            credentials={"earthdata_token": "user-token", "earthdata_username": "user"},
        )
    )
    with pytest.raises(AuthenticationError, match="supplied together"):
        provider.validate(
            request(
                "srtm",
                ("elevation",),
                credentials={"earthdata_username": "user"},
            )
        )


@pytest.mark.parametrize(
    ("status", "expected", "error"),
    (
        (200, True, None),
        (404, False, None),
        (401, None, AuthenticationError),
        (503, None, DownloadError),
    ),
)
def test_srtm_preflight_distinguishes_ocean_auth_and_server_failures(
    status: int,
    expected: bool | None,
    error: type[Exception] | None,
) -> None:
    class Response:
        status_code = status

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def close() -> None:
            return None

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(response=self)  # type: ignore[arg-type]

    class Session:
        @staticmethod
        def head(*_args: object, **_kwargs: object) -> Response:
            return Response()

    if error:
        with pytest.raises(error):
            srtm_module._tile_available(Session(), "N00E000")
    else:
        assert srtm_module._tile_available(Session(), "N00E000") is expected


def test_srtm_preflight_surfaces_transport_failures() -> None:
    class Session:
        @staticmethod
        def head(*_args: object, **_kwargs: object) -> None:
            raise requests.ConnectionError("offline")

    with pytest.raises(DownloadError, match="N00E000"):
        srtm_module._tile_available(Session(), "N00E000")


def test_srtm_slices_skip_only_confirmed_404_tiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    downloaded: list[str] = []
    mosaicked: list[Path] = []
    mosaic_options: dict[str, object] = {}
    events: list[str] = []

    @contextmanager
    def session(_request: DownloadRequest):
        yield object()

    def download(url: str, destination: Path, **_kwargs: object) -> Path:
        downloaded.append(url)
        destination.write_bytes(b"zip")
        return destination

    def extract(
        _archive: Path, destination: Path, **_kwargs: object
    ) -> Path:
        destination.write_bytes(b"hgt")
        return destination

    def mosaic(
        sources: tuple[Path, ...] | list[Path],
        destination: Path,
        **_kwargs: object,
    ) -> Path:
        mosaicked.extend(sources)
        mosaic_options.update(_kwargs)
        destination.write_bytes(b"raster")
        return destination

    monkeypatch.setattr(srtm_module, "_earthdata_session", session)
    monkeypatch.setattr(
        srtm_module,
        "_tile_available",
        lambda _session, label: label == "N26E080",
    )
    monkeypatch.setattr(srtm_module, "stream_download", download)
    monkeypatch.setattr(srtm_module, "_extract_hgt", extract)
    monkeypatch.setattr(srtm_module, "mosaic_to_cog", mosaic)

    item = request("srtm", ("elevation",), bbox=BBox(79, 26, 81, 27))
    iterator = SrtmProvider().slices(
        item,
        tmp_path,
        lambda event: events.append(event.message),
        CancellationToken(),
    )
    assert next(iterator).variable == "elevation"
    iterator.close()

    assert len(downloaded) == 1 and "N26E080" in downloaded[0]
    assert [source.name for source in mosaicked] == ["N26E080.hgt"]
    assert mosaic_options["nodata"] == -32768
    assert isinstance(mosaic_options["cancel"], CancellationToken)
    assert any("N26E079" in message and "no land" in message for message in events)


def test_glc_selects_only_observation_epochs() -> None:
    assert selected_years("1986", "2001") == (1990, 1995, 2000, 2001)
    assert selected_years("2000-06-01", "2002-12-31") == (2000, 2001, 2002)


def test_glc_western_tiles_and_archives() -> None:
    assert glc_tiles(BBox(-10, -5, -4, 1)) == (
        (-10, 5),
        (-10, 0),
        (-5, 5),
        (-5, 0),
    )
    assert archive_name(-5).endswith("W5-W10.zip")
    assert archive_name(-10).endswith("W5-W10.zip")
    assert archive_name(-125).endswith("W125-W130.zip")
    assert archive_name(-130).endswith("W125-W130.zip")
    assert archive_name(5).endswith("E0-E5.zip")


def test_glc_source_bands_match_epoch_family() -> None:
    historical = source_spec(1995, -5, 10)
    annual = source_spec(2022, -5, 10)
    assert historical[0] == "GLC_FCS30D_19852000_W5N10_5years_V1.1.tif"
    assert historical[1:] == (3, 3)
    assert annual[0] == "GLC_FCS30D_20002022_W5N10_Annual_V1.1.tif"
    assert annual[1:] == (23, 23)


def test_glc_plan_lists_dynamic_western_archive() -> None:
    item = request(
        "glc_fcs30d",
        ("land_cover",),
        bbox=BBox(-7, 40, -4, 42),
        time=TimeRange("1995", "2000"),
    )
    plan = GlcFcs30dProvider().plan(item)
    assert plan.period_count == plan.slice_count == 2
    assert plan.details["archives"] == ("GLC_FCS30D_19852022maps_W5-W10.zip",)
    assert plan.details["source_tile_count"] == 4


def test_glc_missing_ocean_member_is_a_warned_nodata_gap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    item = request(
        "glc_fcs30d",
        ("land_cover",),
        bbox=BBox(0, 0, 10, 5),
        time=TimeRange("2000", "2000"),
    )
    present, _band, _count = source_spec(2000, 0, 5)
    archive = archive_name(0)

    class Info:
        filename = present

        @staticmethod
        def is_dir() -> bool:
            return False

    class Remote:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> "Remote":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def infolist() -> list[Info]:
            return [Info()]

    class Session:
        @staticmethod
        def close() -> None:
            return None

    used_sources: list[Path] = []
    mosaic_options: dict[str, object] = {}
    events: list[str] = []

    def extract(
        _remote: object,
        _member: str,
        destination: Path,
        **_kwargs: object,
    ) -> Path:
        destination.write_bytes(b"source")
        return destination

    def mosaic(sources: tuple[Path, ...], destination: Path, **_kwargs: object) -> Path:
        used_sources.extend(sources)
        mosaic_options.update(_kwargs)
        destination.write_bytes(b"slice")
        return destination

    fake_remotezip = ModuleType("remotezip")
    fake_remotezip.RemoteZip = Remote  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "remotezip", fake_remotezip)
    monkeypatch.setattr(glc_module, "require_module", lambda *_args: None)
    monkeypatch.setattr(glc_module, "make_session", Session)
    monkeypatch.setattr(
        glc_module,
        "_record_files",
        lambda _session: ({}, {archive: "https://example.invalid/archive.zip"}),
    )
    monkeypatch.setattr(glc_module, "_extract_member", extract)
    monkeypatch.setattr(glc_module, "mosaic_to_cog", mosaic)

    iterator = GlcFcs30dProvider().slices(
        item,
        tmp_path,
        lambda event: events.append(event.message),
        CancellationToken(),
    )
    assert next(iterator).time == date(2000, 1, 1)
    iterator.close()

    assert [source.name for source in used_sources] == [present]
    assert mosaic_options["nodata"] == 0
    assert isinstance(mosaic_options["cancel"], CancellationToken)
    assert any("no land tile" in message for message in events)


def test_glc_errors_when_every_source_member_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    item = request(
        "glc_fcs30d",
        ("land_cover",),
        bbox=BBox(0, 0, 5, 5),
        time=TimeRange("2000", "2000"),
    )
    archive = archive_name(0)

    class Remote:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> "Remote":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def infolist() -> list[object]:
            return []

    class Session:
        @staticmethod
        def close() -> None:
            return None

    fake_remotezip = ModuleType("remotezip")
    fake_remotezip.RemoteZip = Remote  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "remotezip", fake_remotezip)
    monkeypatch.setattr(glc_module, "require_module", lambda *_args: None)
    monkeypatch.setattr(glc_module, "make_session", Session)
    monkeypatch.setattr(
        glc_module,
        "_record_files",
        lambda _session: ({}, {archive: "https://example.invalid/archive.zip"}),
    )

    with pytest.raises(DownloadError, match="No GLC_FCS30D source data"):
        list(
            GlcFcs30dProvider().slices(
                item, tmp_path, lambda _event: None, CancellationToken()
            )
        )


def test_gimms_half_month_time_selection() -> None:
    assert parse_time_bound("2001-02-15", lower=True, last_year=2020) == HalfMonth(
        2001, 2, 1
    )
    assert parse_time_bound("2001-02-16", lower=False, last_year=2020) == HalfMonth(
        2001, 2, 2
    )
    periods = selected_periods("2001-02-15", "2001-03-01", last_year=2020)
    assert periods == (
        HalfMonth(2001, 2, 1),
        HalfMonth(2001, 2, 2),
        HalfMonth(2001, 3, 1),
    )
    assert [period.nominal_date for period in periods] == [
        date(2001, 2, 1),
        date(2001, 2, 16),
        date(2001, 3, 1),
    ]


def test_gimms_variant_archive_and_member_names() -> None:
    period = HalfMonth(2015, 12, 2)
    assert gimms_archive("AVHRR_solely", period).endswith("2011_2015.zip")
    assert gimms_source("AVHRR_solely", period) == (
        "GIMMS_LAI4g(AVHRR)_V1.2_20151202.tif"
    )
    assert gimms_source("AVHRR_MODIS_consolidated", period) == (
        "GIMMS_LAI4g_V1.2_20151202.tif"
    )


def test_gimms_consolidated_qc_is_strictly_limited_to_2003() -> None:
    item = request(
        "gimms_lai4g",
        ("lai", "qc"),
        time=TimeRange("2020-12-01", "2020-12-31"),
    )
    with pytest.raises(ConfigurationError, match="QC band is available only through 2003"):
        GimmsLai4gProvider().plan(item)


def test_gimms_partial_plan_counts_only_supported_qc_slices() -> None:
    item = request(
        "gimms_lai4g",
        ("lai", "qc"),
        time=TimeRange("2020-12-01", "2020-12-31"),
        allow_partial=True,
    )
    plan = GimmsLai4gProvider().plan(item)
    assert plan.period_count == 2
    assert plan.slice_count == 2
    assert plan.details["skipped_qc_slices"] == 2
    assert plan.warnings and "Skipping 2" in plan.warnings[0]
    assert plan.details["first_period"] == "2020-12-H1"
    assert plan.details["last_period"] == "2020-12-H2"


def test_gimms_qc_cutoff_and_avhrr_only_coverage() -> None:
    cutoff = request(
        "gimms_lai4g",
        ("qc",),
        time=TimeRange("2003-12-16", "2004-01-01"),
        allow_partial=True,
    )
    cutoff_plan = GimmsLai4gProvider().plan(cutoff)
    assert cutoff_plan.period_count == 2
    assert cutoff_plan.slice_count == 1
    assert cutoff_plan.details["source_member_count"] == 1

    avhrr = request(
        "gimms_lai4g",
        ("qc",),
        time=TimeRange("2015-12-01", "2015-12-31"),
        options={"variant": "AVHRR_solely"},
    )
    avhrr_plan = GimmsLai4gProvider().plan(avhrr)
    assert avhrr_plan.slice_count == 2
    assert not avhrr_plan.warnings


@pytest.mark.parametrize(
    ("period", "allow_partial", "expected_variables", "expected_bands"),
    (
        (HalfMonth(2001, 1, 1), False, ("lai", "qc"), (1, 2)),
        (HalfMonth(2004, 1, 1), True, ("lai",), (1,)),
    ),
)
def test_gimms_slices_respect_qc_coverage_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    period: HalfMonth,
    allow_partial: bool,
    expected_variables: tuple[str, ...],
    expected_bands: tuple[int, ...],
) -> None:
    archive = gimms_archive("AVHRR_MODIS_consolidated", period)
    member = gimms_source("AVHRR_MODIS_consolidated", period)

    class Info:
        filename = member
        file_size = 1

        @staticmethod
        def is_dir() -> bool:
            return False

    class Remote:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> "Remote":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def infolist() -> list[Info]:
            return [Info()]

    class Session:
        @staticmethod
        def close() -> None:
            return None

    bands: list[int] = []
    translate_options: list[dict[str, object]] = []

    def extract(
        _remote: object,
        _member: str,
        destination: Path,
        **_kwargs: object,
    ) -> Path:
        destination.write_bytes(b"source")
        return destination

    def translate(_source: Path, destination: Path, **kwargs: object) -> Path:
        bands.append(int(kwargs["band"]))
        translate_options.append(kwargs)
        destination.write_bytes(b"slice")
        return destination

    fake_remotezip = ModuleType("remotezip")
    fake_remotezip.RemoteZip = Remote  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "remotezip", fake_remotezip)
    monkeypatch.setattr(gimms_module, "require_module", lambda *_args: None)
    monkeypatch.setattr(gimms_module, "make_session", Session)
    monkeypatch.setattr(
        gimms_module,
        "_record_files",
        lambda _session: ({}, {archive: "https://example.invalid/archive.zip"}),
    )
    monkeypatch.setattr(gimms_module, "_extract_member", extract)
    monkeypatch.setattr(gimms_module, "translate_to_cog", translate)

    item = request(
        "gimms_lai4g",
        ("lai", "qc"),
        time=TimeRange(period.nominal_date.isoformat(), period.nominal_date.isoformat()),
        allow_partial=allow_partial,
    )
    iterator = GimmsLai4gProvider().slices(
        item, tmp_path, lambda _event: None, CancellationToken()
    )
    items = tuple(iterator)

    assert tuple(item.variable for item in items) == expected_variables
    assert all(item.time == period.nominal_date for item in items)
    assert items[0].attrs["scale_factor"] == 0.001
    assert tuple(bands) == expected_bands
    assert all(options["nodata"] == 65535 for options in translate_options)
    assert all(
        isinstance(options["cancel"], CancellationToken)
        for options in translate_options
    )


@pytest.mark.parametrize(
    "provider",
    (SoilGridsProvider(), SrtmProvider(), GlcFcs30dProvider(), GimmsLai4gProvider()),
)
def test_land_provider_specs_have_complete_source_metadata(provider: object) -> None:
    spec = provider.spec
    assert spec.id and spec.title and spec.summary
    assert spec.variables and all(variable.id and variable.title for variable in spec.variables)
    assert spec.source_url.startswith("https://")
    assert spec.citation and spec.license
