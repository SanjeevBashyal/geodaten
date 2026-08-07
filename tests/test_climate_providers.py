from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from geodaten.common.errors import AuthenticationError, ConfigurationError, DownloadError, OutputError
from geodaten.common.models import BBox, DownloadRequest, OutputFormat, TimeRange
from geodaten.downloader.ifs_fesom import (
    IfsFesomProvider,
    count_jobs,
    describe_fields,
    expand_times,
    iter_jobs,
    resolve_polytope_credentials,
)
from geodaten.downloader.nex_gddp_cmip6 import (
    NexAsset,
    NexGddpProvider,
    NexSubsetJob,
    _catalog_refs,
    ncss_query_url,
    parse_asset_catalog,
    subset_jobs,
)


def request(
    dataset: str,
    variables: tuple[str, ...],
    *,
    start: str,
    end: str,
    options: dict | None = None,
    credentials: dict | None = None,
    bbox: BBox | None = None,
    allow_partial: bool = False,
) -> DownloadRequest:
    return DownloadRequest(
        dataset=dataset,
        bbox=bbox or BBox(79, 26, 89, 31),
        variables=variables,
        output_format=OutputFormat.TIF,
        output_dir=Path("output"),
        time=TimeRange(start, end),
        options=options or {},
        credentials=credentials or {},
        allow_partial=allow_partial,
    )


def catalog(*body: str) -> bytes:
    return (
        '<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0" '
        'xmlns:xlink="http://www.w3.org/1999/xlink"><dataset name="root">'
        + "".join(body)
        + "</dataset></catalog>"
    ).encode()


def ref(title: str, href: str) -> str:
    return f'<catalogRef xlink:href="{href}" xlink:title="{title}" name="{title}" />'


def test_nex_catalog_discovers_actual_grid_and_file_size() -> None:
    xml = catalog(
        '<dataset name="pr_day_ACCESS-CM2_historical_r1i1p1f1_gr1_2014_v2.0.nc" '
        'urlPath="AMES/NEX/file-2014.nc"><dataSize units="Mbytes">1.5</dataSize></dataset>'
    )
    assets = parse_asset_catalog(
        xml,
        model="ACCESS-CM2",
        scenario="historical",
        variant="r1i1p1f1",
        variable="pr",
    )
    assert [(item.year, item.grid, item.version, item.size) for item in assets] == [
        (2014, "gr1", "v2.0", 1_500_000)
    ]
    # A few THREDDS model folders encode the grid (for example
    # GFDL-CM4_gr2), while the actual filename correctly retains GFDL-CM4.
    grid_folder = parse_asset_catalog(
        catalog(
            '<dataset name="pr_day_GFDL-CM4_historical_r1i1p1f1_gr2_1950_v2.0.nc" '
            'urlPath="AMES/NEX/gfdl.nc" />'
        ),
        model="GFDL-CM4_gr2",
        scenario="historical",
        variant="r1i1p1f1",
        variable="pr",
    )
    assert grid_folder[0].grid == "gr2"


def test_nex_catalog_refs_are_resolved_relative_to_each_catalog() -> None:
    refs = _catalog_refs(
        catalog(ref("historical", "historical/catalog.xml")),
        "https://example.test/thredds/catalog/model/catalog.xml",
    )
    assert refs == {
        "historical": "https://example.test/thredds/catalog/model/historical/catalog.xml"
    }


def test_nex_jobs_intersect_short_year_boundaries_exactly() -> None:
    assets = tuple(
        NexAsset("M", "historical", "r1", "pr", year, "gn", f"data/{year}.nc")
        for year in (2013, 2014, 2015)
    )
    jobs = subset_jobs(assets, date(2014, 12, 31), date(2015, 1, 2))
    assert [(job.asset.year, job.start, job.end) for job in jobs] == [
        (2014, date(2014, 12, 31), date(2014, 12, 31)),
        (2015, date(2015, 1, 1), date(2015, 1, 2)),
    ]


def test_nex_jobs_require_every_applicable_annual_asset() -> None:
    assets = (NexAsset("M", "historical", "r1", "pr", 2014, "gn", "data/2014.nc"),)
    with pytest.raises(DownloadError, match="2013"):
        subset_jobs(assets, date(2013, 12, 31), date(2014, 1, 1))
    jobs = subset_jobs(
        assets,
        date(2013, 12, 31),
        date(2014, 1, 1),
        allow_partial=True,
    )
    assert [(job.start, job.end) for job in jobs] == [
        (date(2014, 1, 1), date(2014, 1, 1))
    ]


def test_nex_ncss_request_uses_exact_extent_dates_and_variable() -> None:
    req = request("nex_gddp_cmip6", ("pr",), start="2014-12-31", end="2014-12-31")
    asset = NexAsset("M", "historical", "r1", "pr", 2014, "gr1", "AMES/NEX/a.nc")
    url = ncss_query_url(NexSubsetJob(asset, date(2014, 12, 31), date(2014, 12, 31)), req)
    query = parse_qs(urlparse(url).query)
    assert query["var"] == ["pr"]
    assert query["time_start"] == ["2014-12-31T12:00:00Z"]
    assert query["time_end"] == ["2014-12-31T12:00:00Z"]
    assert {key: query[key][0] for key in ("west", "south", "east", "north")} == {
        "west": "79",
        "south": "26",
        "east": "89",
        "north": "31",
    }


def test_nex_plan_walks_dynamic_catalogs_without_guessing_grid(monkeypatch) -> None:
    root = "https://example.test/thredds/catalog/root/catalog.xml"
    provider = NexGddpProvider(catalog_root=root)
    documents = {
        root: catalog(ref("ACCESS-CM2", "ACCESS-CM2/catalog.xml")),
        "https://example.test/thredds/catalog/root/ACCESS-CM2/catalog.xml": catalog(
            ref("historical", "historical/catalog.xml"), ref("ssp370", "ssp370/catalog.xml")
        ),
        "https://example.test/thredds/catalog/root/ACCESS-CM2/historical/catalog.xml": catalog(
            ref("r1i1p1f1", "r1i1p1f1/catalog.xml")
        ),
        "https://example.test/thredds/catalog/root/ACCESS-CM2/ssp370/catalog.xml": catalog(
            ref("r1i1p1f1", "r1i1p1f1/catalog.xml")
        ),
        "https://example.test/thredds/catalog/root/ACCESS-CM2/historical/r1i1p1f1/catalog.xml": catalog(
            ref("pr", "pr/catalog.xml")
        ),
        "https://example.test/thredds/catalog/root/ACCESS-CM2/ssp370/r1i1p1f1/catalog.xml": catalog(
            ref("pr", "pr/catalog.xml")
        ),
        "https://example.test/thredds/catalog/root/ACCESS-CM2/historical/r1i1p1f1/pr/catalog.xml": catalog(
            '<dataset name="pr_day_ACCESS-CM2_historical_r1i1p1f1_gn_2014_v2.0.nc" '
            'urlPath="AMES/NEX/historical-2014.nc" />'
        ),
        "https://example.test/thredds/catalog/root/ACCESS-CM2/ssp370/r1i1p1f1/pr/catalog.xml": catalog(
            '<dataset name="pr_day_ACCESS-CM2_ssp370_r1i1p1f1_gr1_2015_v2.0.nc" '
            'urlPath="AMES/NEX/ssp370-2015.nc" />'
        ),
    }

    class Session:
        def close(self) -> None:
            pass

    monkeypatch.setattr("geodaten.downloader.nex_gddp_cmip6.make_session", Session)
    monkeypatch.setattr(provider, "_fetch_catalog", lambda url, _session: documents[url])
    req = request(
        "nex_gddp_cmip6",
        ("pr",),
        start="2014-12-31",
        end="2015-01-02",
        options={
            "model": ("ACCESS-CM2",),
            "scenario": ("historical", "ssp370"),
            "variant": ("r1i1p1f1",),
        },
    )
    plan = provider.plan(req)
    assert plan.period_count == 2
    assert plan.slice_count == 3
    assert plan.details["grids"] == ["gn", "gr1"]


def test_ifs_time_range_and_daily_jobs_remain_lazy() -> None:
    req = request(
        "ifs_fesom",
        ("2t", "avg_tprate"),
        start="2014-12-31",
        end="2015-01-02",
        options={
            "experiment": ("historical", "projection"),
            "times": "0000/1200",
        },
    )
    iterator = iter_jobs(req)
    assert iter(iterator) is iterator
    jobs = list(iterator)
    assert [job.period for job in jobs] == [
        date(2014, 12, 31),
        date(2015, 1, 1),
        date(2015, 1, 2),
    ]
    assert jobs[0].request["param"] == "167/235055"
    assert jobs[0].request["area"] == "31/79/26/89"
    assert jobs[0].request["time"] == "0000/1200"
    assert jobs[0].request["model"] == "IFS-FESOM"
    assert jobs[0].request["realization"] == "1"
    assert jobs[1].request["experiment"] == "SSP3-7.0"
    assert count_jobs(req) == 3
    plan = IfsFesomProvider().plan(req)
    assert plan.period_count == 3
    assert plan.slice_count == 12


def test_ifs_monthly_jobs_and_pressure_levels() -> None:
    req = request(
        "ifs_fesom",
        ("t",),
        start="1990-01-15",
        end="1990-02-03",
        options={
            "experiment": ("historical",),
            "frequency": "clmn",
            "levtype": "pl",
            "levels": ("500", "850"),
        },
    )
    jobs = list(iter_jobs(req))
    assert [job.period for job in jobs] == [date(1990, 1, 1), date(1990, 2, 1)]
    assert jobs[0].request["stream"] == "clmn"
    assert jobs[0].request["levelist"] == "500/850"
    assert jobs[0].request["time"] == "0000"


def test_ifs_time_expression_expansion() -> None:
    assert expand_times("0000/0100/to/0300") == ("0000", "0100", "0200", "0300")
    assert expand_times("0000/to/0600/by/0300") == ("0000", "0300", "0600")


def test_ifs_target_shape_does_not_add_float_rounding_cells() -> None:
    req = request(
        "ifs_fesom",
        ("2t",),
        start="1990-01-01",
        end="1990-01-01",
        bbox=BBox(7, 50, 7.1, 50.1),
        options={"experiment": ("historical",), "target_resolution": 0.1},
    )
    assert IfsFesomProvider().plan(req).details["target_shape"] == [1, 1]


def test_polytope_credentials_are_only_user_sources(tmp_path: Path) -> None:
    rc = tmp_path / ".polytopeapirc"
    rc.write_text(json.dumps({"user_email": "rc@example.test", "user_key": "rc-key"}))
    assert resolve_polytope_credentials({}, environ={}, rc_path=rc) == {
        "user_email": "rc@example.test",
        "user_key": "rc-key",
    }
    assert resolve_polytope_credentials(
        {"user_key": "request-key"},
        environ={"POLYTOPE_USER_EMAIL": "env@example.test", "POLYTOPE_USER_KEY": "env-key"},
        rc_path=rc,
    ) == {"user_email": "env@example.test", "user_key": "request-key"}
    with pytest.raises(AuthenticationError, match="Provide a Polytope key"):
        resolve_polytope_credentials({}, environ={}, rc_path=tmp_path / "missing")


def test_ifs_advertises_only_supported_level_types() -> None:
    req = request(
        "ifs_fesom",
        ("2t",),
        start="1990-01-01",
        end="1990-01-01",
        options={"experiment": ("historical",), "levtype": "o2d"},
    )
    with pytest.raises(ConfigurationError, match="Invalid levtype"):
        IfsFesomProvider().validate(req)
    levtype = next(option for option in IfsFesomProvider.spec.options if option.id == "levtype")
    assert levtype.choices == ("sfc", "pl", "hl")


class Field:
    def __init__(self, **metadata: object) -> None:
        self._metadata = metadata

    def metadata(self, key: str) -> object | None:
        return self._metadata.get(key)


@pytest.mark.parametrize(
    ("frequency", "expected"),
    (
        (
            "clte",
            {
                "z": "129",
                "t": "130",
                "u": "131",
                "v": "132",
                "q": "133",
                "10u": "165",
                "10v": "166",
                "2t": "167",
                "2d": "168",
                "avg_sdswrf": "235035",
                "avg_sdlwrf": "235036",
                "avg_tprate": "235055",
            },
        ),
        (
            "clmn",
            {
                "z": "235129",
                "t": "235130",
                "u": "235131",
                "v": "235132",
                "q": "235133",
                "10u": "235165",
                "10v": "235166",
                "2t": "228004",
                "2d": "235168",
                "avg_sdswrf": "235035",
                "avg_sdlwrf": "235036",
                "avg_tprate": "235055",
            },
        ),
    ),
)
def test_ifs_payload_ids_and_field_descriptors_are_frequency_exact(
    frequency: str, expected: dict[str, str]
) -> None:
    groups = (
        ("pl", ("z", "t", "u", "v", "q"), ("500",), "isobaricInhPa"),
        ("hl", ("u", "v"), ("100",), "hybrid"),
        (
            "sfc",
            (
                "10u",
                "10v",
                "2t",
                "2d",
                "avg_sdswrf",
                "avg_sdlwrf",
                "avg_tprate",
            ),
            (),
            "surface",
        ),
    )
    for levtype, variables, levels, type_of_level in groups:
        req = request(
            "ifs_fesom",
            variables,
            start="1990-01-01",
            end="1990-01-01",
            options={
                "experiment": ("historical",),
                "frequency": frequency,
                "levtype": levtype,
                "levels": levels,
                "times": "0000",
            },
        )
        IfsFesomProvider().validate(req)
        job = next(iter_jobs(req))
        assert job.request["stream"] == frequency
        assert job.request["param"] == "/".join(expected[name] for name in variables)
        fields = [
            Field(
                paramId=expected[name],
                validityDate=19900101,
                validityTime=0,
                typeOfLevel=(
                    "heightAboveGround"
                    if name in {"10u", "10v", "2t", "2d"}
                    else type_of_level
                ),
                level=(
                    {"10u": 10, "10v": 10, "2t": 2, "2d": 2}.get(name, 0)
                    if not levels
                    else levels[0]
                ),
            )
            for name in variables
        ]
        descriptors = describe_fields(fields, job, allow_partial=False)
        assert [item.variable for item in descriptors] == list(variables)
        assert [item.attrs["param_id"] for item in descriptors] == [
            expected[name] for name in variables
        ]
        assert {item.attrs["type_of_level"] for item in descriptors} == {levtype}


def test_ifs_field_validation_rejects_relabeling_and_unexpected_fields() -> None:
    req = request(
        "ifs_fesom",
        ("2t",),
        start="1990-01-01",
        end="1990-01-01",
        options={"experiment": ("historical",), "times": "0000"},
    )
    job = next(iter_jobs(req))
    valid = {
        "paramId": "167",
        "validityDate": 19900101,
        "validityTime": 0,
        "typeOfLevel": "heightAboveGround",
        "level": 2,
    }
    with pytest.raises(DownloadError, match="unexpected CLTE time"):
        describe_fields([Field(**{**valid, "validityDate": 19900102})], job, allow_partial=True)
    with pytest.raises(DownloadError, match="missing validity"):
        describe_fields(
            [Field(paramId="167", typeOfLevel="heightAboveGround", level=2)],
            job,
            allow_partial=True,
        )
    with pytest.raises(DownloadError, match="unexpected parameter"):
        describe_fields([Field(**{**valid, "paramId": "999"})], job, allow_partial=True)
    with pytest.raises(DownloadError, match="levtype=pl"):
        describe_fields(
            [Field(**{**valid, "typeOfLevel": "isobaricInhPa", "level": 500})],
            job,
            allow_partial=True,
        )
    with pytest.raises(OutputError, match="duplicate field"):
        describe_fields([Field(**valid), Field(**valid)], job, allow_partial=True)

    monthly = request(
        "ifs_fesom",
        ("2t",),
        start="1990-01",
        end="1990-01",
        options={"experiment": ("historical",), "frequency": "clmn"},
    )
    monthly_job = next(iter_jobs(monthly))
    monthly_field = {
        "paramId": "228004",
        "validityDate": 19900101,
        "validityTime": 0,
        "typeOfLevel": "heightAboveGround",
        "level": 2,
    }
    with pytest.raises(DownloadError, match="unexpected CLMN month"):
        describe_fields(
            [Field(**{**monthly_field, "validityDate": 19900201})],
            monthly_job,
            allow_partial=True,
        )
    with pytest.raises(DownloadError, match="unexpected CLMN time"):
        describe_fields(
            [Field(**{**monthly_field, "validityTime": 100})],
            monthly_job,
            allow_partial=True,
        )


def test_ifs_allow_partial_skips_only_missing_requested_fields() -> None:
    req = request(
        "ifs_fesom",
        ("2t", "2d"),
        start="1990-01-01",
        end="1990-01-01",
        options={"experiment": ("historical",), "times": "0000"},
    )
    job = next(iter_jobs(req))
    field = Field(
        paramId="167",
        validityDate=19900101,
        validityTime=0,
        typeOfLevel="heightAboveGround",
        level=2,
    )
    with pytest.raises(DownloadError, match="incomplete"):
        describe_fields([field], job, allow_partial=False)
    descriptors = describe_fields([field], job, allow_partial=True)
    assert [item.variable for item in descriptors] == ["2t"]


def test_ifs_temporal_coverage_is_strict_or_explicitly_partial() -> None:
    strict = request(
        "ifs_fesom",
        ("2t",),
        start="2014-12-31",
        end="2015-01-02",
        options={"experiment": ("historical",), "times": "0000"},
    )
    with pytest.raises(ConfigurationError, match="2015-01-01..2015-01-02"):
        IfsFesomProvider().plan(strict)
    partial = request(
        "ifs_fesom",
        ("2t",),
        start="2014-12-31",
        end="2015-01-02",
        options={"experiment": ("historical",), "times": "0000"},
        allow_partial=True,
    )
    plan = IfsFesomProvider().plan(partial)
    assert plan.period_count == plan.slice_count == 1
    assert plan.warnings == (
        "Omitting unsupported IFS-FESOM interval 2015-01-01..2015-01-02.",
    )
    assert [job.period for job in iter_jobs(partial)] == [date(2014, 12, 31)]


def test_ifs_monthly_coverage_compares_whole_month_periods() -> None:
    partial = request(
        "ifs_fesom",
        ("2t",),
        start="2014-12-15",
        end="2015-01-02",
        options={
            "experiment": ("historical",),
            "frequency": "clmn",
        },
        allow_partial=True,
    )
    plan = IfsFesomProvider().plan(partial)
    assert plan.period_count == 1
    assert plan.warnings == (
        "Omitting unsupported IFS-FESOM interval 2015-01..2015-01.",
    )


def test_ifs_control_remains_an_independent_experiment() -> None:
    strict = request(
        "ifs_fesom",
        ("2t",),
        start="2000-01-01",
        end="2000-01-02",
        options={
            "experiment": ("historical", "control"),
            "times": "0000",
        },
    )
    with pytest.raises(ConfigurationError, match="control 2000-01-01..2000-01-02"):
        IfsFesomProvider().plan(strict)

    partial = request(
        "ifs_fesom",
        ("2t",),
        start="2000-01-01",
        end="2000-01-02",
        options={
            "experiment": ("historical", "control"),
            "times": "0000",
        },
        allow_partial=True,
    )
    plan = IfsFesomProvider().plan(partial)
    assert plan.period_count == plan.slice_count == 2
    assert plan.warnings == (
        "Omitting unsupported IFS-FESOM interval control 2000-01-01..2000-01-02.",
    )


def test_ifs_validates_published_pressure_and_hybrid_levels() -> None:
    invalid_pressure = request(
        "ifs_fesom",
        ("t",),
        start="1990-01-01",
        end="1990-01-01",
        options={"experiment": ("historical",), "levtype": "pl", "levels": ("750",)},
    )
    with pytest.raises(ConfigurationError, match="Unsupported pressure"):
        IfsFesomProvider().validate(invalid_pressure)
    invalid_hybrid = request(
        "ifs_fesom",
        ("u",),
        start="1990-01-01",
        end="1990-01-01",
        options={"experiment": ("historical",), "levtype": "hl", "levels": ("99",)},
    )
    with pytest.raises(ConfigurationError, match="only level 100"):
        IfsFesomProvider().validate(invalid_hybrid)


def test_polytope_credentials_preserve_exact_user_values_and_ignore_none(tmp_path: Path) -> None:
    assert resolve_polytope_credentials(
        {"user_email": " user@example.test ", "user_key": " key with spaces "},
        environ={},
        rc_path=tmp_path / "missing",
    ) == {
        "user_email": " user@example.test ",
        "user_key": " key with spaces ",
    }
    assert resolve_polytope_credentials(
        {"user_email": None, "user_key": None},  # type: ignore[dict-item]
        environ={"POLYTOPE_USER_EMAIL": "env@example.test", "POLYTOPE_USER_KEY": "env-key"},
        rc_path=tmp_path / "missing",
    ) == {"user_email": "env@example.test", "user_key": "env-key"}
    assert resolve_polytope_credentials(
        {"user_email": "   ", "user_key": "\t"},
        environ={"POLYTOPE_USER_EMAIL": "env@example.test", "POLYTOPE_USER_KEY": "env-key"},
        rc_path=tmp_path / "missing",
    ) == {"user_email": "env@example.test", "user_key": "env-key"}


def test_nex_rejects_extent_outside_v2_latitude_coverage() -> None:
    req = request(
        "nex_gddp_cmip6",
        ("pr",),
        start="2014-01-01",
        end="2014-01-01",
        bbox=BBox(-10, -61, 10, 0),
    )
    with pytest.raises(ConfigurationError, match="60 degrees south"):
        NexGddpProvider().validate(req)


def test_nex_historical_cannot_silently_clip_projection_dates() -> None:
    req = request(
        "nex_gddp_cmip6",
        ("pr",),
        start="2014-12-31",
        end="2015-01-02",
        options={
            "model": ("ACCESS-CM2",),
            "scenario": ("historical",),
            "variant": ("r1i1p1f1",),
        },
    )
    with pytest.raises(ConfigurationError, match="2015-01-01..2015-01-02"):
        NexGddpProvider().plan(req)


def test_nex_allow_partial_warns_and_keeps_exact_applicable_jobs(monkeypatch) -> None:
    root = "https://example.test/thredds/catalog/root/catalog.xml"
    provider = NexGddpProvider(catalog_root=root)
    documents = {
        root: catalog(ref("ACCESS-CM2", "ACCESS-CM2/catalog.xml")),
        "https://example.test/thredds/catalog/root/ACCESS-CM2/catalog.xml": catalog(
            ref("historical", "historical/catalog.xml")
        ),
        "https://example.test/thredds/catalog/root/ACCESS-CM2/historical/catalog.xml": catalog(
            ref("r1i1p1f1", "r1i1p1f1/catalog.xml")
        ),
        "https://example.test/thredds/catalog/root/ACCESS-CM2/historical/r1i1p1f1/catalog.xml": catalog(
            ref("pr", "pr/catalog.xml")
        ),
        "https://example.test/thredds/catalog/root/ACCESS-CM2/historical/r1i1p1f1/pr/catalog.xml": catalog(
            '<dataset name="pr_day_ACCESS-CM2_historical_r1i1p1f1_gn_2014_v2.0.nc" '
            'urlPath="AMES/NEX/historical-2014.nc" />'
        ),
    }

    class Session:
        def close(self) -> None:
            pass

    monkeypatch.setattr("geodaten.downloader.nex_gddp_cmip6.make_session", Session)
    monkeypatch.setattr(provider, "_fetch_catalog", lambda url, _session: documents[url])
    req = request(
        "nex_gddp_cmip6",
        ("pr",),
        start="2014-12-31",
        end="2015-01-02",
        options={
            "model": ("ACCESS-CM2",),
            "scenario": ("historical",),
            "variant": ("r1i1p1f1",),
        },
        allow_partial=True,
    )
    plan = provider.plan(req)
    assert plan.period_count == plan.slice_count == 1
    assert plan.warnings == ("Omitting unsupported NEX interval 2015-01-01..2015-01-02.",)
    assert [(job.start, job.end) for job in provider._jobs(req)] == [
        (date(2014, 12, 31), date(2014, 12, 31))
    ]


def test_nex_spec_uses_current_official_source_and_license() -> None:
    assert NexGddpProvider.spec.source_url == "https://www.nccs.nasa.gov/data-collections/nex-gddp-cmip6/"
    assert NexGddpProvider.spec.license == "CC0-1.0"
