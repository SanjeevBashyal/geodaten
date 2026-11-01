from __future__ import annotations

from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from geodaten.common.errors import ConfigurationError, DownloadError
from geodaten.common.models import BBox, DownloadRequest, OutputFormat, TimeRange
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
    *,
    start: str,
    end: str,
    bbox: BBox = BBox(79, 26, 89, 31),
    options: dict | None = None,
    allow_partial: bool = False,
) -> DownloadRequest:
    return DownloadRequest(
        dataset="nex_gddp_cmip6",
        bbox=bbox,
        variables=("pr",),
        output_format=OutputFormat.TIF,
        output_dir=Path("output"),
        time=TimeRange(start, end),
        options=options or {},
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


def test_nex_catalog_discovers_grid_version_and_size() -> None:
    xml = catalog(
        '<dataset name="pr_day_ACCESS-CM2_historical_r1i1p1f1_gr1_2014_v2.0.nc" '
        'urlPath="AMES/NEX/file.nc"><dataSize units="Mbytes">1.5</dataSize></dataset>'
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


def test_nex_catalog_refs_resolve_relative_urls() -> None:
    refs = _catalog_refs(
        catalog(ref("historical", "historical/catalog.xml")),
        "https://example.test/thredds/catalog/model/catalog.xml",
    )
    assert refs == {
        "historical": "https://example.test/thredds/catalog/model/historical/catalog.xml"
    }


def test_nex_jobs_require_every_applicable_year() -> None:
    assets = (NexAsset("M", "historical", "r1", "pr", 2014, "gn", "2014.nc"),)
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


def test_nex_jobs_intersect_year_boundaries_exactly() -> None:
    assets = tuple(
        NexAsset("M", "historical", "r1", "pr", year, "gn", f"{year}.nc")
        for year in (2014, 2015)
    )
    jobs = subset_jobs(assets, date(2014, 12, 31), date(2015, 1, 2))
    assert [(job.asset.year, job.start, job.end) for job in jobs] == [
        (2014, date(2014, 12, 31), date(2014, 12, 31)),
        (2015, date(2015, 1, 1), date(2015, 1, 2)),
    ]


def test_nex_ncss_query_preserves_extent_date_and_variable() -> None:
    item = request(start="2014-12-31", end="2014-12-31")
    asset = NexAsset("M", "historical", "r1", "pr", 2014, "gr1", "a.nc")
    url = ncss_query_url(
        NexSubsetJob(asset, date(2014, 12, 31), date(2014, 12, 31)), item
    )
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


def test_nex_rejects_unsupported_spatial_and_temporal_coverage() -> None:
    provider = NexGddpProvider()
    with pytest.raises(ConfigurationError, match="60 degrees south"):
        provider.validate(
            request(start="2014", end="2014", bbox=BBox(10, -61, 11, -59))
        )
    with pytest.raises(ConfigurationError, match="do not cover"):
        provider.plan(
            request(
                start="2014-12-31",
                end="2015-01-01",
                options={
                    "model": ("ACCESS-CM2",),
                    "scenario": ("historical",),
                    "variant": ("r1i1p1f1",),
                },
            )
        )


def test_nex_plan_walks_advertised_catalogs(monkeypatch) -> None:
    root = "https://example.test/root/catalog.xml"
    provider = NexGddpProvider(catalog_root=root)
    documents = {
        root: catalog(ref("ACCESS-CM2", "ACCESS-CM2/catalog.xml")),
        "https://example.test/root/ACCESS-CM2/catalog.xml": catalog(
            ref("historical", "historical/catalog.xml")
        ),
        "https://example.test/root/ACCESS-CM2/historical/catalog.xml": catalog(
            ref("r1i1p1f1", "r1i1p1f1/catalog.xml")
        ),
        "https://example.test/root/ACCESS-CM2/historical/r1i1p1f1/catalog.xml": catalog(
            ref("pr", "pr/catalog.xml")
        ),
        "https://example.test/root/ACCESS-CM2/historical/r1i1p1f1/pr/catalog.xml": catalog(
            '<dataset name="pr_day_ACCESS-CM2_historical_r1i1p1f1_gn_2014_v2.0.nc" '
            'urlPath="AMES/NEX/2014.nc" />'
        ),
    }

    class Session:
        @staticmethod
        def close() -> None:
            pass

    monkeypatch.setattr("geodaten.downloader.nex_gddp_cmip6.make_session", Session)
    monkeypatch.setattr(provider, "_fetch_catalog", lambda url, _session: documents[url])
    plan = provider.plan(
        request(
            start="2014-12-31",
            end="2014-12-31",
            options={
                "model": ("ACCESS-CM2",),
                "scenario": ("historical",),
                "variant": ("r1i1p1f1",),
            },
        )
    )
    assert plan.period_count == 1
    assert plan.slice_count == 1
    assert plan.details["grids"] == ["gn"]


def test_nex_spec_uses_current_source_and_license() -> None:
    spec = NexGddpProvider.spec
    assert spec.source_url == "https://www.nccs.nasa.gov/data-collections/nex-gddp-cmip6/"
    assert spec.license == "CC0-1.0"
