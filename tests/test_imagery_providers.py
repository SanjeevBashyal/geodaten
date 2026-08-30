from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from geodaten.common.errors import ConfigurationError, DownloadError
from geodaten.common.models import BBox, DownloadRequest, OutputFormat, TimeRange
from geodaten.common.progress import CancellationToken, ProgressEvent
from geodaten.downloader.registry import get_provider
from geodaten.downloader.stac_imagery import (
    SEARCH_URL,
    Landsat9Provider,
    SasTokens,
    Sentinel2Provider,
    gdal_environment,
    group_by_date,
    item_datetime,
    resolve_date_window,
    search,
    vsiaz,
)

HREF = (
    "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/33/U/US/2025/06/21/"
    "S2C_MSIL2A_20250621T100611.SAFE/GRANULE/IMG_DATA/R10m/T33UUS_B04_10m.tif"
)


def item(
    identifier: str = "S2C_MSIL2A_20250621T100611_R022_T33UUS",
    *,
    when: str = "2025-06-21T10:06:11.025000Z",
    baseline: str | None = "05.11",
    assets: tuple[str, ...] = ("B04", "B08", "SCL"),
    eo_cloud: float = 17.9,
    tile: str = "33UUS",
    **properties: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "datetime": when,
        "platform": "Sentinel-2C",
        "eo:cloud_cover": eo_cloud,
        "s2:mgrs_tile": tile,
        **properties,
    }
    if baseline is not None:
        payload["s2:processing_baseline"] = baseline
    return {
        "id": identifier,
        "properties": payload,
        "assets": {
            name: {"href": HREF.replace("B04", name).replace("33UUS", tile)}
            for name in assets
        },
    }


def request(
    dataset: str = "sentinel2",
    *,
    variables: tuple[str, ...] = ("b04",),
    start: str = "2025-06-01",
    end: str = "2025-06-30",
    bbox: BBox = BBox(12.2, 51.2, 12.6, 51.5),
    output_dir: Path | None = None,
    output_format: OutputFormat = OutputFormat.TIF,
    options: dict[str, Any] | None = None,
    allow_partial: bool = False,
) -> DownloadRequest:
    return DownloadRequest(
        dataset=dataset,
        bbox=bbox,
        time=TimeRange(start, end),
        variables=variables,
        output_format=output_format,
        output_dir=output_dir or Path("geodaten-output"),
        options=options or {},
        allow_partial=allow_partial,
    )


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class FakeSession:
    """Records every call so the request bodies themselves can be asserted."""

    def __init__(self, pages: list[Any] | None = None, token: Any = None) -> None:
        self.pages = list(pages or [])
        self.token = token
        self.posts: list[tuple[str, Any, dict[str, str]]] = []
        self.gets: list[str] = []

    def post(self, url: str, *, json: Any = None, headers: Any = None, timeout: Any = None):
        self.posts.append((url, json, dict(headers or {})))
        return FakeResponse(self.pages.pop(0))

    def get(self, url: str, *, headers: Any = None, timeout: Any = None):
        self.gets.append(url)
        return FakeResponse(self.token)

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


def test_search_body_carries_the_exact_extent_window_and_filters() -> None:
    session = FakeSession([{"features": [item()], "links": []}])
    search(
        session,
        collection="landsat-c2-l2",
        bbox=BBox(12.2, 51.2, 12.6, 51.5),
        start=date(2025, 6, 1),
        end=date(2025, 6, 30),
        max_cloud_cover=10,
        platform="landsat-9",
    )
    url, body, _ = session.posts[0]
    assert url == SEARCH_URL
    assert body["collections"] == ["landsat-c2-l2"]
    assert body["bbox"] == [12.2, 51.2, 12.6, 51.5]
    assert body["datetime"] == "2025-06-01T00:00:00Z/2025-06-30T23:59:59Z"
    assert body["query"] == {
        "eo:cloud_cover": {"lte": 10},
        "platform": {"eq": "landsat-9"},
    }


def test_search_omits_a_platform_filter_when_the_collection_is_single_mission() -> None:
    session = FakeSession([{"features": [], "links": []}])
    search(
        session,
        collection="sentinel-2-l2a",
        bbox=BBox(12.2, 51.2, 12.6, 51.5),
        start=date(2025, 6, 1),
        end=date(2025, 6, 30),
        max_cloud_cover=20,
    )
    assert "platform" not in session.posts[0][1]["query"]


def test_search_follows_next_links_and_returns_every_page_in_time_order() -> None:
    later = item("later", when="2025-06-24T10:05:00Z")
    earlier = item("earlier", when="2025-06-21T10:06:11Z")
    session = FakeSession(
        [
            {
                "features": [later],
                "links": [
                    {"rel": "next", "href": f"{SEARCH_URL}?page=2", "body": {"token": "p2"}}
                ],
            },
            {"features": [earlier], "links": []},
        ]
    )
    found = search(
        session,
        collection="sentinel-2-l2a",
        bbox=BBox(12.2, 51.2, 12.6, 51.5),
        start=date(2025, 6, 1),
        end=date(2025, 6, 30),
        max_cloud_cover=20,
    )
    assert [entry["id"] for entry in found] == ["earlier", "later"]
    assert session.posts[1][0] == f"{SEARCH_URL}?page=2"
    assert session.posts[1][1] == {"token": "p2"}


def test_search_reports_a_transport_failure_without_leaking_the_query() -> None:
    import requests

    class Broken(FakeSession):
        def post(self, url: str, **kwargs: Any):
            raise requests.ConnectionError("boom")

    with pytest.raises(DownloadError, match="Could not search sentinel-2-l2a"):
        search(
            Broken(),
            collection="sentinel-2-l2a",
            bbox=BBox(12.2, 51.2, 12.6, 51.5),
            start=date(2025, 6, 1),
            end=date(2025, 6, 30),
            max_cloud_cover=20,
        )


# --------------------------------------------------------------------------
# temporal boundaries
# --------------------------------------------------------------------------


def test_boundaries_resolve_against_each_mission_first_acquisition() -> None:
    sentinel = Sentinel2Provider()
    landsat = Landsat9Provider()
    start, end = resolve_date_window(
        request(start="earliest", end="2016-01-01"), sentinel.first_date
    )
    assert start == date(2015, 6, 27)
    start, _ = resolve_date_window(
        request("landsat9", variables=("red",), start="earliest", end="2022-01-01"),
        landsat.first_date,
    )
    assert start == date(2021, 10, 31)


def test_year_boundaries_expand_to_whole_years_and_latest_is_today() -> None:
    start, end = resolve_date_window(request(start="2023", end="2023"), date(2015, 6, 27))
    assert (start, end) == (date(2023, 1, 1), date(2023, 12, 31))
    _, end = resolve_date_window(request(start="2023", end="latest"), date(2015, 6, 27))
    assert end == date.today()


def test_reversed_and_pre_mission_windows_are_configuration_errors() -> None:
    with pytest.raises(ConfigurationError, match="on or before"):
        resolve_date_window(request(start="2025-06-30", end="2025-06-01"), date(2015, 6, 27))
    with pytest.raises(ConfigurationError, match="no acquisitions before"):
        resolve_date_window(request(start="2019-01-01", end="2020-01-01"), date(2021, 10, 31))
    with pytest.raises(ConfigurationError, match="Invalid temporal boundary"):
        resolve_date_window(request(start="June 2025", end="latest"), date(2015, 6, 27))


def test_item_datetime_is_naive_utc_truncated_to_seconds() -> None:
    assert item_datetime(item()) == datetime(2025, 6, 21, 10, 6, 11)
    assert item_datetime(
        {"id": "x", "properties": {"start_datetime": "2025-06-21T12:06:11+02:00"}}
    ) == datetime(2025, 6, 21, 10, 6, 11)
    with pytest.raises(DownloadError, match="no acquisition time"):
        item_datetime({"id": "x", "properties": {}})


# --------------------------------------------------------------------------
# signing
# --------------------------------------------------------------------------


def _expiry(minutes: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat().replace(
        "+00:00", "Z"
    )


def test_one_token_is_fetched_per_container_and_reused_across_bands() -> None:
    session = FakeSession(token={"token": "st=a&se=b", "msft:expiry": _expiry(60)})
    tokens = SasTokens(session)
    assert tokens.token(HREF) == "st=a&se=b"
    tokens.token(HREF.replace("T33UUS_B04_10m", "T33UUS_B08_10m"))
    assert session.gets == [
        "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel2l2a01/sentinel2-l2"
    ]
    assert SasTokens.container(HREF) == ("sentinel2l2a01", "sentinel2-l2")


def test_a_token_close_to_expiry_is_refreshed_before_gdal_uses_it() -> None:
    session = FakeSession(token={"token": "st=a", "msft:expiry": _expiry(1)})
    tokens = SasTokens(session)
    tokens.token(HREF)
    tokens.token(HREF)
    assert len(session.gets) == 2


def test_signing_failures_and_missing_containers_are_reported() -> None:
    session = FakeSession(token={"msft:expiry": _expiry(60)})
    with pytest.raises(DownloadError, match="no token"):
        SasTokens(session).token(HREF)
    with pytest.raises(DownloadError, match="signing container"):
        SasTokens(session).token("file:///local/path.tif")


def test_the_signature_reaches_gdal_out_of_band_rather_than_in_the_path() -> None:
    path = vsiaz(f"{HREF}?st=a&se=b&sp=rl")
    assert path.startswith("/vsiaz/sentinel2-l2/33/U/US/")
    assert "st=a" not in path and "?" not in path
    environment = gdal_environment("sentinel2l2a01", "st=a&se=b&sp=rl")
    assert environment["AZURE_STORAGE_ACCOUNT"] == "sentinel2l2a01"
    assert environment["AZURE_STORAGE_SAS_TOKEN"] == "st=a&se=b&sp=rl"


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------


def test_sentinel2_offset_follows_the_item_processing_baseline() -> None:
    provider = Sentinel2Provider()
    assert provider.band_attrs("b04", item(baseline="05.11"))["add_offset"] == -0.1
    assert provider.band_attrs("b04", item(baseline="03.01"))["add_offset"] == 0.0
    assert provider.band_attrs("b04", item(baseline=None))["add_offset"] == 0.0
    assert provider.band_attrs("b04", item())["scale_factor"] == 1e-4
    # Categorical layers must never be given a radiometric scale.
    assert "scale_factor" not in provider.band_attrs("scl", item())


def test_landsat_reflectance_and_thermal_carry_their_published_packing() -> None:
    provider = Landsat9Provider()
    reflectance = provider.band_attrs("red", {"properties": {}})
    assert (reflectance["scale_factor"], reflectance["add_offset"]) == (2.75e-05, -0.2)
    thermal = provider.band_attrs("lwir11", {"properties": {}})
    assert (thermal["scale_factor"], thermal["add_offset"], thermal["units"]) == (
        0.00341802,
        149.0,
        "K",
    )
    assert "scale_factor" not in provider.band_attrs("qa_pixel", {"properties": {}})


def test_scenes_group_by_acquisition_date_not_by_timestamp() -> None:
    """Landsat rows of one pass are seconds apart and must still merge."""

    groups = group_by_date(
        [
            item("row025", when="2026-08-13T10:09:20Z"),
            item("row024", when="2026-08-13T10:08:56Z"),
            item("other", when="2026-08-22T10:02:46Z"),
        ]
    )
    assert list(groups) == [date(2026, 8, 13), date(2026, 8, 22)]
    # Chronological within a date, so mosaic order is deterministic.
    assert [entry["id"] for entry in groups[date(2026, 8, 13)]] == ["row024", "row025"]
    assert [entry["id"] for entry in groups[date(2026, 8, 22)]] == ["other"]


# --------------------------------------------------------------------------
# provider contract
# --------------------------------------------------------------------------


def test_registry_exposes_both_imagery_datasets_as_tif_only() -> None:
    for identifier in ("sentinel2", "landsat9"):
        spec = get_provider(identifier).spec
        assert spec.group == "Imagery"
        assert spec.formats == (OutputFormat.TIF,)
        assert spec.temporal
        assert spec.credentials == ()


def test_validate_rejects_netcdf_antimeridian_and_impossible_cloud_cover() -> None:
    provider = Sentinel2Provider()
    with pytest.raises(ConfigurationError, match="cannot be written as"):
        provider.validate(request(output_format=OutputFormat.NETCDF))
    with pytest.raises(ConfigurationError, match="antimeridian"):
        provider.validate(request(bbox=BBox(179.0, 51.2, -179.0, 51.5)))
    with pytest.raises(ConfigurationError, match="max_cloud_cover"):
        provider.validate(request(options={"max_cloud_cover": 140}))


def test_plan_counts_every_scene_and_variable_combination(monkeypatch) -> None:
    provider = Sentinel2Provider()
    items = (item("a"), item("b", when="2025-06-24T10:05:00Z"))
    monkeypatch.setattr(provider, "_items_for", lambda _request: items)
    plan = provider.plan(request(variables=("b04", "b08")))
    assert plan.period_count == 2
    assert plan.slice_count == 4
    assert plan.details["scenes"] == ("a", "b")
    assert plan.details["assets"] == ("B04", "B08")
    assert plan.details["first_acquisition"] == "2025-06-21T10:06:11"
    assert plan.estimated_bytes is None


def test_landsat_keeps_every_scene_whatever_the_cloud_cover(monkeypatch) -> None:
    """A 16-day revisit makes a default cloud threshold too costly."""

    provider = Landsat9Provider()
    assert provider.spec.options[0].default is None
    calls: list[float | None] = []

    def fake_search(_session, *, max_cloud_cover=None, **kwargs):
        calls.append(max_cloud_cover)
        return [item("a", eo_cloud=88.51), item("b", eo_cloud=42.38)]

    monkeypatch.setattr("geodaten.downloader.stac_imagery.search", fake_search)
    monkeypatch.setattr("geodaten.downloader.stac_imagery.make_session", FakeSession)
    plan = provider.plan(request("landsat9", variables=("red",)))
    assert calls == [None]
    # Both fixtures share a date, so they merge into one output.
    assert plan.period_count == 1
    assert plan.details["scene_count"] == 2
    assert plan.details["max_cloud_cover"] is None


def test_landsat_still_honours_an_explicitly_requested_limit(monkeypatch) -> None:
    provider = Landsat9Provider()
    calls: list[float | None] = []

    def fake_search(_session, *, max_cloud_cover=None, **kwargs):
        calls.append(max_cloud_cover)
        return [item("a", eo_cloud=5.0)]

    monkeypatch.setattr("geodaten.downloader.stac_imagery.search", fake_search)
    monkeypatch.setattr("geodaten.downloader.stac_imagery.make_session", FakeSession)
    provider.plan(request("landsat9", variables=("red",), options={"max_cloud_cover": 10}))
    assert calls == [10]


def test_an_empty_result_names_the_cloud_threshold_that_caused_it(monkeypatch) -> None:
    """A cloud-filtered miss must not read as missing archive coverage."""

    provider = Sentinel2Provider()
    calls: list[float | None] = []

    def fake_search(_session, *, max_cloud_cover=None, **kwargs):
        calls.append(max_cloud_cover)
        if max_cloud_cover is None:
            return [
                item("a", eo_cloud=88.51),
                item("b", eo_cloud=42.38),
            ]
        return []

    monkeypatch.setattr("geodaten.downloader.stac_imagery.search", fake_search)
    monkeypatch.setattr("geodaten.downloader.stac_imagery.make_session", FakeSession)
    with pytest.raises(DownloadError) as raised:
        provider.plan(request())
    message = str(raised.value)
    assert "at or below 20%" in message
    assert "2 scene(s) do cover it, at 42.38%, 88.51%" in message
    assert "Raise max_cloud_cover" in message
    # The unfiltered count is only paid for on the failure path.
    assert calls == [20.0, None]


def test_an_empty_search_is_an_error_rather_than_an_empty_download(monkeypatch) -> None:
    provider = Landsat9Provider()
    monkeypatch.setattr(
        "geodaten.downloader.stac_imagery.search", lambda *args, **kwargs: []
    )
    monkeypatch.setattr("geodaten.downloader.stac_imagery.make_session", FakeSession)
    with pytest.raises(DownloadError, match="No Landsat 9 .* scenes matched"):
        provider.plan(request("landsat9", variables=("red",)))


def _stub_slices(monkeypatch, provider, items, tmp_path: Path) -> dict[str, list[Any]]:
    """Run slices() without GDAL or the network, capturing every GDAL call."""

    log: dict[str, list[Any]] = {"warp": [], "vrt": [], "run": []}

    def fake_warp(source, destination, **kwargs):
        sources = [str(value) for value in source] if isinstance(source, list) else [str(source)]
        log["warp"].append(
            {"sources": sources, "destination": Path(destination), **kwargs}
        )
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"tif")
        return Path(destination)

    def fake_build_vrt(sources, destination, **kwargs):
        log["vrt"].append({"sources": [Path(s).name for s in sources], **kwargs})
        Path(destination).write_bytes(b"vrt")
        return Path(destination)

    def fake_run(command, **kwargs):
        command = [str(value) for value in command]
        log["run"].append(command)
        Path(command[-1]).write_bytes(b"vrt")
        return ""

    monkeypatch.setattr("geodaten.downloader.stac_imagery.gdal.require_commands", lambda *a: None)
    monkeypatch.setattr("geodaten.downloader.stac_imagery.gdal.warp_to_cog", fake_warp)
    monkeypatch.setattr("geodaten.downloader.stac_imagery.gdal.build_vrt", fake_build_vrt)
    monkeypatch.setattr("geodaten.downloader.stac_imagery.gdal.run", fake_run)
    monkeypatch.setattr(
        "geodaten.downloader.stac_imagery.gdal.info", lambda *a, **k: {"size": [429, 429]}
    )
    monkeypatch.setattr(provider, "_items_for", lambda _request: items)
    monkeypatch.setattr(
        "geodaten.downloader.stac_imagery.make_session",
        lambda: FakeSession(token={"token": "st=a", "msft:expiry": _expiry(60)}),
    )
    return log


#: The four tiles that really cover bbox 12.2,51.2 -> 12.6,51.5 on 2025-06-21,
#: spanning EPSG:32632 and EPSG:32633.
FOUR_TILES = (
    item("a", tile="33UUT"),
    item("b", tile="33UUS"),
    item("c", tile="32UQC"),
    item("d", tile="32UQB"),
)


def test_every_tile_of_one_date_merges_into_a_single_output(
    monkeypatch, tmp_path: Path
) -> None:
    provider = Sentinel2Provider()
    log = _stub_slices(monkeypatch, provider, FOUR_TILES, tmp_path)
    ask = request(variables=("b04", "scl"))
    produced = list(provider.slices(ask, tmp_path, lambda _event: None, CancellationToken()))

    # One slice per variable, not one per tile.
    assert [entry.variable for entry in produced] == ["b04", "scl"]
    assert all(entry.time == date(2025, 6, 21) for entry in produced)
    assert all(entry.dimensions == {} for entry in produced)
    assert all(entry.band == 1 for entry in produced)

    assert len(log["warp"]) == 2
    for call in log["warp"]:
        assert call["bbox"] == ask.bbox
        # All four tiles reach one gdalwarp, which reprojects across UTM zones.
        assert len(call["sources"]) == 4
        assert call["env"]["AZURE_STORAGE_ACCOUNT"] == "sentinel2l2a01"
        assert call["env"]["AZURE_STORAGE_SAS_TOKEN"] == "st=a"
    assert sorted(Path(s).name for s in log["warp"][0]["sources"]) == [
        "T32UQB_B04_10m.tif",
        "T32UQC_B04_10m.tif",
        "T33UUS_B04_10m.tif",
        "T33UUT_B04_10m.tif",
    ]
    assert log["warp"][0]["scale"] == 1e-4 and log["warp"][0]["offset"] == -0.1
    assert log["warp"][1]["scale"] is None
    assert log["warp"][0]["metadata"]["scene_count"] == 4
    # Each staged mosaic is removed once the writer has consumed it.
    assert not any(call["destination"].exists() for call in log["warp"])


def test_separate_dates_stay_separate_outputs(monkeypatch, tmp_path: Path) -> None:
    provider = Landsat9Provider()
    scenes = (
        item("row024", when="2026-08-13T10:08:56Z", assets=("red",)),
        item("row025", when="2026-08-13T10:09:20Z", assets=("red",)),
        item("later", when="2026-08-22T10:02:46Z", assets=("red",)),
    )
    log = _stub_slices(monkeypatch, provider, scenes, tmp_path)
    produced = list(
        provider.slices(
            request("landsat9", variables=("red",)),
            tmp_path,
            lambda _event: None,
            CancellationToken(),
        )
    )
    assert [entry.time for entry in produced] == [date(2026, 8, 13), date(2026, 8, 22)]
    # The two rows of one pass merge; the later date does not join them.
    assert [len(call["sources"]) for call in log["warp"]] == [2, 1]


def test_rgb_stacks_three_assets_onto_one_grid(monkeypatch, tmp_path: Path) -> None:
    provider = Sentinel2Provider()
    log = _stub_slices(
        monkeypatch, provider, (item("a", assets=("B04", "B03", "B02")),), tmp_path
    )
    produced = list(
        provider.slices(
            request(variables=("rgb",)), tmp_path, lambda _e: None, CancellationToken()
        )
    )
    (entry,) = produced
    assert entry.variable == "rgb"
    # None keeps every band, so the writer does not extract band 1.
    assert entry.band is None
    assert entry.time == date(2025, 6, 21)

    assert [Path(call["sources"][0]).name for call in log["warp"]] == [
        "T33UUS_B04_10m.tif",
        "T33UUS_B03_10m.tif",
        "T33UUS_B02_10m.tif",
    ]
    # The first merge sets the grid; the others are pinned to it so they stack.
    assert log["warp"][0]["shape"] is None
    assert [call["shape"] for call in log["warp"][1:]] == [(429, 429), (429, 429)]

    assert log["vrt"] == [
        {
            "sources": ["component-0.tif", "component-1.tif", "component-2.tif"],
            "separate": True,
            "cancel": log["vrt"][0]["cancel"],
        }
    ]
    (translate,) = log["run"]
    assert translate[translate.index("-colorinterp") + 1] == "red,green,blue"


def test_landsat_rgb_uses_its_own_asset_names(monkeypatch, tmp_path: Path) -> None:
    provider = Landsat9Provider()
    assert provider.bands["rgb"].stack == ("red", "green", "blue")
    assert provider.bands["rgb"].assets == ("red", "green", "blue")
    assert Sentinel2Provider().bands["b04"].assets == ("B04",)


def test_a_group_spanning_storage_containers_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    provider = Sentinel2Provider()
    stray = item("stray")
    stray["assets"]["B04"]["href"] = HREF.replace("sentinel2-l2/", "other-container/")
    _stub_slices(monkeypatch, provider, (item("a"), stray), tmp_path)
    with pytest.raises(DownloadError, match="several storage containers"):
        list(
            provider.slices(
                request(variables=("b04",)), tmp_path, lambda _e: None, CancellationToken()
            )
        )


def test_mixed_processing_baselines_warn_before_merging(
    monkeypatch, tmp_path: Path
) -> None:
    provider = Sentinel2Provider()
    scenes = (item("a", baseline="05.11"), item("b", tile="32UQB", baseline="03.01"))
    _stub_slices(monkeypatch, provider, scenes, tmp_path)
    events: list[ProgressEvent] = []
    produced = list(
        provider.slices(
            request(variables=("b04",)), tmp_path, events.append, CancellationToken()
        )
    )
    assert len(produced) == 1
    assert any(
        event.stage == "warning" and "03.01" in event.message and "05.11" in event.message
        for event in events
    )


def test_a_missing_asset_stops_the_run_unless_partial_is_allowed(
    monkeypatch, tmp_path: Path
) -> None:
    provider = Sentinel2Provider()
    scenes = (item("a", assets=("B04",)),)
    _stub_slices(monkeypatch, provider, scenes, tmp_path)
    with pytest.raises(DownloadError, match="no b08 asset"):
        list(
            provider.slices(
                request(variables=("b04", "b08")), tmp_path, lambda _e: None, CancellationToken()
            )
        )

    events: list[ProgressEvent] = []
    _stub_slices(monkeypatch, provider, scenes, tmp_path)
    produced = list(
        provider.slices(
            request(variables=("b04", "b08"), allow_partial=True),
            tmp_path,
            events.append,
            CancellationToken(),
        )
    )
    assert [entry.variable for entry in produced] == ["b04"]
    assert any(event.stage == "warning" and "b08" in event.message for event in events)
