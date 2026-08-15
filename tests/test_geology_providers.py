from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

import geodaten.downloader.glim as glim_module
import geodaten.downloader.service as service_module
from geodaten.common.errors import ConfigurationError, DownloadError, OutputError
from geodaten.common.gdal import run, validate_vector, vector_layers, vector_to_shapefile
from geodaten.common.models import BBox, DownloadRequest, OutputFormat, TimeRange, VectorSlice
from geodaten.common.progress import CancellationToken, null_progress
from geodaten.common.writers import ShapefileWriter, make_writer
from geodaten.downloader.glim import (
    ARCHIVE_URL,
    CLASSES,
    FIELD_NAMES,
    GEODATABASE_NAME,
    SOURCE_CRS,
    SOURCE_LAYER,
    GlimProvider,
    archive_url,
    extent_area,
    extract_geodatabase,
    prefilter_bounds,
    selected_fields,
    source_sql,
)
from geodaten.downloader.service import DownloadService


def request(
    *,
    bbox: BBox = BBox(79, 26, 90, 31),
    output_dir: Path | None = None,
    options: dict[str, object] | None = None,
    output_format: OutputFormat = OutputFormat.SHP,
    time: TimeRange | None = None,
    overwrite: bool = False,
) -> DownloadRequest:
    return DownloadRequest(
        dataset="glim",
        bbox=bbox,
        time=time,
        variables=("lithology",),
        output_format=output_format,
        output_dir=output_dir or Path("unused"),
        options=options or {},
        overwrite=overwrite,
    )


def fake_geodatabase(tmp_path: Path) -> Path:
    """Build a small stand-in for the GLiM layer in its native projection.

    The published geodatabase is 1.2 GB, so the tests build a real file
    geodatabase carrying the same driver, layer name, projection, and columns.
    """

    features = [
        ("IND201", "sumx__", "su", 10.0, 20.0, [(79.5, 26.5), (81.0, 26.5), (81.0, 28.0), (79.5, 28.0)]),
        ("CHN36110", "vapysr", "va", 30.0, 40.0, [(88.0, 29.0), (95.0, 29.0), (95.0, 30.5), (88.0, 30.5)]),
        ("NPL7", "scmx__", "sc", 1.5, 2.5, [(120.0, 5.0), (121.0, 5.0), (121.0, 6.0), (120.0, 6.0)]),
        # The equal-area source projection ends at the antimeridian, so a unit
        # reaching across it is stored as one polygon per side, as here.
        ("FJI1", "mtmx__", "mt", 7.0, 8.0, [(179.2, -17.5), (179.9, -17.5), (179.9, -16.5), (179.2, -16.5)]),
        ("FJI2", "mtmx__", "mt", 7.0, 8.0, [(-179.9, -17.5), (-179.2, -17.5), (-179.2, -16.5), (-179.9, -16.5)]),
    ]
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "IDENTITY_": identity,
                    "Litho": litho,
                    "xx": code,
                    # Large enough that a default DBF width would lose digits.
                    "Shape_Length": length * 1e6 + 0.123456,
                    "Shape_Area": area * 1e11 + 0.654321,
                },
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[*ring, ring[0]]]],
                },
            }
            for identity, litho, code, length, area, ring in features
        ],
    }
    source = tmp_path / "source.geojson"
    source.write_text(json.dumps(geojson), encoding="utf-8")
    geodatabase = tmp_path / "fake.gdb"
    run(
        (
            "ogr2ogr",
            "-f",
            "OpenFileGDB",
            str(geodatabase),
            str(source),
            "-nln",
            SOURCE_LAYER,
            "-s_srs",
            "EPSG:4326",
            "-t_srs",
            SOURCE_CRS,
            "-nlt",
            "MULTIPOLYGON",
        )
    )
    return geodatabase


def features(path: Path) -> list[dict[str, object]]:
    output = run(
        ("ogr2ogr", "-f", "GeoJSON", "/vsistdout/", str(path)),
        capture=True,
    )
    return [item["properties"] for item in json.loads(output)["features"]]


_EXTENT = re.compile(
    r"Extent:\s*\(\s*(\S+?),\s*(\S+?)\s*\)\s*-\s*\(\s*(\S+?),\s*(\S+?)\s*\)"
)


def bounds(path: Path) -> tuple[float, float, float, float]:
    output = run(("ogrinfo", "-ro", "-so", "-nomd", str(path), "-al"), capture=True)
    match = _EXTENT.search(output)
    assert match is not None, output
    return tuple(float(value) for value in match.groups())  # type: ignore[return-value]


def crop(provider: GlimProvider, item: DownloadRequest, workspace: Path) -> list[dict[str, object]]:
    """Read each yielded slice while the provider still owns its scratch files."""

    workspace.mkdir(parents=True, exist_ok=True)
    captured: list[dict[str, object]] = []
    for produced in provider.slices(item, workspace, null_progress, CancellationToken()):
        source = Path(produced.source)
        captured.append(
            {
                "variable": produced.variable,
                "attrs": dict(produced.attrs),
                "rows": features(source),
                "layers": vector_layers(source),
                "bounds": bounds(source),
                "type": type(produced),
            }
        )
    return captured


def test_archive_url_requests_the_file_rather_than_the_preview_page() -> None:
    assert parse_qs(urlsplit(ARCHIVE_URL).query)["dl"] == ["0"]
    query = parse_qs(urlsplit(archive_url()).query)
    assert query["dl"] == ["1"]
    assert query["rlkey"] == ["89q1aj8zfpmiv82y41xe6ttmp"]
    assert urlsplit(archive_url()).path == urlsplit(ARCHIVE_URL).path


def test_spec_is_a_static_vector_dataset() -> None:
    spec = GlimProvider.spec
    assert spec.temporal is False
    assert spec.format_ids == ("shp",)
    assert spec.variable_ids == ("lithology",)
    assert dict(CLASSES)["sc"] == "carbonate sedimentary rocks"


@pytest.mark.parametrize("output_format", (OutputFormat.TIF, OutputFormat.NETCDF))
def test_raster_formats_are_rejected(output_format: OutputFormat) -> None:
    with pytest.raises(ConfigurationError, match="choose from: shp"):
        GlimProvider().validate(request(output_format=output_format))


def test_temporal_bounds_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="static"):
        GlimProvider().validate(request(time=TimeRange("2000", "2001")))


def test_selected_fields_defaults_to_the_whole_attribute_table() -> None:
    assert selected_fields(request()) == FIELD_NAMES
    assert selected_fields(request(options={"fields": ("xx", "IDENTITY_")})) == (
        "IDENTITY_",
        "xx",
    )
    assert selected_fields(request(options={"fields": "xx"})) == ("xx",)
    with pytest.raises(ConfigurationError, match="at least one"):
        selected_fields(request(options={"fields": ()}))


def test_source_sql_widens_doubles_and_pre_launders_long_names() -> None:
    statement = source_sql(FIELD_NAMES)
    assert statement.endswith('FROM "GLiM_export"')
    assert 'CAST("Shape_Area" AS numeric(24,6)) AS "Shape_Area"' in statement
    # Aliased up front so an appended antimeridian segment matches the column
    # that the shapefile driver actually created.
    assert 'CAST("Shape_Length" AS numeric(24,6)) AS "Shape_Leng"' in statement
    assert '"Shape_Length"' not in statement.split(" AS ")[-1]
    assert source_sql(("xx",)) == 'SELECT "xx" AS "xx" FROM "GLiM_export"'


def test_prefilter_bounds_cover_the_curved_projected_extent() -> None:
    bbox = BBox(-30, -10, 30, 10)
    minimum_x, minimum_y, maximum_x, maximum_y = prefilter_bounds(bbox)
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", SOURCE_CRS, always_xy=True)
    for longitude in (-30, -15, 0, 15, 30):
        for latitude in (-10, -5, 0, 5, 10):
            x, y = transformer.transform(longitude, latitude)
            assert minimum_x <= x <= maximum_x
            assert minimum_y <= y <= maximum_y


def test_extent_area_counts_both_antimeridian_segments() -> None:
    assert extent_area(BBox(79, 26, 90, 31)) == pytest.approx(55.0)
    assert extent_area(BBox(179, -18, -179, -16)) == pytest.approx(4.0)


def test_unclipped_antimeridian_requests_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="clip=true"):
        GlimProvider().validate(
            request(bbox=BBox(179, -18, -179, -16), options={"clip": False})
        )


def test_missing_local_source_is_reported_before_any_transfer(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="local_source does not exist"):
        GlimProvider().validate(request(options={"local_source": tmp_path / "absent.gdb"}))


def test_plan_reports_staging_transfer_and_shapefile_limits(tmp_path: Path) -> None:
    geodatabase = fake_geodatabase(tmp_path)
    plan = GlimProvider().plan(request(options={"local_source": geodatabase}))

    assert plan.slice_count == 1
    assert plan.estimated_bytes == 0
    assert plan.details["fields"] == FIELD_NAMES
    assert plan.details["shapefile_field_names"] == (
        "IDENTITY_",
        "Litho",
        "xx",
        "Shape_Leng",
        "Shape_Area",
    )
    assert plan.details["clip"] is True
    assert any("Shape_Length and Shape_Area" in warning for warning in plan.warnings)
    assert not any("source archive is transferred" in warning for warning in plan.warnings)

    untouched = GlimProvider().plan(request(output_dir=tmp_path / "empty"))
    assert untouched.estimated_bytes == glim_module.ARCHIVE_BYTES
    assert any("source archive is transferred" in warning for warning in untouched.warnings)

    global_plan = GlimProvider().plan(
        request(bbox=BBox(-180, -60, 180, 80), options={"local_source": geodatabase})
    )
    assert any("two gigabytes" in warning for warning in global_plan.warnings)


def test_slices_clip_to_the_extent_and_keep_the_attribute_table(tmp_path: Path) -> None:
    geodatabase = fake_geodatabase(tmp_path)

    (produced,) = crop(
        GlimProvider(),
        request(options={"local_source": geodatabase}),
        tmp_path / "workspace",
    )

    assert produced["type"] is VectorSlice
    assert produced["variable"] == "lithology"
    assert produced["attrs"]["clipped"] is True
    assert produced["layers"] == {"lithology": 2}
    rows = {row["IDENTITY_"]: row for row in produced["rows"]}
    # NPL7 lies outside the extent; CHN36110 straddles its eastern edge.
    assert set(rows) == {"IND201", "CHN36110"}
    assert rows["IND201"]["xx"] == "su"
    assert rows["IND201"]["Litho"] == "sumx__"
    assert rows["IND201"]["Shape_Area"] == pytest.approx(20.0 * 1e11 + 0.654321, abs=1e-3)
    assert rows["IND201"]["Shape_Leng"] == pytest.approx(10.0 * 1e6 + 0.123456, abs=1e-3)
    assert produced["bounds"] == pytest.approx((79.5, 26.5, 90.0, 30.5), abs=1e-6)


def test_slices_can_keep_whole_polygons(tmp_path: Path) -> None:
    geodatabase = fake_geodatabase(tmp_path)

    (produced,) = crop(
        GlimProvider(),
        request(options={"local_source": geodatabase, "clip": False}),
        tmp_path / "workspace",
    )

    assert produced["attrs"]["clipped"] is False
    # The unclipped polygon keeps reaching past the eastern edge of the request.
    assert produced["bounds"][2] > 90.0
    assert {row["IDENTITY_"] for row in produced["rows"]} == {"IND201", "CHN36110"}


def test_slices_merge_both_antimeridian_segments(tmp_path: Path) -> None:
    geodatabase = fake_geodatabase(tmp_path)

    (produced,) = crop(
        GlimProvider(),
        request(bbox=BBox(179, -18, -179, -16), options={"local_source": geodatabase}),
        tmp_path / "workspace",
    )

    # Both sides of the antimeridian land in the same shapefile, and the appended
    # segment keeps its attributes instead of arriving as NULL.
    rows = {row["IDENTITY_"]: row for row in produced["rows"]}
    assert sorted(rows) == ["FJI1", "FJI2"]
    for row in rows.values():
        assert row["xx"] == "mt"
        assert row["Shape_Leng"] == pytest.approx(7.0 * 1e6 + 0.123456, abs=1e-3)
        assert row["Shape_Area"] == pytest.approx(8.0 * 1e11 + 0.654321, abs=1e-3)
    west, _south, east, _north = produced["bounds"]
    assert west == pytest.approx(-179.9, abs=1e-3)
    assert east == pytest.approx(179.9, abs=1e-3)


def test_local_zip_archive_is_unpacked(tmp_path: Path) -> None:
    geodatabase = fake_geodatabase(tmp_path)
    archive = tmp_path / "LiMW_GIS-2015.gdb.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for member in sorted(geodatabase.iterdir()):
            bundle.write(member, f"LiMW_GIS 2015.gdb/{member.name}")

    (produced,) = crop(
        GlimProvider(),
        request(options={"local_source": archive}),
        tmp_path / "workspace",
    )

    assert produced["layers"] == {"lithology": 2}


def test_extraction_refuses_members_outside_the_geodatabase(tmp_path: Path) -> None:
    archive = tmp_path / "hostile.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("LiMW_GIS 2015.gdb/a00000001.gdbtable", "payload")
        bundle.writestr("LiMW_GIS 2015.gdb/../escaped.txt", "payload")

    with pytest.raises(DownloadError, match="unsafe GLiM archive member"):
        extract_geodatabase(
            archive, tmp_path / "out.gdb", progress=null_progress, cancel=CancellationToken()
        )
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path / "out.gdb").exists()


def test_extraction_rejects_an_archive_without_a_geodatabase(tmp_path: Path) -> None:
    archive = tmp_path / "wrong.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("notes/readme.txt", "payload")

    with pytest.raises(DownloadError, match="exactly one file geodatabase"):
        extract_geodatabase(
            archive, tmp_path / "out.gdb", progress=null_progress, cancel=CancellationToken()
        )


def test_shapefile_writer_installs_every_sidecar_and_resumes(tmp_path: Path) -> None:
    geodatabase = fake_geodatabase(tmp_path)
    staged = tmp_path / "staged" / "lithology.shp"
    vector_to_shapefile(
        geodatabase,
        staged,
        layer=SOURCE_LAYER,
        layer_name="lithology",
        target_crs="EPSG:4326",
    )
    item = VectorSlice(variable="lithology", source=staged)

    first = ShapefileWriter(request(output_dir=tmp_path / "out"))
    destination = first.write(item)
    outputs = first.close()

    assert destination.name == "glim_lithology.shp"
    assert [path.suffix for path in outputs] == [".shp", ".shx", ".dbf", ".prj", ".cpg"]
    assert all(path.is_file() for path in outputs)
    assert (destination.parent / "glim_lithology.cpg").read_text().strip() == "UTF-8"
    assert first.skipped == 0

    resumed = ShapefileWriter(request(output_dir=tmp_path / "out"))
    resumed.write(item)
    assert resumed.skipped == 1
    assert resumed.close() == outputs

    forced = ShapefileWriter(request(output_dir=tmp_path / "out", overwrite=True))
    forced.write(item)
    assert forced.skipped == 0


def test_shapefile_writer_leaves_no_staging_directory_on_failure(tmp_path: Path) -> None:
    writer = ShapefileWriter(request(output_dir=tmp_path / "out"))
    with pytest.raises(Exception):
        writer.write(VectorSlice(variable="lithology", source=tmp_path / "absent.shp"))
    writer.abort()

    root = writer.root
    assert not root.exists() or not any(path.name.endswith(".part") for path in root.iterdir())


def test_shapefile_writer_replaces_a_stale_sidecar(tmp_path: Path) -> None:
    geodatabase = fake_geodatabase(tmp_path)
    staged = tmp_path / "staged" / "lithology.shp"
    vector_to_shapefile(
        geodatabase, staged, layer=SOURCE_LAYER, layer_name="lithology", target_crs="EPSG:4326"
    )
    writer = ShapefileWriter(request(output_dir=tmp_path / "out", overwrite=True))
    stale = writer.root / "glim_lithology.qix"
    writer.root.mkdir(parents=True, exist_ok=True)
    (writer.root / "glim_lithology.cpg").write_text("LATIN1", encoding="utf-8")
    stale.write_text("stale index", encoding="utf-8")

    writer.write(VectorSlice(variable="lithology", source=staged))

    assert (writer.root / "glim_lithology.cpg").read_text().strip() == "UTF-8"
    # Files outside the known sidecar set are left alone rather than guessed at.
    assert stale.is_file()
    assert stale not in writer.close()


def test_shapefile_writer_refuses_two_layers_for_one_variable(tmp_path: Path) -> None:
    geodatabase = fake_geodatabase(tmp_path)
    staged = tmp_path / "staged" / "lithology.shp"
    vector_to_shapefile(
        geodatabase, staged, layer=SOURCE_LAYER, layer_name="lithology", target_crs="EPSG:4326"
    )
    writer = ShapefileWriter(request(output_dir=tmp_path / "out"))
    item = VectorSlice(variable="lithology", source=staged)
    writer.write(item)

    with pytest.raises(OutputError, match="written twice"):
        writer.write(item)


def test_clipping_without_a_target_crs_is_refused(tmp_path: Path) -> None:
    geodatabase = fake_geodatabase(tmp_path)
    with pytest.raises(OutputError, match="explicit target CRS"):
        vector_to_shapefile(
            geodatabase,
            tmp_path / "clipped.shp",
            layer=SOURCE_LAYER,
            clip_bbox=BBox(79, 26, 90, 31),
        )


def test_an_extent_without_polygons_is_reported(tmp_path: Path) -> None:
    geodatabase = fake_geodatabase(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(DownloadError, match="No GLiM lithology polygons"):
        list(
            GlimProvider().slices(
                request(bbox=BBox(-40, -40, -30, -30), options={"local_source": geodatabase}),
                workspace,
                null_progress,
                CancellationToken(),
            )
        )


def test_make_writer_selects_the_shapefile_writer(tmp_path: Path) -> None:
    assert isinstance(make_writer(request(output_dir=tmp_path)), ShapefileWriter)


def test_service_writes_a_shapefile_set_and_records_every_member(
    tmp_path: Path, monkeypatch
) -> None:
    geodatabase = fake_geodatabase(tmp_path)
    monkeypatch.setattr(service_module, "get_provider", lambda _identifier: GlimProvider())
    item = request(
        output_dir=tmp_path / "out",
        options={"local_source": geodatabase, "cache_source": False},
    )

    result = DownloadService().download(item)

    assert [path.suffix for path in result.outputs] == [
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
    ]
    assert validate_vector(result.outputs[0])
    manifest = json.loads(result.manifest.read_text())
    assert {Path(entry["path"]).name for entry in manifest["outputs"]} == {
        path.name for path in result.outputs
    }
    assert manifest["provider"]["source_layer"] == SOURCE_LAYER
    assert manifest["provider"]["primary_classes"]["sc"] == "carbonate sedimentary rocks"
    assert manifest["request"]["format"] == "shp"


def test_service_resumes_a_complete_shapefile_request_without_staging(
    tmp_path: Path, monkeypatch
) -> None:
    geodatabase = fake_geodatabase(tmp_path)
    calls: list[Path] = []

    class CountingProvider(GlimProvider):
        def slices(self, item, workspace, progress, cancel):
            calls.append(workspace)
            yield from super().slices(item, workspace, progress, cancel)

    provider = CountingProvider()
    monkeypatch.setattr(service_module, "get_provider", lambda _identifier: provider)
    item = request(
        output_dir=tmp_path / "out",
        options={"local_source": geodatabase, "cache_source": False},
    )

    first = DownloadService().download(item)
    resumed = DownloadService().download(item)

    assert len(calls) == 1
    assert resumed.outputs == first.outputs
    assert resumed.skipped == 1

    forced = DownloadService().download(
        DownloadRequest(**{**item.__dict__, "overwrite": True})
    )
    assert len(calls) == 2
    assert forced.skipped == 0


def test_service_does_not_resume_an_incomplete_shapefile_set(
    tmp_path: Path, monkeypatch
) -> None:
    geodatabase = fake_geodatabase(tmp_path)
    monkeypatch.setattr(service_module, "get_provider", lambda _identifier: GlimProvider())
    item = request(
        output_dir=tmp_path / "out",
        options={"local_source": geodatabase, "cache_source": False},
    )
    first = DownloadService().download(item)
    next(path for path in first.outputs if path.suffix == ".dbf").unlink()

    resumed = DownloadService().download(item)

    assert resumed.skipped == 0
    assert all(path.is_file() for path in resumed.outputs)


def test_cached_source_is_reused_by_a_later_extent(tmp_path: Path) -> None:
    geodatabase = fake_geodatabase(tmp_path)
    archive = tmp_path / "LiMW_GIS-2015.gdb.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for member in sorted(geodatabase.iterdir()):
            bundle.write(member, f"LiMW_GIS 2015.gdb/{member.name}")
    cache = tmp_path / "cache"
    provider = GlimProvider()
    downloads: list[str] = []

    def fake_download(url, destination, **_kwargs):
        downloads.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read_bytes())
        return destination

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(glim_module, "stream_download", fake_download)
    try:
        for bbox in (BBox(79, 26, 90, 31), BBox(79, 26, 85, 30)):
            workspace = tmp_path / f"workspace-{bbox.east}"
            workspace.mkdir()
            item = DownloadRequest(
                dataset="glim",
                bbox=bbox,
                variables=("lithology",),
                output_format=OutputFormat.SHP,
                output_dir=tmp_path / "out",
                cache_dir=cache,
            )
            list(provider.slices(item, workspace, null_progress, CancellationToken()))
    finally:
        monkeypatched.undo()

    assert len(downloads) == 1
    assert validate_vector(cache / "glim" / GEODATABASE_NAME, layer=SOURCE_LAYER)
