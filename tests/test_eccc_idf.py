from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.connectors.adapters import CurveFitIngestPolicy
from crc_sdk.connectors.parquet import read_hazard_metadata
from crc_sdk.providers import eccc_idf as provider
from crc_sdk.workflows import (
    HazardDataset,
    curve_quantiles,
    return_periods_to_probabilities,
)

PERIODS = [2, 5, 10, 25, 50, 100]
STATION = "6158731"


def station_text(station: str = STATION, start: int = 1950) -> bytes:
    lines = [
        f"SYNTHETIC A                       ON        {station}",
        "Latitude:  43 41'N    Longitude: 79 38'W    Elevation/Altitude: 173 m",
        f"Years/Années : {start} - 2017   # Years/Années : 40",
        "Table 1 : Annual Maximum (mm)",
        "1950 -99.9 -99.9 -99.9",
        "Table 2a : Return Period Rainfall Amounts (mm)",
        "1 h 9999 9999 9999 9999 9999 9999 40",
        "Table 2b :",
        "Return Period Rainfall Rates (mm/h) - 95% Confidence limits",
        "Duration/Durée        2        5       10       25       50      100   #Years",
        "yr/ans yr/ans yr/ans yr/ans yr/ans yr/ans Années",
    ]
    for duration, minutes in provider.DURATIONS.items():
        rates = [
            round((20 - 5 * math.log(-math.log(1 - 1 / t))) * 60 / minutes, 1)
            for t in PERIODS
        ]
        label = duration.replace("min", " min").replace("h", " h")
        lines.append(f"{label} " + " ".join(map(str, rates)) + " 40")
        lines.append(" ".join(["+/- 0.5"] * len(PERIODS)) + " 40")
    lines.append("Table 3 : Interpolation Equation")
    return "\r\n".join(lines).encode("cp1252")


def fake_fetch(version: str, province: str, stations: Any) -> Any:
    for station in stations or [STATION, "6158355"]:
        raw = station_text(station, 1950 if station == STATION else 1960)
        yield (
            station,
            raw,
            {
                "url": provider.archive_url(version, province),
                "member": f"_{province}_{station}_SYNTHETIC.txt",
                "checksum": hashlib.sha256(raw).hexdigest(),
            },
        )


def test_parser_labels_rates_confidence_and_record() -> None:
    record = provider.parse_station(station_text(), STATION, "ON")
    assert record["return_periods"] == PERIODS
    assert record["latitude"] == pytest.approx(43 + 41 / 60)
    assert record["longitude"] == pytest.approx(-79 - 38 / 60)
    assert record["curves"][4]["intensity"][0] < 30
    assert record["curves"][4]["confidence_half_width"] == [0.5] * 6
    assert record["curves"][4]["record_years"] == 40
    assert record["curves"][-1]["duration_minutes"] == 1440


@pytest.mark.parametrize(
    "old,new",
    [
        ("      100   #Years", "      50   #Years"),
        ("Table 2b :", "Missing table"),
        ("1 h ", "3 h "),
        ("+/- 0.5", "+/- -0.5"),
        ("43 41'N", "43 99'N"),
        (" 40\r\n", " 400\r\n"),
    ],
)
def test_malformed_scientific_input_rejected(old: str, new: str) -> None:
    raw = station_text().decode("cp1252").replace(old, new).encode("cp1252")
    with pytest.raises(ValueError):
        provider.parse_station(raw, STATION, "ON")


def test_lazy_immutable_selection_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider, "fetch_stations", lambda *a: pytest.fail("I/O"))
    base = HazardDataset.eccc_idf("1h", province="ON")
    plan = base.for_stations([STATION]).duration("24h")
    assert base.selected_stations == () and base.duration_name == "1h"
    assert plan.explain(format="json")["unit"] == "mm/h"  # type: ignore[index]
    for args in [
        {"duration": "1d", "province": "ON"},
        {"duration": "1h", "province": "Ontario"},
        {"duration": "1h", "province": "ON", "version": "latest"},
    ]:
        with pytest.raises(ValueError):
            HazardDataset.eccc_idf(**args)
    with pytest.raises(TypeError):
        base.for_stations(STATION)
    with pytest.raises(ValueError):
        base.for_stations([STATION, STATION])
    with pytest.raises(ValueError):
        base.for_area((0, 0, -1, 1))
    with pytest.raises(ValueError, match="quantile"):
        base.canonicalize(
            policy=CurveFitIngestPolicy(
                h3_resolution=5,
                family="gumbel_r",
                producer="test",
                fit_method="sample_mle",
            )
        )
    with pytest.raises(ValueError):
        base.cache(None)


def test_area_union_is_lazy_normalized_and_replaces_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider, "fetch_stations", lambda *a: pytest.fail("I/O"))
    base = HazardDataset.eccc_idf("1h", province="ON").for_stations([STATION])
    west = (-80, 43, -79, 44)
    east = (-78, 45, -77, 46)
    plan = base.for_areas([east, west, east])
    assert base.selected_stations == (STATION,) and base.requested_areas == ()
    assert plan.selected_stations == () and plan.requested_areas == (west, east)
    assert plan.explain(format="json")["areas"] == (west, east)  # type: ignore[index]
    assert plan == base.for_areas([west, east])
    assert plan.for_area(west) == base.for_areas([west])
    assert plan.for_stations([STATION]) == base
    with pytest.raises(ValueError, match="at least one"):
        base.for_areas([])
    with pytest.raises(ValueError, match="ordered"):
        base.for_areas([west, (0, 0, -1, 1)])


def test_disjoint_overlapping_areas_share_acquisition_and_offline_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fetch(version: str, province: str, stations: Any) -> Any:
        calls.append((version, province))
        assert not stations
        for station, latitude, longitude in (
            (STATION, "43 41'N", "79 38'W"),
            ("6158355", "45 00'N", "78 00'W"),
            ("6150001", "44 30'N", "78 30'W"),
            ("6150002", "47 00'N", "76 00'W"),
        ):
            raw = station_text(station).replace(b"43 41'N", latitude.encode())
            raw = raw.replace(b"79 38'W", longitude.encode())
            yield station, raw, {"checksum": hashlib.sha256(raw).hexdigest()}

    monkeypatch.setattr(provider, "fetch_stations", fetch)
    west, east = (-80, 43, -79, 44), (-78, 45, -77, 46)
    plan = (
        HazardDataset.eccc_idf("1h", province="ON")
        .for_areas([east, west, west, (-79.9, 43.1, -79.1, 43.9)])
        .cache(tmp_path / "cache")
    )
    result = plan.prefetch()
    assert result.resources == 2 and result.cache_misses == 4
    assert calls == [("3.40", "ON")]
    monkeypatch.setattr(provider, "fetch_stations", lambda *a: pytest.fail("network"))
    offline = plan.cache(tmp_path / "cache", mode="offline")
    source = offline.source_curves()
    assert sorted(source["station"].to_pylist()) == ["6158355", STATION]
    output = tmp_path / "curves.parquet"
    offline.materialize(output)
    assert pq.ParquetFile(output).metadata.num_rows == 2
    assert offline.for_area(east).source_curves()["station"].to_pylist() == ["6158355"]
    assert offline.for_area(west).prefetch().cache_misses == 0
    with pytest.raises(ValueError, match="no ECCC station"):
        offline.for_areas([(0, 0, 1, 1)]).source_curves()


def test_cache_duration_reuse_offline_refresh_and_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider, "fetch_stations", fake_fetch)
    plan = (
        HazardDataset.eccc_idf("1h", province="ON")
        .for_stations([STATION])
        .cache(tmp_path)
    )
    assert plan.prefetch().cache_misses == 1
    assert plan.duration("24h").prefetch().cache_hits == 1
    monkeypatch.setattr(provider, "fetch_stations", lambda *a: pytest.fail("network"))
    offline = plan.cache(tmp_path, mode="offline")
    assert offline.source_curves().num_rows == 1
    raw = tmp_path / "eccc-idf-3.40-ON" / f"{STATION}.txt"
    raw.write_bytes(b"corrupt")
    with pytest.raises(FileNotFoundError, match="corrupt"):
        offline.prefetch()
    monkeypatch.setattr(provider, "fetch_stations", fake_fetch)
    assert plan.prefetch().cache_misses == 1
    assert plan.cache(tmp_path, mode="refresh").prefetch().cache_misses == 1
    manifest = raw.parent / "manifest.json"
    data = json.loads(manifest.read_text())
    data["entries"][STATION]["checksum"] = "different pinned bytes"
    manifest.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="changed"):
        plan.prefetch()
    with pytest.raises(FileNotFoundError):
        plan.cache(tmp_path, mode="offline").for_stations(["6158355"]).prefetch()


def test_area_mixed_records_point_geometry_fit_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h3  # type: ignore[import-untyped]
    from shapely import wkb  # type: ignore[import-untyped]

    monkeypatch.setattr(provider, "fetch_stations", fake_fetch)
    diagnostics = tmp_path / "diagnostics.parquet"
    policy = CurveFitIngestPolicy(
        h3_resolution=5,
        family="gumbel_r",
        producer="test",
        maximum_absolute_residual=0.15,
        diagnostics=diagnostics,
    )
    plan = (
        HazardDataset.eccc_idf("1h", province="ON")
        .for_area((-80, 43, -79, 44))
        .cache(tmp_path / "cache")
        .canonicalize(policy=policy)
    )
    output = tmp_path / "curves.parquet"
    plan.materialize(output)
    table = pq.ParquetFile(output).read()
    rows = table.to_pylist()
    assert len(rows) == 2
    assert {r["horizon"] for r in rows} == {1983, 1988}
    assert set(pq.read_table(diagnostics)["horizon"].to_pylist()) == {1983, 1988}
    assert len({r["cell_index"] for r in rows}) == 1
    assert rows[0]["cell_index"] == h3.str_to_int(
        h3.latlng_to_cell(43 + 41 / 60, -79 - 38 / 60, 5)
    )
    assert all(wkb.loads(r["source_geometry"]).geom_type == "Point" for r in rows)
    metadata = read_hazard_metadata(output)
    assert metadata.temporal_window is None
    assert metadata.probability_semantics == "annual_exceedance"
    assert metadata.source_return_period_convention == "one_minus_inverse"
    assert metadata.source.licence == provider.LICENCE
    assert (
        metadata.fitting is not None
        and metadata.fitting.input_kind == "probability_labelled"
    )
    values = curve_quantiles(table, return_periods_to_probabilities(PERIODS))
    expected = plan.source_curves().to_pylist()[0]["intensity"]
    assert values[0] == pytest.approx(expected, abs=0.15)
    single = plan.for_stations([STATION])
    single.materialize(tmp_path / "single.parquet")
    window = read_hazard_metadata(tmp_path / "single.parquet").temporal_window
    assert window is not None and window.start_year == 1950 and window.end_year == 2017
    assert single.ensure_materialized().provider is not None


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    return buffer.getvalue()


@pytest.mark.parametrize("version", ["3.20", "3.40"])
def test_real_zip_range_access_without_network(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    raw = station_text()
    inner = zip_bytes(
        {f"province/_ON_{STATION}_SYNTHETIC.txt": raw, "ignored.png": b"not text"}
    )
    data = (
        zip_bytes({"idf_v3-40_2025-12-5/IDF_Files_Fichiers/ON.zip": inner})
        if version == "3.40"
        else inner
    )
    ranges: list[tuple[int, int]] = []

    class Response(io.BytesIO):
        def __init__(
            self, payload: bytes, headers: dict[str, str], status: int
        ) -> None:
            super().__init__(payload)
            self.headers = headers
            self.status = status

    def urlopen(request: Any, **kwargs: Any) -> Any:
        if request.get_method() == "HEAD":
            return Response(
                b"", {"Content-Length": str(len(data)), "ETag": '"test"'}, 200
            )
        assert request.get_header("If-match") == '"test"'
        start, end = map(int, request.get_header("Range").split("=")[1].split("-"))
        ranges.append((start, end))
        return Response(
            data[start : end + 1],
            {"Content-Range": f"bytes {start}-{end}/{len(data)}"},
            206,
        )

    monkeypatch.setattr(provider, "urlopen", urlopen)
    fetched = list(provider.fetch_stations(version, "ON", [STATION]))
    assert len(fetched) == 1 and fetched[0][1] == raw
    assert fetched[0][2]["checksum"] == hashlib.sha256(raw).hexdigest()
    assert len(ranges) > 1
    with pytest.raises(ValueError, match="no stations"):
        list(provider.fetch_stations(version, "ON", ["9999999"]))
