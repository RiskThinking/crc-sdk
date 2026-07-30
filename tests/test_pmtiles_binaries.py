import pytest

from crc_sdk.geometry.pmtiles.binaries import require_tile_join, require_tippecanoe


def test_require_tippecanoe_returns_resolved_path_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/local/bin/tippecanoe" if name == "tippecanoe" else None,
    )
    assert require_tippecanoe() == "/usr/local/bin/tippecanoe"


def test_require_tippecanoe_raises_with_install_hint_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError) as excinfo:
        require_tippecanoe()
    assert "tippecanoe" in str(excinfo.value)
    assert "brew install tippecanoe" in str(excinfo.value)


def test_require_tile_join_returns_resolved_path_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/local/bin/tile-join" if name == "tile-join" else None,
    )
    assert require_tile_join() == "/usr/local/bin/tile-join"


def test_require_tile_join_raises_with_install_hint_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError) as excinfo:
        require_tile_join()
    assert "tile-join" in str(excinfo.value)
    assert "brew install tippecanoe" in str(excinfo.value)
