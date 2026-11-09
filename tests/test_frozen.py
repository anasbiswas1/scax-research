import pytest

from scax.frozen import load_frozen, write_frozen


def test_frozen_round_trip_and_tamper(tmp_path):
    p = tmp_path / "configs" / "task.json"
    h = write_frozen(p, {"b": 1, "a": [0.05, 0.7]})
    cfg, h2 = load_frozen(p, h)
    assert h == h2 and cfg["a"] == [0.05, 0.7]
    p.write_text(p.read_text().replace("0.05", "0.06"))
    with pytest.raises(ValueError):
        load_frozen(p, h)
