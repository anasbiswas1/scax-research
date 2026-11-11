"""Raw BRFSS rebuild: per-year names, recodes, code checks, zip and XPT reading, and the G0 comparison."""
import zipfile

import numpy as np
import pandas as pd
import pytest

from scax.data import brfss_raw as br


def _raw_year(year, n=3000, seed=0):
    """Synthetic raw records in CDC coding for the names used in `year`, with missing and refusal codes."""
    rng = np.random.default_rng(seed)
    cols = {}
    for ind in br.INDICATORS + [br.TARGET]:
        name = br.NAMES[year].get(ind)
        if name is None:
            continue
        mapping, missing = br.rule(ind, name)
        if mapping is None:
            v = rng.integers(1500, 5000, n).astype(float)
        else:
            codes = list(mapping) + sorted(missing)
            p = np.r_[np.full(len(mapping), 0.95 / len(mapping)), np.full(len(missing), 0.05 / max(1, len(missing)))]
            v = rng.choice(codes, n, p=p / p.sum()).astype(float)
        v[rng.random(n) < 0.03] = np.nan
        cols[name] = v
    cols["_STATE"] = rng.integers(1, 57, n).astype(float)
    cols["_LLCPWT"] = rng.random(n) * 100
    cols["OTHER"] = rng.random(n)
    return pd.DataFrame(cols)


def _reference(raw, year):
    """Plain row-by-row reading of the recipe, written independently of br.build."""
    names = {k: v for k, v in br.NAMES[year].items() if v is not None}
    rows = []
    for _, r in raw.iterrows():
        if any(pd.isna(r[n]) for n in names.values()):
            continue
        out, ok = {}, True
        for ind, n in names.items():
            v = r[n]
            if ind == "BMI":
                out[ind] = float(np.round(v / 100))
                continue
            mapping, missing = br.rule(ind, n)
            if v in missing:
                ok = False
                break
            out[ind] = float(mapping[int(v)])
        if ok:
            rows.append(out)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("year", sorted(br.NAMES))
def test_build_matches_plain_reading(year):
    raw = _raw_year(year, n=1500, seed=year)
    table, drops = br.build(raw, year)
    ref = _reference(raw, year)
    cols = [c for c in br.INDICATORS if br.NAMES[year].get(c)] + [br.TARGET]
    assert len(table) == len(ref) == drops.records.iloc[-1]
    assert np.array_equal(table[cols].to_numpy(), ref[cols].to_numpy())
    assert (drops.records.diff().dropna() <= 0).all()
    if year == 2023:
        assert "Fruits" not in table and "Veggies" not in table


def test_income3_collapse_and_unexpected_codes():
    raw = _raw_year(2021, n=500, seed=1)
    raw.loc[raw.index[:20], "INCOME3"] = 11.0
    table, _ = br.build(raw, 2021)
    assert table.Income.max() <= 8
    raw.loc[raw.index[0], "_RFHYPE6"] = 5.0
    with pytest.raises(ValueError, match="unexpected codes"):
        br.build(raw, 2021)
    with pytest.raises(KeyError, match="not in the file"):
        br.build(raw.drop(columns=["TOLDHI3"]), 2021)


def test_zip_with_trailing_space_and_xpt_roundtrip(tmp_path):
    import pyreadstat
    raw = _raw_year(2015, n=200, seed=3)
    xpt = tmp_path / "plain.xpt"
    pyreadstat.write_xport(raw, str(xpt), table_name="LLCP2015")
    z = tmp_path / "LLCP2015XPT.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.write(xpt, arcname="LLCP2015.XPT ")
    out = br.extract_xpt(z, tmp_path)
    assert br.year_of(z) == 2015 and out.exists()
    cols = br.xpt_columns(out)
    assert "_RFHYPE5" in cols and "OTHER" in cols
    back = br.read_xpt(out, ["_RFHYPE5", "_BMI5", "_STATE"])
    assert list(back.columns) == ["_RFHYPE5", "_BMI5", "_STATE"]
    assert np.allclose(back._BMI5.fillna(-1), raw._BMI5.fillna(-1))


def test_compare_with_public():
    raw = _raw_year(2015, n=800, seed=4)
    table, _ = br.build(raw, 2015)
    X = table[br.INDICATORS].copy(); y = table[br.TARGET].astype(int).to_numpy()
    res = br.compare_with_public(table, X, y)
    assert res["passed"] and res["row_by_row_equal"]
    X2 = X.copy(); X2.loc[X2.index[0], "Income"] = 1 if X2.Income.iloc[0] != 1 else 2
    res = br.compare_with_public(table, X2, y)
    assert not res["passed"] and res["marginal_mismatches"] == ["Income"]


def test_xpt_with_latin1_label(tmp_path):
    import pyreadstat
    raw = _raw_year(2023, n=50, seed=5)
    xpt = tmp_path / "LLCP2023.XPT"
    pyreadstat.write_xport(raw, str(xpt), table_name="LLCP2023", column_labels=["LABELXYZ"] + [""] * (raw.shape[1] - 1))
    b = xpt.read_bytes()
    assert b.count(b"LABELXYZ") == 1
    xpt.write_bytes(b.replace(b"LABELXYZ", b"LABE\xb4XYZ"))
    with pytest.raises(UnicodeDecodeError):
        pyreadstat.read_xport(str(xpt), metadataonly=True)
    assert br.xpt_columns(xpt) == list(raw.columns)
    back = br.read_xpt(xpt, ["_RFHYPE6", "_BMI5"])
    assert np.allclose(back._BMI5.fillna(-1), raw._BMI5.fillna(-1))
