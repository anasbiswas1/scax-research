"""Rebuilding the BRFSS indicator tables from CDC's raw annual files (plan Section 6.5).

The 2015 recipe is the one behind the public 2015 table: keep respondents with every selected item
present, drop "don't know", "refused" and "missing" codes, and recode to the public coding. Later
years use the same recipe with each year's variable names. The names were checked against CDC's
variable layouts for 2015, 2016, 2017, 2019, 2021 and 2023: even years do not ask about blood
pressure, cholesterol or fruit and vegetables, and 2023 does not ask about fruit and vegetables.
Any raw code outside the expected set stops the build, so a renamed or recoded item cannot pass
silently.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

INDICATORS = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes", "PhysActivity",
              "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
              "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
TARGET = "HeartDiseaseorAttack"
EXTRA = ["_STATE", "_LLCPWT"]          # kept for the geographic shift and for reference; never predictors

_BASE = {"HighBP": "_RFHYPE5", "HighChol": "TOLDHI2", "CholCheck": "_CHOLCHK", "BMI": "_BMI5", "Smoker": "SMOKE100",
         "Stroke": "CVDSTRK3", "Diabetes": "DIABETE3", "PhysActivity": "_TOTINDA", "Fruits": "_FRTLT1",
         "Veggies": "_VEGLT1", "HvyAlcoholConsump": "_RFDRHV5", "AnyHealthcare": "HLTHPLN1", "NoDocbcCost": "MEDCOST",
         "GenHlth": "GENHLTH", "MentHlth": "MENTHLTH", "PhysHlth": "PHYSHLTH", "DiffWalk": "DIFFWALK", "Sex": "SEX",
         "Age": "_AGEG5YR", "Education": "EDUCA", "Income": "INCOME2", TARGET: "_MICHD"}
NAMES = {
    2015: dict(_BASE),
    2017: {**_BASE, "CholCheck": "_CHOLCH1", "Fruits": "_FRTLT1A", "Veggies": "_VEGLT1A"},
    2019: {**_BASE, "CholCheck": "_CHOLCH2", "Diabetes": "DIABETE4", "Fruits": "_FRTLT1A", "Veggies": "_VEGLT1A",
           "HvyAlcoholConsump": "_RFDRHV7", "Sex": "SEXVAR"},
    2021: {**_BASE, "HighBP": "_RFHYPE6", "HighChol": "TOLDHI3", "CholCheck": "_CHOLCH3", "Diabetes": "DIABETE4",
           "Fruits": "_FRTLT1A", "Veggies": "_VEGLT1A", "HvyAlcoholConsump": "_RFDRHV7", "AnyHealthcare": "_HLTHPLN",
           "NoDocbcCost": "MEDCOST1", "Sex": "SEXVAR", "Income": "INCOME3"},
    2023: {**_BASE, "HighBP": "_RFHYPE6", "HighChol": "TOLDHI3", "CholCheck": "_CHOLCH3", "Diabetes": "DIABETE4",
           "Fruits": None, "Veggies": None, "HvyAlcoholConsump": "_RFDRHV8", "AnyHealthcare": "_HLTHPL1",
           "NoDocbcCost": "MEDCOST1", "Sex": "SEXVAR", "Income": "INCOME3"},
}

YES1 = {1: 1, 2: 0}            # 1 yes, 2 no
RISK2 = {1: 0, 2: 1}           # calculated risk factors: 1 no, 2 yes
# indicator -> (mapping from kept raw codes to public codes, or None for numeric; codes dropped)
RULES = {
    "HighBP": (RISK2, {9}), "HighChol": (YES1, {7, 9}), "CholCheck": ({1: 1, 2: 0, 3: 0}, {9}),
    "BMI": (None, set()), "Smoker": (YES1, {7, 9}), "Stroke": (YES1, {7, 9}),
    "Diabetes": ({1: 2, 2: 0, 3: 0, 4: 1}, {7, 9}), "PhysActivity": (YES1, {9}), "Fruits": (YES1, {9}),
    "Veggies": (YES1, {9}), "HvyAlcoholConsump": (RISK2, {9}), "AnyHealthcare": (YES1, {7, 9}),
    "NoDocbcCost": (YES1, {7, 9}), "GenHlth": ({i: i for i in range(1, 6)}, {7, 9}),
    "MentHlth": ({**{i: i for i in range(1, 31)}, 88: 0}, {77, 99}),
    "PhysHlth": ({**{i: i for i in range(1, 31)}, 88: 0}, {77, 99}), "DiffWalk": (YES1, {7, 9}),
    "Sex": (YES1, {7, 9}), "Age": ({i: i for i in range(1, 14)}, {14}), "Education": ({i: i for i in range(1, 7)}, {9}),
    "Income": ({i: i for i in range(1, 9)}, {77, 99}), TARGET: (YES1, set()),
}
INCOME3_COLLAPSE = {**{i: i for i in range(1, 9)}, 9: 8, 10: 8, 11: 8}   # 11 levels -> the 8 INCOME2 levels


def rule(indicator, raw_name):
    if indicator == "Income" and raw_name == "INCOME3":
        return INCOME3_COLLAPSE, {77, 99}
    return RULES[indicator]


def year_of(path) -> int:
    m = re.search(r"LLCP(\d{4})", Path(path).name.upper())
    if not m:
        raise ValueError(f"cannot read the year from {path}")
    return int(m.group(1))


def extract_xpt(zip_path, workdir) -> Path:
    """The single .XPT member of a CDC zip (2015 stores it with a trailing space in its name)."""
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.strip().upper().endswith(".XPT")]
        if len(members) != 1:
            raise ValueError(f"{zip_path}: expected one .XPT file, found {members}")
        out = Path(workdir) / f"LLCP{year_of(zip_path)}.XPT"
        with z.open(members[0]) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
    return out


def _read(xpt_path, **kw):
    """pyreadstat reader; files whose variable labels are not valid UTF-8 (2023 has a Latin-1 byte)
    are read as Latin-1. Labels are not used, column names are ASCII and values are numeric."""
    import pyreadstat
    try:
        return pyreadstat.read_xport(str(xpt_path), **kw)
    except UnicodeDecodeError:
        return pyreadstat.read_xport(str(xpt_path), encoding="latin1", **kw)


def xpt_columns(xpt_path):
    _, meta = _read(xpt_path, metadataonly=True)
    return list(meta.column_names)


def read_xpt(xpt_path, columns):
    df, _ = _read(xpt_path, usecols=list(columns))
    return df


def presence(year, columns):
    """Which raw name each indicator uses in this year, checked against the file's own columns."""
    names = NAMES[year]
    cols = set(columns)
    out = {}
    for ind in INDICATORS + [TARGET]:
        n = names.get(ind)
        if n is None:
            out[ind] = None
        elif n in cols:
            out[ind] = n
        else:
            raise KeyError(f"{year}: {ind} is expected as {n}, which is not in the file")
    return out


def build(raw, year):
    """Indicator table in the public coding, plus counts of records removed at each step."""
    used = {ind: n for ind, n in presence(year, raw.columns).items() if n is not None}
    drops = [("raw records", len(raw))]
    keep = np.array(raw[list(used.values())].notna().all(axis=1), dtype=bool)   # writable copy
    drops.append(("every selected item present", int(keep.sum())))
    out = pd.DataFrame(index=raw.index[keep])
    for ind, name in used.items():
        mapping, missing = rule(ind, name)
        v = raw.loc[keep, name].to_numpy(dtype=float)
        if mapping is None:
            if np.any(v <= 0):
                raise ValueError(f"{year}: {name} has non-positive values")
            out[ind] = np.round(v / 100.0)               # BMI x 100 -> BMI rounded half to even, as in the public table
            continue
        allowed = set(mapping) | set(missing)
        bad = sorted(set(np.unique(raw[name].dropna().to_numpy(dtype=float))) - allowed)   # every record, not only kept ones
        if bad:
            raise ValueError(f"{year}: {name} has unexpected codes {bad[:10]}")
        out[ind] = pd.Series(v, index=out.index).map(mapping).to_numpy()
    for ind, name in used.items():
        before = int(keep.sum())
        mapping, missing = rule(ind, name)
        if missing:
            ok = ~raw.loc[keep, name].isin(list(missing)).to_numpy()
            keep_idx = np.flatnonzero(keep)
            keep[keep_idx[~ok]] = False
        drops.append((f"{ind} ({name}) codes {sorted(missing)} removed" if missing else f"{ind} ({name})",
                      int(keep.sum())))
    out = out.loc[raw.index[keep]].astype(float)
    for c in EXTRA:
        if c in raw.columns:
            out[c] = raw.loc[raw.index[keep], c].to_numpy()
    cols = [c for c in INDICATORS if c in used] + [TARGET] + [c for c in EXTRA if c in out.columns]
    return out[cols].reset_index(drop=True), pd.DataFrame(drops, columns=["step", "records"])


def compare_with_public(rebuilt, public_X, public_y):
    """G0: rows, every marginal frequency, and row-by-row equality against the public 2015 table."""
    cols = list(public_X.columns)
    res = {"rows_rebuilt": len(rebuilt), "rows_public": len(public_X), "marginal_mismatches": []}
    for c in cols + [TARGET]:
        a = rebuilt[c].value_counts().sort_index()
        b = (public_X[c] if c != TARGET else pd.Series(public_y)).astype(float).value_counts().sort_index()
        if not a.equals(b.astype(a.dtype)) or not np.array_equal(a.index.to_numpy(), b.index.to_numpy()):
            res["marginal_mismatches"].append(c)
    res["rows_equal"] = res["rows_rebuilt"] == res["rows_public"]
    res["row_by_row_equal"] = bool(res["rows_equal"] and np.array_equal(
        rebuilt[cols].to_numpy(dtype=float), public_X[cols].to_numpy(dtype=float))
        and np.array_equal(rebuilt[TARGET].to_numpy(dtype=float), np.asarray(public_y, dtype=float)))
    res["passed"] = bool(res["rows_equal"] and not res["marginal_mismatches"])
    return res
