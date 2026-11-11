"""NHANES 2015-2016 and 2017-2018 in the BRFSS indicator coding (plan Sections 6.5 and 6.6, measurement shift).

Population: adults aged 20 and over (the age range of the medical-conditions and education items).
Outcome: ever told coronary heart disease (MCQ160C) or a heart attack (MCQ160E), built like BRFSS _MICHD.
Self-reported predictors follow the BRFSS questions as closely as the NHANES instruments allow; the rules
and their comparability are listed in VARIABLE_MAP. Measured BMI (examination), measured blood pressure
(mean of the examination readings, 140/90) and measured total cholesterol (240 mg/dL) are added for the
measurement shift.

Not available in NHANES: days of poor physical and mental health (HSQ470 and HSQ480 were dropped after
2012), "could not see a doctor because of cost" (not asked) and fruit and vegetables (no comparable item).
The common predictor set therefore has 16 predictors. Education and income are coarsened to the levels
both surveys can express, and brfss_common applies the same coarsening to BRFSS tables.

Every categorical item is checked against its documented codes on every record, so a recoded item
stops the build instead of passing silently.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CYCLES = {2015: "I", 2017: "J"}
COMPONENTS = ["DEMO", "BPQ", "BPX", "BMX", "TCHOL", "WHQ", "SMQ", "MCQ", "DIQ", "PAQ", "ALQ", "HIQ", "HUQ", "PFQ"]
TARGET = "HeartDiseaseorAttack"
N_PREDICTORS = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes", "PhysActivity",
                "HvyAlcoholConsump", "AnyHealthcare", "GenHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
MEASURED = {"BMI": "BMI_measured", "HighBP": "HighBP_measured", "HighChol": "HighChol_measured"}

_SY = [f"BPXSY{i}" for i in range(1, 5)]
_DI = [f"BPXDI{i}" for i in range(1, 5)]
VARIABLES = {
    "DEMO": ["SEQN", "RIAGENDR", "RIDAGEYR", "DMDEDUC2", "INDHHIN2", "WTMEC2YR"],
    "BPQ": ["SEQN", "BPQ020", "BPQ060", "BPQ070", "BPQ080"],
    "BPX": ["SEQN"] + _SY + _DI,
    "BMX": ["SEQN", "BMXBMI"],
    "TCHOL": ["SEQN", "LBXTC"],
    "WHQ": ["SEQN", "WHD010", "WHD020"],
    "SMQ": ["SEQN", "SMQ020"],
    "MCQ": ["SEQN", "MCQ160C", "MCQ160E", "MCQ160F"],
    "DIQ": ["SEQN", "DIQ010", "DIQ160"],
    "PAQ": ["SEQN", "PAQ635", "PAQ650", "PAQ665"],
    "ALQ": {2015: ["SEQN", "ALQ101", "ALQ110", "ALQ120Q", "ALQ120U", "ALQ130"],
            2017: ["SEQN", "ALQ111", "ALQ121", "ALQ130"]},
    "HIQ": ["SEQN", "HIQ011"],
    "HUQ": ["SEQN", "HUQ010"],
    "PFQ": ["SEQN", "PFQ049", "PFQ051", "PFQ054", "PFQ057", "PFQ059", "PFQ061B"],
}

_YN = {1, 2, 7, 9}
CODES = {
    "RIAGENDR": {1, 2}, "DMDEDUC2": {1, 2, 3, 4, 5, 7, 9},
    "INDHHIN2": set(range(1, 11)) | {12, 13, 14, 15, 77, 99},
    "BPQ020": _YN, "BPQ060": _YN, "BPQ070": {1, 2, 3, 4, 7, 9}, "BPQ080": _YN, "SMQ020": _YN,
    "MCQ160C": _YN, "MCQ160E": _YN, "MCQ160F": _YN, "DIQ010": {1, 2, 3, 7, 9}, "DIQ160": _YN,
    "PAQ635": _YN, "PAQ650": _YN, "PAQ665": _YN, "HIQ011": _YN, "HUQ010": {1, 2, 3, 4, 5, 7, 9},
    "PFQ049": _YN, "PFQ051": _YN, "PFQ054": _YN, "PFQ057": _YN, "PFQ059": _YN, "PFQ061B": {1, 2, 3, 4, 5, 7, 9},
    "ALQ101": _YN, "ALQ110": _YN, "ALQ111": _YN, "ALQ120U": {1, 2, 3, 7, 9}, "ALQ121": set(range(0, 11)) | {77, 99},
}

# coarsened coding shared by both surveys
EDUCATION_FROM_BRFSS = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}          # never attended and grades 1-8 merged
INCOME_FROM_BRFSS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 7}   # $35k to $75k merged
EDUCATION_FROM_NHANES = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
INCOME_FROM_NHANES = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6, 9: 6, 10: 6, 14: 7, 15: 7}
# 2017-2018 drinking frequency categories (ALQ121) as occasions per week
ALQ121_PER_WEEK = {0: 0.0, 1: 7.0, 2: 5.5, 3: 3.5, 4: 2.0, 5: 1.0, 6: 2.5 * 12 / 52, 7: 12 / 52, 8: 9 / 52,
                   9: 4.5 / 52, 10: 1.5 / 52}
ALQ120U_PER_WEEK = {1: 1.0, 2: 12 / 52, 3: 1 / 52}

VARIABLE_MAP = [
    ("HighBP", "_RFHYPE5 (ever told high blood pressure)", "BPQ020", "BPQ020", "1 yes, 2 no", "same question"),
    ("HighChol", "TOLDHI2 (ever told cholesterol high; asked only if ever checked)", "BPQ080, BPQ060", "BPQ080, BPQ060",
     "1 if BPQ080 = 1; 0 if BPQ080 = 2 and ever checked (BPQ060 = 1); missing if never checked, as in BRFSS", "same question"),
    ("CholCheck", "_CHOLCHK (checked within 5 years)", "BPQ070", "BPQ070", "1 if less than 5 years ago, 0 if 5 years or more",
     "same question"),
    ("BMI", "_BMI5 (self-reported height and weight)", "WHD010, WHD020", "WHD010, WHD020",
     "703.0696 x pounds / inches^2, rounded", "same construction"),
    ("Smoker", "SMOKE100 (at least 100 cigarettes)", "SMQ020", "SMQ020", "1 yes, 2 no", "same question"),
    ("Stroke", "CVDSTRK3 (ever told stroke)", "MCQ160F", "MCQ160F", "1 yes, 2 no", "same question"),
    ("Diabetes", "DIABETE3 (0 no, 1 prediabetes or borderline, 2 diabetes)", "DIQ010, DIQ160", "DIQ010, DIQ160",
     "2 if DIQ010 = 1; 1 if borderline (DIQ010 = 3) or told prediabetes (DIQ160 = 1); 0 otherwise", "close"),
    ("PhysActivity", "_TOTINDA (any exercise outside work, past 30 days)", "PAQ635, PAQ650, PAQ665",
     "PAQ635, PAQ650, PAQ665", "1 if walking or cycling for travel, or vigorous or moderate recreation, in a typical week",
     "approximate (typical week, 10 minutes or more)"),
    ("HvyAlcoholConsump", "_RFDRHV5 (men more than 14, women more than 7 drinks a week, past 30 days)",
     "ALQ101, ALQ110, ALQ120Q, ALQ120U, ALQ130", "ALQ111, ALQ121, ALQ130",
     "drinking days per week x drinks per drinking day, same thresholds", "approximate (past 12 months)"),
    ("AnyHealthcare", "HLTHPLN1 (any health care coverage)", "HIQ011", "HIQ011", "1 yes, 2 no", "same question"),
    ("GenHlth", "GENHLTH (1 excellent to 5 poor)", "HUQ010", "HUQ010", "same five levels", "same question"),
    ("DiffWalk", "DIFFWALK (serious difficulty walking or climbing stairs)", "PFQ054, PFQ061B, PFQ049-PFQ059",
     "PFQ054, PFQ061B, PFQ049-PFQ059",
     "1 if special equipment needed to walk, or much difficulty or unable to walk a quarter mile; 0 if no or some "
     "difficulty, or routed past the questions because no limitation was reported (age 59 or under)", "approximate"),
    ("Sex", "SEX", "RIAGENDR", "RIAGENDR", "1 male, 0 female", "same"),
    ("Age", "_AGEG5YR (13 groups)", "RIDAGEYR", "RIDAGEYR", "20-24 in group 1, then five-year groups, 80 and over in 13",
     "same groups (NHANES starts at 20)"),
    ("Education", "EDUCA (6 levels), coarsened to 5", "DMDEDUC2", "DMDEDUC2",
     "1 less than 9th grade, 2 grades 9-11, 3 high school, 4 some college, 5 college graduate", "same after coarsening"),
    ("Income", "INCOME2 (8 levels), coarsened to 7", "INDHHIN2", "INDHHIN2",
     "1 under $10k, 2 $10k-15k, 3 $15k-20k, 4 $20k-25k, 5 $25k-35k, 6 $35k-75k, 7 $75k and over; "
     "the codes 'under $20k' and '$20k and over' are missing", "same after coarsening"),
    (TARGET, "_MICHD (ever told coronary heart disease or heart attack)", "MCQ160C, MCQ160E", "MCQ160C, MCQ160E",
     "1 if either is yes, 0 if both are no, missing otherwise", "same questions"),
    ("BMI_measured", "(not in BRFSS)", "BMXBMI", "BMXBMI", "examination BMI, rounded", "measurement"),
    ("HighBP_measured", "(not in BRFSS)", "BPXSY1-4, BPXDI1-4", "BPXSY1-4, BPXDI1-4",
     "1 if mean systolic 140 or more or mean diastolic 90 or more (diastolic 0 readings left out)", "measurement"),
    ("HighChol_measured", "(not in BRFSS)", "LBXTC", "LBXTC", "1 if total cholesterol 240 mg/dL or more", "measurement"),
]


def file_stem(component: str, year: int) -> str:
    return f"{component}_{CYCLES[year]}"


def url(component: str, year: int) -> str:
    return f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{file_stem(component, year)}.xpt"


def stem_of(path):
    """(component, year) for a file named like DEMO_I.xpt, or None."""
    m = re.fullmatch(r"([A-Z]+)_([A-Z])\.XPT", Path(path).name.strip().upper())
    if not m:
        return None
    years = [y for y, s in CYCLES.items() if s == m.group(2)]
    return (m.group(1), years[0]) if years and m.group(1) in COMPONENTS else None


def needed(component: str, year: int) -> list:
    v = VARIABLES[component]
    return list(v[year] if isinstance(v, dict) else v)


def read_component(path, component: str, year: int) -> pd.DataFrame:
    """The needed columns of one NHANES file, names upper-cased; a missing variable stops the build."""
    from .brfss_raw import read_xpt, xpt_columns
    cols = xpt_columns(path)
    upper = {c.upper(): c for c in cols}
    want = needed(component, year)
    missing = [v for v in want if v not in upper]
    if missing:
        raise KeyError(f"{Path(path).name}: expected variables {missing} are not in the file")
    df = read_xpt(path, [upper[v] for v in want])
    df.columns = [c.upper() for c in df.columns]
    return df


def merge(frames: dict) -> pd.DataFrame:
    """Every component joined to the demographics file on SEQN (people missing from a component get NaN)."""
    out = frames["DEMO"].copy()
    for comp, df in frames.items():
        if comp == "DEMO":
            continue
        if df.SEQN.duplicated().any():
            raise ValueError(f"{comp}: duplicated SEQN")
        out = out.merge(df, on="SEQN", how="left")
    return out


def check_codes(df: pd.DataFrame, year: int):
    for name, allowed in CODES.items():
        if name not in df:
            continue
        bad = sorted(set(np.unique(df[name].dropna().to_numpy(dtype=float))) - set(map(float, allowed)))
        if bad:
            raise ValueError(f"{year}: {name} has undocumented codes {bad[:10]}")


def _map(s, mapping):
    return pd.Series(s, dtype=float).map({float(k): float(v) for k, v in mapping.items()})


def age_group(age):
    """BRFSS _AGEG5YR groups: 18-24 is 1, then five-year groups, 80 and over is 13."""
    a = np.asarray(age, dtype=float)
    g = np.where(a < 25, 1.0, np.minimum(13.0, np.floor((a - 25) / 5) + 2))
    return np.where(np.isfinite(a) & (a >= 18), g, np.nan)


def drinks_per_week(df: pd.DataFrame, year: int) -> pd.Series:
    if year == 2015:
        q, u = df.ALQ120Q, df.ALQ120U
        none = ((df.ALQ101 == 2) & (df.ALQ110 == 2)) | (q == 0)
        freq = q.where(q <= 366) * _map(u, ALQ120U_PER_WEEK).to_numpy()
    else:
        none = (df.ALQ111 == 2) | (df.ALQ121 == 0)
        freq = _map(df.ALQ121, ALQ121_PER_WEEK).to_numpy()
    out = pd.Series(np.asarray(freq, dtype=float) * df.ALQ130.where(df.ALQ130 <= 15).to_numpy(), index=df.index)
    out[none.to_numpy()] = 0.0
    return out


def diabetes(dq010, dq160):
    d, p = pd.Series(dq010, dtype=float), pd.Series(dq160, dtype=float)
    out = pd.Series(np.nan, index=d.index)
    out[(d == 2) & p.isin([2, 9])] = 0
    out[(d == 2) & (p == 1)] = 1
    out[d == 3] = 1
    out[d == 1] = 2
    return out


def any_activity(*items):
    m = np.column_stack([np.asarray(i, dtype=float) for i in items])
    out = np.full(len(m), np.nan)
    out[(m == 2).all(axis=1)] = 0
    out[(m == 1).any(axis=1)] = 1
    return out


def diff_walk(df: pd.DataFrame) -> pd.Series:
    b = df.PFQ061B
    out = pd.Series(np.nan, index=df.index)
    out[b.isin([1, 2])] = 0
    out[b.isin([3, 4])] = 1
    out[df.PFQ054 == 1] = 1
    routed_past = b.isna() & (df.RIDAGEYR <= 59) & (df.PFQ049 == 2) & (df.PFQ051 == 2) & (df.PFQ054 == 2) \
        & (df.PFQ057 == 2) & (df.PFQ059 == 2)
    out[routed_past] = 0
    return out


def michd(c, e):
    c, e = pd.Series(c, dtype=float), pd.Series(e, dtype=float)
    out = pd.Series(np.nan, index=c.index)
    out[(c == 2) & (e == 2)] = 0
    out[(c == 1) | (e == 1)] = 1
    return out


def measured_high_bp(df: pd.DataFrame):
    sy = df[_SY].mean(axis=1, skipna=True)
    di = df[_DI].where(df[_DI] > 0).mean(axis=1, skipna=True)
    high = (sy >= 140) | (di >= 90)
    return pd.Series(np.where(sy.notna(), high.astype(float), np.nan), index=df.index), sy, di


def harmonise(df: pd.DataFrame, year: int):
    """One merged cycle in the BRFSS coding, plus measured values; returns the table and drop counts."""
    check_codes(df, year)
    YES1 = {1: 1, 2: 0}
    out = pd.DataFrame(index=df.index)
    out["SEQN"] = df.SEQN.astype(int)
    out["cycle"] = year
    out["HighBP"] = _map(df.BPQ020, YES1)
    checked = (df.BPQ080 == 1) | (df.BPQ060 == 1)
    hc = pd.Series(np.nan, index=df.index)
    hc[(df.BPQ080 == 2) & (df.BPQ060 == 1)] = 0
    hc[df.BPQ080 == 1] = 1
    out["HighChol"] = hc
    out["CholCheck"] = _map(df.BPQ070, {1: 1, 2: 1, 3: 1, 4: 0}).where(checked)
    h, w = df.WHD010.where(df.WHD010 < 7777), df.WHD020.where(df.WHD020 < 7777)
    out["BMI_self_exact"] = 703.0696 * w / h ** 2
    out["BMI"] = np.round(out.BMI_self_exact)
    out["Smoker"] = _map(df.SMQ020, YES1)
    out["Stroke"] = _map(df.MCQ160F, YES1)
    out["Diabetes"] = diabetes(df.DIQ010, df.DIQ160)
    out["PhysActivity"] = any_activity(df.PAQ635, df.PAQ650, df.PAQ665)
    sex = _map(df.RIAGENDR, YES1)
    dpw = drinks_per_week(df, year)
    out["drinks_per_week"] = dpw
    out["HvyAlcoholConsump"] = np.where(dpw.notna(), (dpw > np.where(sex == 1, 14, 7)).astype(float), np.nan)
    out["AnyHealthcare"] = _map(df.HIQ011, YES1)
    out["GenHlth"] = _map(df.HUQ010, {i: i for i in range(1, 6)})
    out["DiffWalk"] = diff_walk(df)
    out["Sex"] = sex
    out["Age"] = age_group(df.RIDAGEYR)
    out["Education"] = _map(df.DMDEDUC2, EDUCATION_FROM_NHANES)
    out["Income"] = _map(df.INDHHIN2, INCOME_FROM_NHANES)
    out[TARGET] = michd(df.MCQ160C, df.MCQ160E)
    out["BMI_measured_exact"] = df.BMXBMI
    out["BMI_measured"] = np.round(df.BMXBMI)
    out["HighBP_measured"], out["SBP_mean"], out["DBP_mean"] = measured_high_bp(df)
    out["HighChol_measured"] = np.where(df.LBXTC.notna(), (df.LBXTC >= 240).astype(float), np.nan)
    out["age_years"] = df.RIDAGEYR
    out["WTMEC2YR"] = df.WTMEC2YR

    drops = [("persons in the cycle", len(df))]
    keep = np.array(df.RIDAGEYR >= 20, dtype=bool)               # writable copy
    drops.append(("aged 20 or over", int(keep.sum())))
    keep &= out[TARGET].notna().to_numpy()
    drops.append(("heart disease outcome present (MCQ160C, MCQ160E)", int(keep.sum())))
    for p in N_PREDICTORS:
        keep &= out[p].notna().to_numpy()
        drops.append((f"{p} present", int(keep.sum())))
    keep &= (out.BMI_measured.notna() & out.HighBP_measured.notna()).to_numpy()
    drops.append(("measured BMI and blood pressure present", int(keep.sum())))
    table = out.loc[keep].reset_index(drop=True)
    for c in N_PREDICTORS + [TARGET, "BMI_measured", "HighBP_measured"]:
        table[c] = table[c].astype(float)
    return table, pd.DataFrame(drops, columns=["step", "records"])


def brfss_common(df: pd.DataFrame) -> pd.DataFrame:
    """A BRFSS table (public coding) restricted to the 16 common predictors, education and income coarsened."""
    out = df[N_PREDICTORS].astype(float).copy()
    for col, mapping in (("Education", EDUCATION_FROM_BRFSS), ("Income", INCOME_FROM_BRFSS)):
        bad = sorted(set(np.unique(out[col].to_numpy())) - set(map(float, mapping)))
        if bad:
            raise ValueError(f"BRFSS {col} has unexpected codes {bad}")
        out[col] = _map(out[col], mapping).to_numpy()
    return out
