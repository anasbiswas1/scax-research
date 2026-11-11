"""NHANES harmonisation rules, BRFSS common coding and UCI site preparation."""
import numpy as np
import pandas as pd
import pytest

from scax.data import nhanes as nh
from scax.data import uci
from scax.data.outcomes import OUTCOME_MAP

NAN = np.nan


def _person(year, **kw):
    """One adult with every item answered 'no' or a neutral value; keyword arguments override."""
    base = {"SEQN": 1, "RIAGENDR": 1, "RIDAGEYR": 45, "DMDEDUC2": 3, "INDHHIN2": 6, "WTMEC2YR": 1000.0,
            "BPQ020": 2, "BPQ060": 1, "BPQ070": 1, "BPQ080": 2,
            "BPXSY1": 120, "BPXSY2": 122, "BPXSY3": 118, "BPXSY4": NAN, "BPXDI1": 70, "BPXDI2": 72, "BPXDI3": 0, "BPXDI4": NAN,
            "BMXBMI": 27.4, "LBXTC": 190.0, "WHD010": 70, "WHD020": 180, "SMQ020": 2,
            "MCQ160C": 2, "MCQ160E": 2, "MCQ160F": 2, "DIQ010": 2, "DIQ160": 2,
            "PAQ635": 2, "PAQ650": 2, "PAQ665": 1, "HIQ011": 1, "HUQ010": 3,
            "PFQ049": 2, "PFQ051": 2, "PFQ054": 2, "PFQ057": 2, "PFQ059": 2, "PFQ061B": NAN}
    if year == 2015:
        base.update({"ALQ101": 1, "ALQ110": NAN, "ALQ120Q": 2, "ALQ120U": 1, "ALQ130": 2})
    else:
        base.update({"ALQ111": 1, "ALQ121": 4, "ALQ130": 2})
    base.update(kw)
    return base


def _frame(year, people):
    df = pd.DataFrame([_person(year, SEQN=i + 1, **p) for i, p in enumerate(people)])
    return df.astype(float)


def test_urls_and_file_names():
    assert nh.url("DEMO", 2015) == "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.xpt"
    assert nh.url("ALQ", 2017).endswith("/2017/DataFiles/ALQ_J.xpt")
    assert nh.stem_of("datasets/raw/nhanes/demo_i.XPT") == ("DEMO", 2015)
    assert nh.stem_of("P_DEMO.xpt") is None and nh.stem_of("DEMO_H.xpt") is None
    assert len(nh.N_PREDICTORS) == 16 and not {"MentHlth", "PhysHlth", "NoDocbcCost", "Fruits", "Veggies"} & set(nh.N_PREDICTORS)


def test_age_groups_follow_brfss():
    g = nh.age_group([18, 20, 24, 25, 29, 30, 74, 75, 79, 80, 10])
    assert list(g[:10]) == [1, 1, 1, 2, 2, 3, 11, 12, 12, 13] and np.isnan(g[10])


def test_each_rule_on_one_cycle():
    people = [
        {},                                                      # 0 neutral adult
        {"BPQ080": 1, "BPQ060": NAN, "BPQ070": 4},               # 1 told high cholesterol; checked 5+ years ago
        {"BPQ080": 2, "BPQ060": 2, "BPQ070": NAN},               # 2 never checked: HighChol missing (as in BRFSS)
        {"DIQ010": 3, "DIQ160": NAN},                            # 3 borderline diabetes
        {"DIQ010": 2, "DIQ160": 1},                              # 4 told prediabetes
        {"DIQ010": 1, "DIQ160": NAN, "MCQ160E": 1},              # 5 diabetes; heart attack
        {"PFQ054": 1, "PFQ061B": NAN, "RIDAGEYR": 70},           # 6 needs equipment to walk
        {"PFQ061B": 3, "RIDAGEYR": 65},                          # 7 much difficulty
        {"PFQ061B": 2, "RIDAGEYR": 65},                          # 8 some difficulty
        {"PFQ061B": NAN, "RIDAGEYR": 65},                        # 9 over 59 without an answer: missing
        {"INDHHIN2": 13},                                        # 10 'under $20k': income missing
        {"BPXSY1": 150, "BPXSY2": 146, "BPXSY3": 140},           # 11 measured high blood pressure
        {"PAQ635": 2, "PAQ650": 2, "PAQ665": 2},                 # 12 no activity
        {"MCQ160C": 9, "MCQ160E": 2},                            # 13 outcome unknown
        {"RIDAGEYR": 19},                                        # 14 under 20
        {"ALQ120Q": 7, "ALQ120U": 1, "ALQ130": 3},               # 15 21 drinks a week, man: heavy
        {"ALQ101": 2, "ALQ110": 2, "ALQ120Q": NAN, "ALQ120U": NAN, "ALQ130": NAN},  # 16 lifetime abstainer
        {"RIAGENDR": 2, "ALQ120Q": 4, "ALQ120U": 1, "ALQ130": 2},  # 17 woman, 8 drinks a week: heavy
        {"WHD010": 9999},                                        # 18 height unknown: BMI missing
        {"DMDEDUC2": 5, "INDHHIN2": 15, "LBXTC": 250.0},         # 19 college, $100k+, measured high cholesterol
    ]
    df = _frame(2015, people)
    t, drops = nh.harmonise(df, 2015)
    kept = set(t.SEQN.astype(int) - 1)
    assert kept == set(range(20)) - {2, 9, 10, 13, 14, 18}
    r = t.set_index(t.SEQN.astype(int) - 1)
    assert r.loc[0, "HighChol"] == 0 and r.loc[0, "CholCheck"] == 1 and r.loc[0, "PhysActivity"] == 1
    assert r.loc[1, "HighChol"] == 1 and r.loc[1, "CholCheck"] == 0
    assert r.loc[3, "Diabetes"] == 1 and r.loc[4, "Diabetes"] == 1 and r.loc[5, "Diabetes"] == 2
    assert r.loc[5, nh.TARGET] == 1 and r.loc[0, nh.TARGET] == 0
    assert r.loc[0, "DiffWalk"] == 0 and r.loc[6, "DiffWalk"] == 1 and r.loc[7, "DiffWalk"] == 1 and r.loc[8, "DiffWalk"] == 0
    assert r.loc[11, "HighBP_measured"] == 1 and r.loc[0, "HighBP_measured"] == 0 and r.loc[0, "HighBP"] == 0
    assert r.loc[12, "PhysActivity"] == 0
    assert r.loc[15, "HvyAlcoholConsump"] == 1 and r.loc[0, "HvyAlcoholConsump"] == 0
    assert r.loc[16, "HvyAlcoholConsump"] == 0 and r.loc[17, "HvyAlcoholConsump"] == 1
    assert r.loc[0, "BMI"] == np.round(703.0696 * 180 / 70 ** 2) and r.loc[0, "BMI_measured"] == 27
    assert r.loc[0, "Age"] == 6 and r.loc[0, "Sex"] == 1 and r.loc[17, "Sex"] == 0
    assert r.loc[0, "Education"] == 3 and r.loc[0, "Income"] == 5 and r.loc[19, "Education"] == 5 and r.loc[19, "Income"] == 7
    assert r.loc[19, "HighChol_measured"] == 1 and r.loc[0, "HighChol_measured"] == 0
    assert drops.records.is_monotonic_decreasing and drops.records.iloc[-1] == len(t)


def test_2017_alcohol_and_undocumented_codes():
    people = [{"ALQ121": 1, "ALQ130": 3}, {"ALQ121": 5, "ALQ130": 3}, {"ALQ111": 2, "ALQ121": NAN, "ALQ130": NAN},
              {"ALQ121": 0, "ALQ130": NAN}, {"ALQ121": 99, "ALQ130": 2}]
    t, _ = nh.harmonise(_frame(2017, people), 2017)
    r = t.set_index(t.SEQN.astype(int) - 1)
    assert r.loc[0, "HvyAlcoholConsump"] == 1 and r.loc[1, "HvyAlcoholConsump"] == 0
    assert r.loc[2, "HvyAlcoholConsump"] == 0 and r.loc[3, "HvyAlcoholConsump"] == 0 and 4 not in r.index
    with pytest.raises(ValueError):
        nh.harmonise(_frame(2017, [{"HUQ010": 6}]), 2017)


def test_brfss_common_coding():
    df = pd.DataFrame({c: [1.0] * 8 for c in nh.N_PREDICTORS})
    df["Education"] = [1, 2, 3, 4, 5, 6, 6, 6]
    df["Income"] = [1, 2, 3, 4, 5, 6, 7, 8]
    out = nh.brfss_common(df)
    assert list(out.Education) == [1, 1, 2, 3, 4, 5, 5, 5] and list(out.Income) == [1, 2, 3, 4, 5, 6, 6, 7]
    assert list(out.columns) == nh.N_PREDICTORS
    df.loc[0, "Income"] = 9
    with pytest.raises(ValueError):
        nh.brfss_common(df)


def test_read_component_checks_variables(tmp_path):
    pyreadstat = pytest.importorskip("pyreadstat")
    f = tmp_path / "SMQ_I.xpt"
    pyreadstat.write_xport(pd.DataFrame({"SEQN": [1.0, 2.0], "SMQ020": [1.0, 2.0]}), str(f))
    assert list(nh.read_component(f, "SMQ", 2015).columns) == ["SEQN", "SMQ020"]
    g = tmp_path / "MCQ_I.xpt"
    pyreadstat.write_xport(pd.DataFrame({"SEQN": [1.0], "MCQ160C": [1.0]}), str(g))
    with pytest.raises(KeyError):
        nh.read_component(g, "MCQ", 2015)


def test_uci_preparation():
    X = pd.DataFrame({"site": ["cleveland"] * 3 + ["hungarian"] * 2 + ["switzerland"] * 2 + ["va"] * 2,
                      **{c: [1.0] * 9 for c in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                                                 "exang", "oldpeak", "slope", "ca", "thal"]}})
    X.loc[7, "trestbps"] = 0.0          # zero blood pressure is missing
    X.loc[3, "thalach"] = np.nan
    X["chol"] = [200.0, 210, 220, 230, np.nan, 0, 0, 240, 250]
    y = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
    t, counts = uci.prepare(X, y)
    assert len(t) == 7 and list(t.columns) == ["site"] + uci.PREDICTORS + [uci.TARGET]
    assert counts.set_index("site").complete_cases.to_dict() == {"cleveland": 3, "hungarian": 1, "switzerland": 2, "va": 1}
    miss = uci.missingness(X)
    assert miss.loc["switzerland", "chol"] == 1.0 and miss.loc["va", "trestbps"] == 0.5


def test_outcome_map_complete():
    assert len(OUTCOME_MAP) == 5 and all(set(r) >= {"dataset", "outcome", "positive", "negative"} for r in OUTCOME_MAP)
