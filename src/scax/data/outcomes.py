"""Outcome definitions, frozen in notebook 03 (plan Section 6.5, outcome policy).

Each dataset is its own task with its own outcome; data are never pooled across different outcomes, and
shift experiments compare only domains with the same outcome variable.
"""
OUTCOME_MAP = [
    {"dataset": "BRFSS 2015 (public heart table) and BRFSS 2015-2023 rebuilt from the raw files",
     "outcome": "HeartDiseaseorAttack", "definition": "ever told coronary heart disease or myocardial infarction",
     "source_items": "_MICHD (from CVDCRHD4 and CVDINFR4)", "positive": "_MICHD = 1", "negative": "_MICHD = 2",
     "missing": "dropped", "kind": "self-reported diagnosis", "used_for": "task A; temporal and geographic shift"},
    {"dataset": "CDC Diabetes Health Indicators (BRFSS 2015)", "outcome": "Diabetes_binary",
     "definition": "ever told diabetes (prediabetes and diabetes only during pregnancy coded as no)",
     "source_items": "DIABETE3", "positive": "DIABETE3 = 1", "negative": "DIABETE3 = 2, 3 or 4",
     "missing": "dropped", "kind": "self-reported diagnosis", "used_for": "task B"},
    {"dataset": "NHANES 2015-2016 and 2017-2018, adults 20 and over", "outcome": "HeartDiseaseorAttack",
     "definition": "ever told coronary heart disease or heart attack, built like _MICHD",
     "source_items": "MCQ160C, MCQ160E", "positive": "MCQ160C = 1 or MCQ160E = 1", "negative": "both = 2",
     "missing": "dropped (refused, don't know or not asked, with no yes)", "kind": "self-reported diagnosis",
     "used_for": "measurement shift (same outcome as BRFSS)"},
    {"dataset": "UCI Heart Disease (Cleveland, Hungary, Switzerland, VA Long Beach)", "outcome": "angiographic_disease",
     "definition": "more than 50% narrowing in any major vessel at angiography", "source_items": "num",
     "positive": "num > 0", "negative": "num = 0", "missing": "none", "kind": "clinical test",
     "used_for": "site shift (Cleveland to the other three sites)"},
    {"dataset": "Cardiovascular Disease dataset (Kaggle, 70,000 examinations)", "outcome": "cardio",
     "definition": "cardiovascular disease as recorded at examination", "source_items": "cardio", "positive": "1",
     "negative": "0", "missing": "none", "kind": "recorded at examination", "used_for": "secondary task"},
]
