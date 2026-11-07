
"""Public datasets used by SCAX: sources, downloads, loaders and checks.

Each dataset is its own task with its own recorded outcome; tables are never pooled across outcomes.
Raw files live under DATASETS/raw/<name>/ on Drive, outside the repository.
"""
from __future__ import annotations

import io
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BRFSS_FEATURES = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes", "PhysActivity",
                  "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
                  "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
UCI_HEART_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                     "oldpeak", "slope", "ca", "thal", "num"]
UCI_HEART_SITES = {"cleveland": 303, "hungarian": 294, "switzerland": 123, "va": 200}


@dataclass(frozen=True)
class Spec:
    name: str
    source: str
    url: str
    file: str
    outcome: str
    outcome_definition: str
    rows: int


SPECS = {
    "brfss2015_heart": Spec(
        "brfss2015_heart", "Kaggle alexteboul/heart-disease-health-indicators-dataset (CDC BRFSS 2015, cleaned)",
        "https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset",
        "heart_disease_health_indicators_BRFSS2015.csv", "HeartDiseaseorAttack",
        "Ever told coronary heart disease or myocardial infarction (_MICHD), self-reported", 253680),
    "cdc_diabetes": Spec(
        "cdc_diabetes", "UCI Machine Learning Repository id 891 (CDC BRFSS 2015, cleaned)",
        "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators",
        "cdc_diabetes_health_indicators.csv", "Diabetes_binary",
        "Diabetes_binary as published (self-reported); notebook 01 checks how prediabetes is coded", 253680),
    "uci_heart": Spec(
        "uci_heart", "UCI Machine Learning Repository id 45 (four sites: Cleveland, Hungary, Switzerland, VA Long Beach)",
        "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
        "processed.{site}.data", "num",
        "Angiographic disease: num > 0 (any major vessel with more than 50% narrowing)", 920),
    "cardio70k": Spec(
        "cardio70k", "Kaggle sulianova/cardiovascular-disease-dataset",
        "https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset",
        "cardio_train.csv", "cardio", "Presence of cardiovascular disease as recorded at examination", 70000),
}

KAGGLE_SLUGS = {"brfss2015_heart": "alexteboul/heart-disease-health-indicators-dataset",
                "cardio70k": "sulianova/cardiovascular-disease-dataset"}


def raw_dir(raw_root, name) -> Path:
    d = Path(raw_root) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_kaggle(drive_root) -> None:
    """Copy kaggle.json from the Drive project folder to ~/.kaggle before the kaggle package is imported."""
    src = Path(drive_root) / "kaggle.json"
    if not src.exists():
        raise FileNotFoundError(f"kaggle.json not found at {src} (Kaggle account, Settings, API, Create New Token)")
    dst = Path.home() / ".kaggle" / "kaggle.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    os.chmod(dst, 0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(dst.parent)


def download_kaggle(name, raw_root, drive_root) -> Path:
    d = raw_dir(raw_root, name)
    target = d / SPECS[name].file
    if target.exists():
        return target
    setup_kaggle(drive_root)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_SLUGS[name], path=str(d), unzip=True, quiet=True)
    if not target.exists():
        raise FileNotFoundError(f"{target} missing after download; folder holds {sorted(os.listdir(d))}")
    return target


def download_cdc_diabetes(raw_root) -> Path:
    d = raw_dir(raw_root, "cdc_diabetes")
    target = d / SPECS["cdc_diabetes"].file
    if target.exists():
        return target
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=891)
    df = pd.concat([ds.data.targets.reset_index(drop=True), ds.data.features.reset_index(drop=True)], axis=1)
    df.to_csv(target, index=False)
    return target


def download_uci_heart(raw_root) -> dict:
    d = raw_dir(raw_root, "uci_heart")
    wanted = {site: d / f"processed.{site}.data" for site in UCI_HEART_SITES}
    if all(p.exists() for p in wanted.values()):
        return wanted
    req = urllib.request.Request(SPECS["uci_heart"].url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        blob = r.read()
    (d / "heart+disease.zip").write_bytes(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        members = z.namelist()
        for site, path in wanted.items():
            match = [m for m in members if m.split("/")[-1] == f"processed.{site}.data"]
            if not match:
                raise FileNotFoundError(f"processed.{site}.data not in archive; members: {members}")
            path.write_bytes(z.read(match[0]))
    return wanted


def load_brfss_heart(raw_root):
    df = pd.read_csv(raw_dir(raw_root, "brfss2015_heart") / SPECS["brfss2015_heart"].file)
    X = df[BRFSS_FEATURES].astype(float)
    y = df["HeartDiseaseorAttack"].astype(int).to_numpy()
    return X, y


def load_cdc_diabetes(raw_root):
    df = pd.read_csv(raw_dir(raw_root, "cdc_diabetes") / SPECS["cdc_diabetes"].file)
    y = df["Diabetes_binary"].astype(int).to_numpy()
    X = df.drop(columns=["Diabetes_binary"]).astype(float)
    return X, y


def load_uci_heart(raw_root):
    frames = []
    for site in UCI_HEART_SITES:
        f = raw_dir(raw_root, "uci_heart") / f"processed.{site}.data"
        part = pd.read_csv(f, header=None, names=UCI_HEART_COLUMNS, na_values="?")
        part.insert(0, "site", site)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    y = (df["num"] > 0).astype(int).to_numpy()
    X = df.drop(columns=["num"])
    return X, y


def load_cardio(raw_root):
    df = pd.read_csv(raw_dir(raw_root, "cardio70k") / SPECS["cardio70k"].file, sep=";")
    y = df["cardio"].astype(int).to_numpy()
    X = df.drop(columns=["cardio", "id"]).astype(float)
    return X, y


LOADERS = {"brfss2015_heart": load_brfss_heart, "cdc_diabetes": load_cdc_diabetes,
           "uci_heart": load_uci_heart, "cardio70k": load_cardio}


def card(name, raw_root) -> dict:
    X, y = LOADERS[name](raw_root)
    spec = SPECS[name]
    missing = int(pd.DataFrame(X).isna().sum().sum())
    return {"dataset": name, "source": spec.source, "url": spec.url, "rows": int(len(y)),
            "expected_rows": spec.rows, "predictors": int(sum(c != "site" for c in X.columns)), "outcome": spec.outcome,
            "outcome_definition": spec.outcome_definition, "positives": int(y.sum()),
            "prevalence": float(np.mean(y)), "missing_values": missing}
