import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "prepared" / "train.csv"
MODEL_DIR = ROOT / "data" / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"
MODEL_PATH = MODEL_DIR / "model.pkl"
CM_PATH = MODEL_DIR / "confusion_matrix.png"

REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def test_pretrain_required_columns_exist():
    assert TRAIN_CSV.exists(), f"Missing dataset: {TRAIN_CSV}"
    df = pd.read_csv(TRAIN_CSV, nrows=200)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    assert not missing_cols, f"Missing required columns: {missing_cols}"
    assert df["Churn"].notna().all(), "Target column Churn contains nulls"


def test_pretrain_hydra_config_is_valid():
    config_path = ROOT / "config" / "config.yaml"
    assert config_path.exists(), "Missing Hydra base config"
    text = config_path.read_text(encoding="utf-8")
    assert "defaults:" in text
    assert "hydra:" in text
    assert "run:" in text


def test_post_train_artifacts_exist():
    assert MODEL_PATH.exists(), f"Missing model artifact: {MODEL_PATH}"
    assert METRICS_PATH.exists(), f"Missing metrics artifact: {METRICS_PATH}"
    assert CM_PATH.exists(), f"Missing confusion matrix: {CM_PATH}"


def test_post_train_quality_gate():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    for key in ("accuracy", "f1_score", "roc_auc"):
        assert key in metrics, f"Metric {key} is not present in metrics.json"

    assert metrics["f1_score"] >= 0.60, "Quality Gate failed: f1_score < 0.60"
