import json
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "prepared" / "train.csv"
TRAIN_CSV_FIXTURE = ROOT / "tests" / "fixtures" / "prepared_train_sample.csv"
CONFIG_DIR = ROOT / "config"
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
    train_path = TRAIN_CSV if TRAIN_CSV.exists() else TRAIN_CSV_FIXTURE
    assert train_path.exists(), (
        f"Missing dataset: neither {TRAIN_CSV} nor fixture {TRAIN_CSV_FIXTURE}"
    )
    df = pd.read_csv(train_path, nrows=200)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    assert not missing_cols, f"Missing required columns: {missing_cols}"
    assert df["Churn"].notna().all(), "Target column Churn contains nulls"


def test_pretrain_hydra_config_is_valid():
    assert CONFIG_DIR.is_dir(), f"Missing Hydra config dir: {CONFIG_DIR}"
    base_config = CONFIG_DIR / "config.yaml"
    assert base_config.exists(), "Missing Hydra base config.yaml"
    # Hydra does not expose the `hydra:` overrides block on the composed cfg (struct has no `hydra` key).
    text = base_config.read_text(encoding="utf-8")
    assert "defaults:" in text
    assert "hydra:" in text
    assert "run:" in text
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config")
    assert cfg.data.train_path and cfg.data.test_path
    assert hasattr(cfg.hpo, "oversample"), "hpo.oversample missing (check config/hpo/*.yaml)"


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
