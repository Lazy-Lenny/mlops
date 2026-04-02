"""
Telco churn ML training DAG: data check → prepare → train → quality branch → MLflow Staging.
Project root is mounted at MLOPS_PROJECT_ROOT (default /opt/mlops in Docker Compose).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

RAW_REL = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def _project_root() -> Path:
    return Path(os.environ.get("MLOPS_PROJECT_ROOT", "/opt/mlops")).resolve()


def check_raw_and_dvc() -> None:
    """Fail fast if raw CSV missing; optionally surface DVC drift (non-fatal)."""
    root = _project_root()
    raw = root / RAW_REL
    if not raw.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw}. Place the file or run dvc pull."
        )
    subprocess.run(
        ["dvc", "status", "-q"],
        cwd=str(root),
        check=False,
    )


def choose_after_training(**_context) -> str:
    metrics_path = _project_root() / "data" / "models" / "metrics.json"
    threshold = float(os.environ.get("F1_THRESHOLD", "0.55"))
    if not metrics_path.is_file():
        return "notify_quality_fail"
    with open(metrics_path, encoding="utf-8") as fh:
        metrics = json.load(fh)
    f1 = float(metrics.get("f1_score", 0.0))
    return "register_model" if f1 >= threshold else "notify_quality_fail"


with DAG(
    dag_id="ml_training_pipeline",
    description="Prepare data, train churn model, register if F1 passes threshold",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "telco", "churn"],
    default_args={"retries": 0},
) as dag:
    # Paths resolved at DAG-parse time (scheduler inherits MLOPS_PROJECT_ROOT from Docker/env).
    env_root = str(_project_root())

    check_data = PythonOperator(
        task_id="check_data_and_dvc",
        python_callable=check_raw_and_dvc,
    )

    dvc_prepare = BashOperator(
        task_id="dvc_repro_prepare",
        bash_command=(
            f'cd "{env_root}" && '
            "(dvc repro prepare || python src/prepare.py "
            "--input_file=data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv "
            "--output_dir=data/prepared)"
        ),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f'cd "{env_root}" && python src/train.py '
            "--model_output_dir=data/models "
            "--model_type=random_forest "
            "--max_depth=20"
        ),
    )

    evaluate_branch = BranchPythonOperator(
        task_id="evaluate_quality_branch",
        python_callable=choose_after_training,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=(
            f'cd "{env_root}" && python src/register_model_mlflow.py '
            "--model_path=data/models/model.pkl "
            "--metrics_path=data/models/metrics.json "
            "--registry_name=TelcoChurnModel"
        ),
    )

    notify_quality_fail = BashOperator(
        task_id="notify_quality_fail",
        bash_command=(
            'echo "Quality gate failed: f1_score below F1_THRESHOLD; '
            'see data/models/metrics.json" && exit 0'
        ),
    )

    pipeline_done = EmptyOperator(
        task_id="pipeline_done",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    check_data >> dvc_prepare >> train_model >> evaluate_branch
    evaluate_branch >> register_model >> pipeline_done
    evaluate_branch >> notify_quality_fail >> pipeline_done
