"""Load local model.pkl, log to MLflow, register in Model Registry as Staging (for Airflow / CI)."""

from __future__ import annotations

import argparse
import json
import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register sklearn pipeline from disk in MLflow Registry")
    p.add_argument(
        "--model_path",
        default="data/models/model.pkl",
        help="Path to joblib model (Pipeline)",
    )
    p.add_argument(
        "--metrics_path",
        default="data/models/metrics.json",
        help="Optional metrics JSON to log with the registration run",
    )
    p.add_argument(
        "--registry_name",
        default="TelcoChurnModel",
        help="Registered model name",
    )
    p.add_argument(
        "--experiment",
        default="telco-churn-registry",
        help="MLflow experiment for the packaging run",
    )
    p.add_argument(
        "--run_name",
        default="airflow_register",
        help="Run name for logged model",
    )
    p.add_argument(
        "--tracking_uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    model = joblib.load(args.model_path)
    metrics = {}
    if os.path.isfile(args.metrics_path):
        with open(args.metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = mlflow.active_run().info.run_id

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=args.registry_name)

    client = MlflowClient(tracking_uri=args.tracking_uri)
    last_ver = int(mv.version)
    try:
        client.transition_model_version_stage(
            name=args.registry_name,
            version=last_ver,
            stage="Staging",
            archive_existing_versions=False,
        )
    except Exception:
        # Some deployments use aliases instead of stages; registration still succeeded.
        pass

    print(f"Registered {args.registry_name} version {last_ver} (run {run_id})")


if __name__ == "__main__":
    main()
