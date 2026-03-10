from __future__ import annotations

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor, load_data, split_features_target

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train churn model with MLflow")

    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="telco-churn-experiment",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="random_forest_baseline",
        help="MLflow run name",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test size for train_test_split",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )

    # hyperparameters
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
        help="Number of trees in RandomForest",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Maximum tree depth",
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=5,
        help="Minimum samples required to split",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=2,
        help="Minimum samples required in leaf",
    )

    # tags
    parser.add_argument(
        "--author",
        type=str,
        default="student",
        help="Run author for MLflow tag",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="v1",
        help="Dataset version tag",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="RandomForest",
        help="Model type tag",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    df = load_data(args.data_path)
    X, y = split_features_target(df, target_col="Churn")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("data_path", args.data_path)

        mlflow.set_tag("author", args.author)
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("project", "telco-churn-prediction")
        mlflow.set_tag("stage", "baseline")

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path)

        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_

        fi_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
        )

        fi_csv_path = "artifacts/feature_importance.csv"
        fi_df.to_csv(fi_csv_path, index=False)
        mlflow.log_artifact(fi_csv_path)

        top20_fi = fi_df.head(20)

        fig, ax = plt.subplots(figsize=(10, 6))
        top20_fi.sort_values("importance").plot(
            x="feature",
            y="importance",
            kind="barh",
            ax=ax,
            legend=False,
        )
        ax.set_title("Top 20 Feature Importances")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        fig.tight_layout()

        fi_plot_path = "artifacts/feature_importance_top20.png"
        plt.savefig(fi_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(fi_plot_path)

        if not fi_df.empty:
            mlflow.set_tag("top_feature", fi_df.iloc[0]["feature"])

        print("=== Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        print("\n=== Top 10 Important Features ===")
        print(fi_df.head(10).to_string(index=False))

        print("\nRun completed successfully.")


if __name__ == "__main__":
    main()
