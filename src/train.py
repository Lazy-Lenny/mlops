import argparse
import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sampling import random_oversample_binary

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train churn model from prepared data")

    parser.add_argument(
        "--train_path",
        type=str,
        default="data/prepared/train.csv",
        help="Path to prepared train CSV",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/prepared/test.csv",
        help="Path to prepared test CSV",
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
        default="random_forest_refactored",
        help="MLflow run name",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="Churn",
        help="Target column name",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument("--n_estimators", type=int, default=257)
    parser.add_argument("--max_depth", type=int, default=9)
    parser.add_argument("--min_samples_split", type=int, default=8)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", type=str, default="lbfgs")
    parser.add_argument("--penalty", type=str, default="l2")
    parser.add_argument("--max_iter", type=int, default=2000)

    parser.add_argument("--author", type=str, default="student")
    parser.add_argument("--dataset_version", type=str, default="v1")
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
    )

    parser.add_argument(
        "--model_output_dir",
        type=str,
        default="data/models",
        help="Directory to save trained model locally",
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        default="model.pkl",
        help="Filename for saved model",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts",
        help="Directory to save plots and reports locally",
    )
    parser.add_argument(
        "--metrics_filename",
        type=str,
        default="metrics.json",
        help="Filename for metrics JSON",
    )
    parser.add_argument(
        "--confusion_matrix_filename",
        type=str,
        default="confusion_matrix.png",
        help="Filename for confusion matrix image",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Optional row limit for CI speedup (0 means all rows)",
    )
    parser.add_argument(
        "--ci_mode",
        action="store_true",
        help="Enable CI-friendly fast training settings",
    )
    parser.add_argument(
        "--no_oversample",
        action="store_true",
        help="Disable random oversampling of the minority class on the training set",
    )

    return parser.parse_args()


def split_xy(df: pd.DataFrame, target_col: str):
    df = df.copy()
    y = df[target_col].map({"No": 0, "Yes": 1})
    X = df.drop(columns=[target_col])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def main():
    args = parse_args()
    ci_mode = args.ci_mode or os.getenv("CI", "").lower() == "true"

    os.makedirs(args.model_output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    if args.max_rows and args.max_rows > 0:
        train_df = train_df.head(args.max_rows).copy()
        test_df = test_df.head(max(1, args.max_rows // 4)).copy()

    X_train, y_train = split_xy(train_df, args.target_col)
    X_test, y_test = split_xy(test_df, args.target_col)

    use_oversample = not args.no_oversample
    if use_oversample:
        X_train, y_train = random_oversample_binary(
            X_train, y_train, random_state=args.random_state
        )

    preprocessor = build_preprocessor(X_train)
    class_weight = None if use_oversample else "balanced"

    if args.model_type == "random_forest":
        n_estimators = min(args.n_estimators, 80) if ci_mode else args.n_estimators
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
    else:
        model = LogisticRegression(
            C=args.C,
            solver=args.solver,
            penalty=args.penalty,
            max_iter=args.max_iter,
            random_state=args.random_state,
            class_weight=class_weight,
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("train_path", args.train_path)
        mlflow.log_param("test_path", args.test_path)
        mlflow.log_param("target_col", args.target_col)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("C", args.C)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("penalty", args.penalty)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("model_output_dir", args.model_output_dir)
        mlflow.log_param("model_filename", args.model_filename)
        mlflow.log_param("artifacts_dir", args.artifacts_dir)
        mlflow.log_param("metrics_filename", args.metrics_filename)
        mlflow.log_param(
            "confusion_matrix_filename", args.confusion_matrix_filename
        )
        mlflow.log_param("max_rows", args.max_rows)
        mlflow.log_param("ci_mode", ci_mode)
        mlflow.log_param("oversample", use_oversample)

        mlflow.set_tag("author", args.author)
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("project", "telco-churn-prediction")
        mlflow.set_tag("stage", "train")

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

        # MLflow model logging
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Local model saving
        model_path = os.path.join(args.model_output_dir, args.model_filename)
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)

        metrics_path = os.path.join(args.model_output_dir, args.metrics_filename)
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)
        mlflow.log_artifact(metrics_path)

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        cm_path = os.path.join(args.model_output_dir, args.confusion_matrix_filename)
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path)

        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        model_step = pipeline.named_steps["model"]

        if hasattr(model_step, "feature_importances_"):
            scores = model_step.feature_importances_
            score_col = "importance"
            artifact_prefix = "feature_importance"
            title = "Top 20 Feature Importances"
        else:
            scores = abs(model_step.coef_[0])
            score_col = "weight_abs"
            artifact_prefix = "feature_weights"
            title = "Top 20 Logistic Regression Weights (abs)"

        fi_df = pd.DataFrame({"feature": feature_names, score_col: scores}).sort_values(
            score_col, ascending=False
        )

        fi_csv_path = os.path.join(args.artifacts_dir, f"{artifact_prefix}.csv")
        fi_df.to_csv(fi_csv_path, index=False)
        mlflow.log_artifact(fi_csv_path)

        top20_fi = fi_df.head(20)

        fig, ax = plt.subplots(figsize=(10, 6))
        top20_fi.sort_values(score_col).plot(
            x="feature",
            y=score_col,
            kind="barh",
            ax=ax,
            legend=False,
        )
        ax.set_title(title)
        ax.set_xlabel(score_col)
        ax.set_ylabel("Feature")
        fig.tight_layout()

        fi_plot_path = os.path.join(args.artifacts_dir, f"{artifact_prefix}_top20.png")
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

        print(f"\nLocal model saved to: {model_path}")
        print("Training completed successfully.")


if __name__ == "__main__":
    main()
