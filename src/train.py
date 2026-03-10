import argparse
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

    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=2)

    parser.add_argument("--author", type=str, default="student")
    parser.add_argument("--dataset_version", type=str, default="v1")
    parser.add_argument("--model_type", type=str, default="RandomForest")

    parser.add_argument(
        "--model_output_dir",
        type=str,
        default="models",
        help="Directory to save trained model locally",
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        default="model.joblib",
        help="Filename for saved model",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts",
        help="Directory to save plots and reports locally",
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

    os.makedirs(args.model_output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    X_train, y_train = split_xy(train_df, args.target_col)
    X_test, y_test = split_xy(test_df, args.target_col)

    preprocessor = build_preprocessor(X_train)

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
        mlflow.log_param("train_path", args.train_path)
        mlflow.log_param("test_path", args.test_path)
        mlflow.log_param("target_col", args.target_col)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("model_output_dir", args.model_output_dir)
        mlflow.log_param("model_filename", args.model_filename)
        mlflow.log_param("artifacts_dir", args.artifacts_dir)

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

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        cm_path = os.path.join(args.artifacts_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path)

        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_

        fi_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        fi_csv_path = os.path.join(args.artifacts_dir, "feature_importance.csv")
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

        fi_plot_path = os.path.join(args.artifacts_dir, "feature_importance_top20.png")
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
