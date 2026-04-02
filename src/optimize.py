from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any

import hydra
import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor, split_features_target
from sampling import random_oversample_binary


def _normalize_model_name(raw_value: str) -> str:
    value = raw_value.strip().lower().replace("-", "_")
    aliases = {
        "randomforest": "random_forest",
        "random_forest": "random_forest",
        "rf": "random_forest",
        "logisticregression": "logistic_regression",
        "logistic_regression": "logistic_regression",
        "lr": "logistic_regression",
    }
    if value not in aliases:
        raise ValueError(
            "Unsupported model value. Use RandomForest or LogisticRegression."
        )
    return aliases[value]


def preprocess_cli_args() -> None:
    # Support user-friendly flags in addition to Hydra overrides.
    # Example: python src/optimize.py --model_type RandomForest
    model_value = None
    filtered_args = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ("--model_type", "--model"):
            if i + 1 >= len(sys.argv):
                raise ValueError(f"Missing value after {arg}")
            model_value = sys.argv[i + 1]
            i += 2
            continue
        filtered_args.append(arg)
        i += 1

    if model_value is not None:
        normalized = _normalize_model_name(model_value)
        filtered_args.append(f"model={normalized}")

    sys.argv = filtered_args


def build_sampler(cfg: DictConfig) -> optuna.samplers.BaseSampler:
    sampler_name = cfg.hpo.sampler.lower()

    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=cfg.seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=cfg.seed)
    if sampler_name == "grid":
        grid_space = OmegaConf.to_container(
            cfg.hpo.grid_search_space[cfg.model.type], resolve=True
        )
        return optuna.samplers.GridSampler(search_space=grid_space)

    raise ValueError(f"Unsupported sampler: {cfg.hpo.sampler}")


def suggest_model_params(trial: optuna.trial.Trial, cfg: DictConfig) -> dict[str, Any]:
    model_type = cfg.model.type
    search_space = cfg.model.search_space
    params: dict[str, Any] = {}

    if model_type == "random_forest":
        params["n_estimators"] = trial.suggest_int(
            "n_estimators",
            search_space.n_estimators.low,
            search_space.n_estimators.high,
        )
        params["max_depth"] = trial.suggest_int(
            "max_depth",
            search_space.max_depth.low,
            search_space.max_depth.high,
        )
        params["min_samples_split"] = trial.suggest_int(
            "min_samples_split",
            search_space.min_samples_split.low,
            search_space.min_samples_split.high,
        )
        params["min_samples_leaf"] = trial.suggest_int(
            "min_samples_leaf",
            search_space.min_samples_leaf.low,
            search_space.min_samples_leaf.high,
        )
        params["random_state"] = cfg.seed
        params.update(OmegaConf.to_container(cfg.model.fixed_params, resolve=True))
        return params

    if model_type == "logistic_regression":
        params["C"] = trial.suggest_float(
            "C",
            search_space.C.low,
            search_space.C.high,
            log=True,
        )
        params["solver"] = trial.suggest_categorical("solver", search_space.solver.choices)
        params["penalty"] = trial.suggest_categorical(
            "penalty", search_space.penalty.choices
        )
        params["random_state"] = cfg.seed
        params.update(OmegaConf.to_container(cfg.model.fixed_params, resolve=True))
        return params

    raise ValueError(f"Unsupported model type: {model_type}")


def build_estimator(
    model_type: str, params: dict[str, Any], *, class_weight: str | None
):
    merged = {**params, "class_weight": class_weight}
    if model_type == "random_forest":
        return RandomForestClassifier(**merged)
    if model_type == "logistic_regression":
        return LogisticRegression(**merged)
    raise ValueError(f"Unsupported model type: {model_type}")


def score_holdout(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    metric_name: str,
) -> float:
    pipeline.fit(x_train, y_train)

    if metric_name == "f1":
        y_pred = pipeline.predict(x_test)
        return float(f1_score(y_test, y_pred))

    if metric_name == "roc_auc":
        y_proba = pipeline.predict_proba(x_test)[:, 1]
        return float(roc_auc_score(y_test, y_proba))

    raise ValueError(f"Unsupported metric: {metric_name}")


def score_cv_stratified(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    metric_name: str,
    cv_folds: int,
    random_state: int,
    *,
    oversample_train_fold: bool,
) -> float:
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    scores: list[float] = []
    for train_idx, val_idx in skf.split(x_train, y_train):
        X_tr = x_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = x_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]
        if oversample_train_fold:
            X_tr, y_tr = random_oversample_binary(X_tr, y_tr, random_state=random_state)
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_tr, y_tr)
        if metric_name == "f1":
            scores.append(float(f1_score(y_val, fold_pipeline.predict(X_val))))
        elif metric_name == "roc_auc":
            y_proba = fold_pipeline.predict_proba(X_val)[:, 1]
            scores.append(float(roc_auc_score(y_val, y_proba)))
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
    return float(sum(scores) / len(scores))


def create_objective(
    cfg: DictConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
):
    class_weight = None if cfg.hpo.oversample else "balanced"

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_model_params(trial, cfg)
        estimator = build_estimator(
            cfg.model.type, params, class_weight=class_weight
        )
        preprocessor = build_preprocessor(x_train)
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.set_tags(
                {
                    "trial_number": trial.number,
                    "sampler": cfg.hpo.sampler,
                    "model_type": cfg.model.type,
                    "seed": cfg.seed,
                }
            )

            if cfg.hpo.use_cv:
                value = score_cv_stratified(
                    pipeline,
                    x_train,
                    y_train,
                    cfg.hpo.metric,
                    cfg.hpo.cv_folds,
                    cfg.seed,
                    oversample_train_fold=cfg.hpo.oversample,
                )
            else:
                value = score_holdout(
                    pipeline,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    cfg.hpo.metric,
                )

            mlflow.log_metric(cfg.hpo.metric, value)
            return value

    return objective


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    train_df = pd.read_csv(cfg.data.train_path)
    test_df = pd.read_csv(cfg.data.test_path)

    x_train, y_train = split_features_target(train_df, target_col=cfg.target_col)
    x_test, y_test = split_features_target(test_df, target_col=cfg.target_col)

    if cfg.hpo.oversample and not cfg.hpo.use_cv:
        x_train, y_train = random_oversample_binary(x_train, y_train, cfg.seed)

    sampler = build_sampler(cfg)
    study = optuna.create_study(
        direction=cfg.hpo.direction,
        sampler=sampler,
    )
    objective = create_objective(cfg, x_train, y_train, x_test, y_test)

    with mlflow.start_run(run_name=cfg.mlflow.parent_run_name):
        mlflow.log_params(
            {
                "seed": cfg.seed,
                "sampler": cfg.hpo.sampler,
                "n_trials": cfg.hpo.n_trials,
                "metric": cfg.hpo.metric,
                "direction": cfg.hpo.direction,
                "model_type": cfg.model.type,
                "use_cv": cfg.hpo.use_cv,
                "cv_folds": cfg.hpo.cv_folds,
                "oversample": cfg.hpo.oversample,
            }
        )

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp_cfg:
            tmp_cfg.write(OmegaConf.to_yaml(cfg))
            cfg_path = tmp_cfg.name
        mlflow.log_artifact(cfg_path, artifact_path="config")
        os.remove(cfg_path)

        study.optimize(objective, n_trials=cfg.hpo.n_trials)

        best_params = study.best_params
        best_value = float(study.best_value)
        mlflow.log_metric(f"best_{cfg.hpo.metric}", best_value)
        mlflow.log_text(json.dumps(best_params, indent=2), "best_params.json")

        best_estimator_params = dict(best_params)
        best_estimator_params["random_state"] = cfg.seed
        best_estimator_params.update(
            OmegaConf.to_container(cfg.model.fixed_params, resolve=True)
        )
        fit_class_weight = None if cfg.hpo.oversample else "balanced"
        best_estimator = build_estimator(
            cfg.model.type, best_estimator_params, class_weight=fit_class_weight
        )
        best_pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(x_train)),
                ("model", best_estimator),
            ]
        )
        x_fit, y_fit = x_train, y_train
        if cfg.hpo.oversample and cfg.hpo.use_cv:
            x_fit, y_fit = random_oversample_binary(x_train, y_train, cfg.seed)
        best_pipeline.fit(x_fit, y_fit)

        os.makedirs("artifacts", exist_ok=True)
        best_model_path = os.path.join("artifacts", "best_model.pkl")
        joblib.dump(best_pipeline, best_model_path)
        mlflow.log_artifact(best_model_path, artifact_path="model")

        trials_df = study.trials_dataframe(
            attrs=("number", "value", "params", "state")
        ).copy()
        trials_df["best_so_far"] = trials_df["value"].cummax()
        trials_csv_path = os.path.join("artifacts", "trials_summary.csv")
        trials_df.to_csv(trials_csv_path, index=False)
        mlflow.log_artifact(trials_csv_path, artifact_path="study")

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_pipeline, artifact_path="best_model_mlflow")

        if cfg.mlflow.register_model:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model_mlflow"
            result = mlflow.register_model(model_uri=model_uri, name=cfg.mlflow.model_name)
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=cfg.mlflow.model_name,
                version=result.version,
                stage=cfg.mlflow.stage,
            )

        print(f"Best {cfg.hpo.metric}: {best_value:.5f}")
        print(f"Best params: {best_params}")


if __name__ == "__main__":
    preprocess_cli_args()
    main()
