"""End-to-end Day 2 training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sklearn.pipeline import Pipeline

from mlforge.data.load import load_churn_data
from mlforge.data.split import DatasetSplits, split_churn_data
from mlforge.features.prepare import ModelData, prepare_model_data
from mlforge.training.baseline import train_logistic_baseline
from mlforge.training.candidates import train_candidate
from mlforge.training.evaluate import (
    EvaluationResult,
    ModelComparison,
    compare_validation_results,
    evaluate_classifier,
)


@dataclass(frozen=True)
class TrainingRun:
    """Outputs from a validation-only model-selection run."""

    splits: DatasetSplits
    train: ModelData
    validation: ModelData
    locked_test: ModelData
    models: dict[str, Pipeline]
    validation_results: tuple[EvaluationResult, ...]
    comparison: ModelComparison


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load the Day 2 YAML training configuration."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a mapping.")
    return config


def run_training_pipeline(
    data_path: str | Path,
    *,
    training_config_path: str | Path = "configs/training.yaml",
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> TrainingRun:
    """Run load, split, train, validation evaluation, and model selection."""
    config = load_training_config(training_config_path)
    training = config["training"]
    models_config = config["models"]
    evaluation = config["evaluation"]
    seed = int(training["random_state"])

    data = load_churn_data(data_path)
    splits = split_churn_data(
        data,
        target_column=training["target_column"],
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        random_state=seed,
    )

    preparation = {
        "target_column": training["target_column"],
        "id_column": training["id_column"],
        "negative_label": training["negative_label"],
        "positive_label": training["positive_label"],
    }
    train = prepare_model_data(splits.train, **preparation)
    validation = prepare_model_data(splits.validation, **preparation)
    locked_test = prepare_model_data(splits.test, **preparation)

    fitted_models: dict[str, Pipeline] = {}

    logistic = models_config["logistic_regression"]
    if logistic["enabled"]:
        trained = train_logistic_baseline(
            train,
            max_iter=int(logistic["max_iter"]),
            class_weight=logistic["class_weight"],
            random_state=seed,
        )
        fitted_models["logistic_regression"] = trained.pipeline

    random_forest = models_config["random_forest"]
    if random_forest["enabled"]:
        trained = train_candidate(
            "random_forest",
            train,
            n_estimators=int(random_forest["n_estimators"]),
            max_depth=random_forest["max_depth"],
            min_samples_leaf=int(random_forest["min_samples_leaf"]),
            class_weight=random_forest["class_weight"],
            n_jobs=int(random_forest["n_jobs"]),
            random_state=seed,
        )
        fitted_models["random_forest"] = trained.pipeline

    boosting = models_config["hist_gradient_boosting"]
    if boosting["enabled"]:
        trained = train_candidate(
            "hist_gradient_boosting",
            train,
            max_iter=int(boosting["max_iter"]),
            learning_rate=float(boosting["learning_rate"]),
            max_leaf_nodes=int(boosting["max_leaf_nodes"]),
            random_state=seed,
        )
        fitted_models["hist_gradient_boosting"] = trained.pipeline

    if not fitted_models:
        raise ValueError("At least one model must be enabled for training.")

    validation_results = tuple(
        evaluate_classifier(
            name,
            pipeline,
            validation,
            threshold=float(evaluation["classification_threshold"]),
            partition="validation",
        )
        for name, pipeline in fitted_models.items()
    )
    comparison = compare_validation_results(
        list(validation_results),
        selection_metric=evaluation["selection_metric"],
    )

    return TrainingRun(
        splits=splits,
        train=train,
        validation=validation,
        locked_test=locked_test,
        models=fitted_models,
        validation_results=validation_results,
        comparison=comparison,
    )
