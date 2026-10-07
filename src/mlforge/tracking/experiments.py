"""MLflow experiment-run tracking for validation model comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow

from mlforge.tracking.config import MLflowConfig
from mlforge.training.evaluate import EvaluationResult


@dataclass(frozen=True)
class TrackedRun:
    """Stable identifiers for one completed MLflow model run."""

    model_name: str
    run_id: str
    experiment_id: str


def configure_mlflow(config: MLflowConfig) -> str:
    """Configure tracking/registry URIs and return the experiment ID."""
    mlflow.set_tracking_uri(config.tracking.uri)
    mlflow.set_registry_uri(config.tracking.registry_uri)

    experiment = mlflow.get_experiment_by_name(config.tracking.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config.tracking.experiment_name,
            artifact_location=config.tracking.artifact_root,
        )
    else:
        experiment_id = experiment.experiment_id
    return str(experiment_id)


def _flatten_params(
    model_name: str,
    model_config: dict[str, Any],
    *,
    random_state: int,
    classification_threshold: float,
) -> dict[str, Any]:
    """Build a compact, MLflow-safe parameter dictionary."""
    params: dict[str, Any] = {
        "model_name": model_name,
        "random_state": random_state,
        "classification_threshold": classification_threshold,
    }
    params.update(
        {
            f"model.{key}": value if value is not None else "None"
            for key, value in model_config.items()
            if key != "enabled"
        }
    )
    return params


def _metric_payload(result: EvaluationResult) -> dict[str, float]:
    return {
        "roc_auc": result.roc_auc,
        "average_precision": result.average_precision,
        "f1": result.f1,
        "precision": result.precision,
        "recall": result.recall,
        "accuracy": result.accuracy,
    }


def track_validation_run(
    *,
    config: MLflowConfig,
    experiment_id: str,
    result: EvaluationResult,
    model_config: dict[str, Any],
    random_state: int,
    classification_threshold: float,
    training_rows: int,
    validation_rows: int,
) -> TrackedRun:
    """Record one already-trained model's validation evidence in MLflow."""
    if result.partition != "validation":
        raise ValueError("Day 3 tracking is restricted to validation results.")

    tags = {
        **config.tags,
        "model_name": result.model_name,
        "evaluation_partition": result.partition,
    }
    params = _flatten_params(
        result.model_name,
        model_config,
        random_state=random_state,
        classification_threshold=classification_threshold,
    )
    params.update(
        {
            "training_rows": training_rows,
            "validation_rows": validation_rows,
        }
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=result.model_name,
        tags=tags,
    ) as active_run:
        mlflow.log_params(params)
        mlflow.log_metrics(_metric_payload(result))
        run_id = active_run.info.run_id

    return TrackedRun(
        model_name=result.model_name,
        run_id=run_id,
        experiment_id=str(experiment_id),
    )
