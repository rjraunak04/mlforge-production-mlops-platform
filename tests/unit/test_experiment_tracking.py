"""Tests for MLflow validation experiment tracking."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from mlforge.tracking.config import (
    MLflowConfig,
    MLflowTrackingConfig,
    ModelRegistryConfig,
)
from mlforge.tracking.experiments import configure_mlflow, track_validation_run
from mlforge.training.evaluate import EvaluationResult


def _config(tmp_path: Path) -> MLflowConfig:
    database = (tmp_path / "mlflow.db").as_posix()
    artifacts = (tmp_path / "artifacts").resolve().as_uri()
    uri = f"sqlite:///{database}"
    return MLflowConfig(
        tracking=MLflowTrackingConfig(
            uri=uri,
            artifact_root=artifacts,
            experiment_name="mlforge-test",
            registry_uri=uri,
        ),
        registry=ModelRegistryConfig(model_name="MLForgeTestClassifier"),
        tags={"project": "mlforge", "dataset": "synthetic"},
    )


def _result(partition: str = "validation") -> EvaluationResult:
    return EvaluationResult(
        model_name="logistic_regression",
        partition=partition,
        roc_auc=0.84,
        average_precision=0.63,
        f1=0.61,
        precision=0.64,
        recall=0.59,
        accuracy=0.80,
        confusion_matrix=((70, 10), (12, 28)),
    )


def test_configure_mlflow_creates_and_reuses_experiment(tmp_path: Path) -> None:
    config = _config(tmp_path)

    first = configure_mlflow(config)
    second = configure_mlflow(config)

    assert first == second
    experiment = MlflowClient().get_experiment(first)
    assert experiment.name == "mlforge-test"


def test_validation_run_logs_params_metrics_and_tags(tmp_path: Path) -> None:
    config = _config(tmp_path)
    experiment_id = configure_mlflow(config)

    tracked = track_validation_run(
        config=config,
        experiment_id=experiment_id,
        result=_result(),
        model_config={"enabled": True, "max_iter": 1000, "class_weight": None},
        random_state=42,
        classification_threshold=0.5,
        training_rows=4930,
        validation_rows=1056,
    )

    run = MlflowClient().get_run(tracked.run_id)
    assert run.data.params["model_name"] == "logistic_regression"
    assert run.data.params["model.max_iter"] == "1000"
    assert run.data.params["training_rows"] == "4930"
    assert run.data.metrics["roc_auc"] == pytest.approx(0.84)
    assert run.data.metrics["average_precision"] == pytest.approx(0.63)
    assert run.data.tags["project"] == "mlforge"
    assert run.data.tags["evaluation_partition"] == "validation"


def test_tracking_rejects_non_validation_result(tmp_path: Path) -> None:
    config = _config(tmp_path)
    experiment_id = configure_mlflow(config)

    with pytest.raises(ValueError, match="restricted to validation"):
        track_validation_run(
            config=config,
            experiment_id=experiment_id,
            result=_result(partition="test"),
            model_config={"enabled": True},
            random_state=42,
            classification_threshold=0.5,
            training_rows=10,
            validation_rows=4,
        )

    assert mlflow.active_run() is None
