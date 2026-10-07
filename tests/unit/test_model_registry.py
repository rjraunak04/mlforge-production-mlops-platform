"""Tests for quality-governed MLflow model registration."""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import pytest
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier

from mlforge.governance.quality_gates import QualityGateResult
from mlforge.registry.model_registry import register_approved_candidate
from mlforge.tracking.config import (
    MLflowConfig,
    MLflowTrackingConfig,
    ModelRegistryConfig,
)
from mlforge.tracking.experiments import TrackedRun


def _config(tmp_path: Path) -> MLflowConfig:
    database = (tmp_path / "mlflow.db").as_posix()
    uri = f"sqlite:///{database}"
    return MLflowConfig(
        tracking=MLflowTrackingConfig(
            uri=uri,
            artifact_root=(tmp_path / "artifacts").resolve().as_uri(),
            experiment_name="registry-test",
            registry_uri=uri,
        ),
        registry=ModelRegistryConfig(model_name="MLForgeTestClassifier"),
        tags={"project": "mlforge"},
    )


def _logged_run(config: MLflowConfig) -> TrackedRun:
    mlflow.set_tracking_uri(config.tracking.uri)
    mlflow.set_registry_uri(config.tracking.registry_uri)
    experiment_id = mlflow.create_experiment(
        config.tracking.experiment_name,
        artifact_location=config.tracking.artifact_root,
    )
    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(model, name="model")
        return TrackedRun(
            model_name="logistic_regression",
            run_id=run.info.run_id,
            experiment_id=str(experiment_id),
            model_uri=f"runs:/{run.info.run_id}/model",
        )


def _gate(passed: bool = True) -> QualityGateResult:
    return QualityGateResult(
        model_name="logistic_regression",
        passed=passed,
        failures=(),
    )


def test_approved_candidate_creates_registered_model_version(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tracked = _logged_run(config)

    registered = register_approved_candidate(
        config=config,
        tracked_run=tracked,
        gate_result=_gate(),
    )

    assert registered.model_name == "MLForgeTestClassifier"
    assert registered.version == "1"
    version = MlflowClient(
        tracking_uri=config.tracking.uri,
        registry_uri=config.tracking.registry_uri,
    ).get_model_version(registered.model_name, registered.version)
    assert version.run_id == tracked.run_id
    assert version.tags["governance.status"] == "approved"


def test_second_approved_candidate_creates_new_version(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = _logged_run(config)
    first_registered = register_approved_candidate(
        config=config,
        tracked_run=first,
        gate_result=_gate(),
    )

    second = _logged_run(config)
    second_registered = register_approved_candidate(
        config=config,
        tracked_run=second,
        gate_result=_gate(),
    )

    assert first_registered.version == "1"
    assert second_registered.version == "2"


def test_failed_quality_gate_blocks_registration(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tracked = _logged_run(config)

    with pytest.raises(ValueError, match="blocks model registration"):
        register_approved_candidate(
            config=config,
            tracked_run=tracked,
            gate_result=_gate(passed=False),
        )

    client = MlflowClient(
        tracking_uri=config.tracking.uri,
        registry_uri=config.tracking.registry_uri,
    )
    assert client.search_registered_models() == []


def test_registration_requires_matching_model_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tracked = _logged_run(config)
    mismatched = QualityGateResult(
        model_name="random_forest",
        passed=True,
        failures=(),
    )

    with pytest.raises(ValueError, match="does not match"):
        register_approved_candidate(
            config=config,
            tracked_run=tracked,
            gate_result=mismatched,
        )


def test_registration_requires_logged_model_artifact(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tracked = TrackedRun(
        model_name="logistic_regression",
        run_id="run-without-model",
        experiment_id="0",
        model_uri=None,
    )

    with pytest.raises(ValueError, match="model artifact"):
        register_approved_candidate(
            config=config,
            tracked_run=tracked,
            gate_result=_gate(),
        )
