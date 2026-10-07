"""Tests for MLflow model and reproducibility artifact logging."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
            experiment_name="mlforge-artifact-test",
            registry_uri=uri,
        ),
        registry=ModelRegistryConfig(model_name="MLForgeTestClassifier"),
        tags={"project": "mlforge", "dataset": "synthetic"},
    )


def _pipeline() -> Pipeline:
    frame = pd.DataFrame({"tenure": [1.0, 2.0, 8.0, 12.0]})
    target = pd.Series([0, 0, 1, 1])
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [("numeric", StandardScaler(), ["tenure"])],
                    remainder="drop",
                ),
            ),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    return pipeline.fit(frame, target)


def _result() -> EvaluationResult:
    return EvaluationResult(
        model_name="logistic_regression",
        partition="validation",
        roc_auc=0.84,
        average_precision=0.63,
        f1=0.61,
        precision=0.64,
        recall=0.59,
        accuracy=0.80,
        confusion_matrix=((70, 10), (12, 28)),
    )


def test_trained_pipeline_and_metadata_are_logged(tmp_path: Path) -> None:
    config = _config(tmp_path)
    experiment_id = configure_mlflow(config)

    tracked = track_validation_run(
        config=config,
        experiment_id=experiment_id,
        result=_result(),
        model_config={"enabled": True, "max_iter": 1000},
        random_state=42,
        classification_threshold=0.5,
        training_rows=4930,
        validation_rows=1056,
        pipeline=_pipeline(),
        selection_metric="roc_auc",
    )

    assert tracked.model_uri == f"runs:/{tracked.run_id}/model"
    loaded = mlflow.sklearn.load_model(tracked.model_uri)
    probabilities = loaded.predict_proba(pd.DataFrame({"tenure": [4.0]}))
    assert probabilities.shape == (1, 2)

    client = MlflowClient()
    metadata_path = client.download_artifacts(
        tracked.run_id,
        "metadata/run_metadata.json",
        dst_path=str(tmp_path / "download"),
    )
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    assert metadata == {
        "classification_threshold": 0.5,
        "model_name": "logistic_regression",
        "random_state": 42,
        "selection_metric": "roc_auc",
        "training_rows": 4930,
        "validation_rows": 1056,
    }


def test_model_artifact_is_attached_to_same_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    experiment_id = configure_mlflow(config)
    tracked = track_validation_run(
        config=config,
        experiment_id=experiment_id,
        result=_result(),
        model_config={"enabled": True},
        random_state=42,
        classification_threshold=0.5,
        training_rows=20,
        validation_rows=8,
        pipeline=_pipeline(),
    )

    artifacts = MlflowClient().list_artifacts(tracked.run_id)
    artifact_paths = {artifact.path for artifact in artifacts}

    assert "metadata" in artifact_paths
    assert tracked.model_uri is not None
