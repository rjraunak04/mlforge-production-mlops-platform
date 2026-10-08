from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException

from mlforge.serving.config import (
    RuntimeConfig,
    ServiceConfig,
    ServingConfig,
    ServingMLflowConfig,
    ServingModelConfig,
)
from mlforge.serving.model_loader import ModelLoadError, load_registry_model


def _config() -> ServingConfig:
    return ServingConfig(
        service=ServiceConfig("api", "0.0.0.0", 8000),
        model=ServingModelConfig("MLForgeChurnClassifier", "champion", 0.5),
        mlflow=ServingMLflowConfig("sqlite:///tracking.db", "sqlite:///registry.db"),
        runtime=RuntimeConfig("INFO", 30),
    )


def test_load_registry_model_resolves_alias_before_loading(monkeypatch) -> None:
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        version="7", run_id="run-123"
    )
    pipeline = object()
    monkeypatch.setattr("mlforge.serving.model_loader.MlflowClient", lambda **_: client)
    monkeypatch.setattr("mlforge.serving.model_loader.mlflow.set_tracking_uri", Mock())
    monkeypatch.setattr("mlforge.serving.model_loader.mlflow.set_registry_uri", Mock())
    load_model = Mock(return_value=pipeline)
    monkeypatch.setattr(
        "mlforge.serving.model_loader.mlflow.sklearn.load_model", load_model
    )

    loaded = load_registry_model(_config())

    client.get_model_version_by_alias.assert_called_once_with(
        "MLForgeChurnClassifier", "champion"
    )
    load_model.assert_called_once_with("models:/MLForgeChurnClassifier/7")
    assert loaded.pipeline is pipeline
    assert loaded.version == "7"
    assert loaded.run_id == "run-123"


def test_load_registry_model_fails_when_alias_cannot_be_resolved(monkeypatch) -> None:
    client = Mock()
    client.get_model_version_by_alias.side_effect = MlflowException("missing alias")
    monkeypatch.setattr("mlforge.serving.model_loader.MlflowClient", lambda **_: client)
    monkeypatch.setattr("mlforge.serving.model_loader.mlflow.set_tracking_uri", Mock())
    monkeypatch.setattr("mlforge.serving.model_loader.mlflow.set_registry_uri", Mock())

    with pytest.raises(ModelLoadError, match="Unable to resolve model alias"):
        load_registry_model(_config())


def test_load_registry_model_wraps_artifact_load_failure(monkeypatch) -> None:
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        version="2", run_id="run-456"
    )
    monkeypatch.setattr("mlforge.serving.model_loader.MlflowClient", lambda **_: client)
    monkeypatch.setattr("mlforge.serving.model_loader.mlflow.set_tracking_uri", Mock())
    monkeypatch.setattr("mlforge.serving.model_loader.mlflow.set_registry_uri", Mock())
    monkeypatch.setattr(
        "mlforge.serving.model_loader.mlflow.sklearn.load_model",
        Mock(side_effect=OSError("artifact unavailable")),
    )

    with pytest.raises(ModelLoadError, match="Unable to load registered model"):
        load_registry_model(_config())
