"""Registry-backed champion model loading for production inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mlforge.serving.config import ServingConfig


class ModelLoadError(RuntimeError):
    """Raised when the configured production model cannot be loaded safely."""


@dataclass(frozen=True)
class LoadedModel:
    """Loaded model plus immutable registry identity used for inference."""

    pipeline: Any
    model_name: str
    alias: str
    version: str
    run_id: str
    model_uri: str


def load_registry_model(config: ServingConfig) -> LoadedModel:
    """Resolve one configured registry alias and load that exact model version."""
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_registry_uri(config.mlflow.registry_uri)
    client = MlflowClient(
        tracking_uri=config.mlflow.tracking_uri,
        registry_uri=config.mlflow.registry_uri,
    )

    try:
        version = client.get_model_version_by_alias(
            config.model.name,
            config.model.alias,
        )
    except MlflowException as exc:
        raise ModelLoadError(
            f"Unable to resolve model alias {config.model.name}@{config.model.alias}."
        ) from exc

    model_uri = f"models:/{config.model.name}/{version.version}"
    try:
        pipeline = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        raise ModelLoadError(
            f"Unable to load registered model {config.model.name} "
            f"version {version.version}."
        ) from exc

    return LoadedModel(
        pipeline=pipeline,
        model_name=config.model.name,
        alias=config.model.alias,
        version=str(version.version),
        run_id=version.run_id,
        model_uri=model_uri,
    )
