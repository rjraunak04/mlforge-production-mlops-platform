"""Configuration contracts for MLflow tracking and registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MLflowTrackingConfig:
    """Validated MLflow tracking settings."""

    uri: str
    artifact_root: str
    experiment_name: str
    registry_uri: str


@dataclass(frozen=True)
class ModelRegistryConfig:
    """Validated model-registry settings."""

    model_name: str


@dataclass(frozen=True)
class MLflowConfig:
    """Complete Day 3 MLflow configuration."""

    tracking: MLflowTrackingConfig
    registry: ModelRegistryConfig
    tags: dict[str, str]


def _required(mapping: dict[str, Any], key: str, section: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section}.{key} must be a non-empty string.")
    return value


def load_mlflow_config(path: str | Path = "configs/mlflow.yaml") -> MLflowConfig:
    """Load and validate MLflow configuration from YAML."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"MLflow configuration not found: {source}")

    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    if not isinstance(raw, dict):
        raise ValueError("MLflow configuration must be a mapping.")

    tracking = raw.get("tracking")
    registry = raw.get("registry")
    tags = raw.get("tags", {})

    if not isinstance(tracking, dict):
        raise ValueError("tracking must be a mapping.")
    if not isinstance(registry, dict):
        raise ValueError("registry must be a mapping.")
    if not isinstance(tags, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in tags.items()
    ):
        raise ValueError("tags must contain string keys and string values.")

    return MLflowConfig(
        tracking=MLflowTrackingConfig(
            uri=_required(tracking, "uri", "tracking"),
            artifact_root=_required(tracking, "artifact_root", "tracking"),
            experiment_name=_required(tracking, "experiment_name", "tracking"),
            registry_uri=_required(tracking, "registry_uri", "tracking"),
        ),
        registry=ModelRegistryConfig(
            model_name=_required(registry, "model_name", "registry"),
        ),
        tags=dict(tags),
    )
