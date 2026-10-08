"""Configuration contracts for the production inference service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ServiceConfig:
    """HTTP service settings."""

    name: str
    host: str
    port: int


@dataclass(frozen=True)
class ServingModelConfig:
    """Registry-backed model selection settings."""

    name: str
    alias: str
    classification_threshold: float


@dataclass(frozen=True)
class ServingMLflowConfig:
    """MLflow locations required by the serving process."""

    tracking_uri: str
    registry_uri: str


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime safeguards for the serving process."""

    log_level: str
    request_timeout_seconds: int


@dataclass(frozen=True)
class ServingConfig:
    """Complete validated Day 04 serving configuration."""

    service: ServiceConfig
    model: ServingModelConfig
    mlflow: ServingMLflowConfig
    runtime: RuntimeConfig


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _text(section: dict[str, Any], key: str, name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}.{key} must be a non-empty string.")
    return value.strip()


def load_serving_config(path: str | Path = "configs/serving.yaml") -> ServingConfig:
    """Load and validate the production serving YAML contract."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Serving configuration not found: {source}")

    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Serving configuration must be a mapping.")

    service = _mapping(raw, "service")
    model = _mapping(raw, "model")
    mlflow = _mapping(raw, "mlflow")
    runtime = _mapping(raw, "runtime")

    port = service.get("port")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("service.port must be an integer between 1 and 65535.")

    threshold = model.get("classification_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError("model.classification_threshold must be numeric.")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("model.classification_threshold must be between 0 and 1.")

    timeout = runtime.get("request_timeout_seconds")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("runtime.request_timeout_seconds must be a positive integer.")

    return ServingConfig(
        service=ServiceConfig(
            name=_text(service, "name", "service"),
            host=_text(service, "host", "service"),
            port=port,
        ),
        model=ServingModelConfig(
            name=_text(model, "name", "model"),
            alias=_text(model, "alias", "model"),
            classification_threshold=threshold,
        ),
        mlflow=ServingMLflowConfig(
            tracking_uri=_text(mlflow, "tracking_uri", "mlflow"),
            registry_uri=_text(mlflow, "registry_uri", "mlflow"),
        ),
        runtime=RuntimeConfig(
            log_level=_text(runtime, "log_level", "runtime"),
            request_timeout_seconds=timeout,
        ),
    )
