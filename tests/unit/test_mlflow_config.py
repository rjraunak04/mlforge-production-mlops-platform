"""Tests for MLflow configuration contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlforge.tracking.config import load_mlflow_config


def test_repository_mlflow_config_loads() -> None:
    config = load_mlflow_config("configs/mlflow.yaml")

    assert config.tracking.experiment_name == "mlforge-churn"
    assert config.registry.model_name == "MLForgeChurnClassifier"
    assert config.tags["project"] == "mlforge"


def test_mlflow_config_rejects_missing_required_value(tmp_path: Path) -> None:
    config_path = tmp_path / "mlflow.yaml"
    config_path.write_text(
        """
tracking:
  uri: "sqlite:///mlruns/mlflow.db"
  artifact_root: "mlruns/artifacts"
  experiment_name: ""
  registry_uri: "sqlite:///mlruns/mlflow.db"
registry:
  model_name: "MLForgeChurnClassifier"
tags:
  project: "mlforge"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="experiment_name"):
        load_mlflow_config(config_path)


def test_mlflow_config_rejects_non_string_tags(tmp_path: Path) -> None:
    config_path = tmp_path / "mlflow.yaml"
    config_path.write_text(
        """
tracking:
  uri: "sqlite:///mlruns/mlflow.db"
  artifact_root: "mlruns/artifacts"
  experiment_name: "mlforge-churn"
  registry_uri: "sqlite:///mlruns/mlflow.db"
registry:
  model_name: "MLForgeChurnClassifier"
tags:
  project: 42
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="string keys and string values"):
        load_mlflow_config(config_path)
