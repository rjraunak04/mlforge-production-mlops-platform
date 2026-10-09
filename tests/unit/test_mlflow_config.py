"""Tests for MLflow configuration loading."""

from pathlib import Path

import pytest
import yaml

from mlforge.tracking.config import load_mlflow_config


def test_repository_mlflow_config_loads() -> None:
    config = load_mlflow_config("configs/mlflow.yaml")

    assert config.tracking.experiment_name == "mlforge-churn"
    assert config.registry.model_name == "MLForgeChurnClassifier"
    assert config.quality_gates.min_roc_auc == 0.80
    assert config.tags["project"] == "mlforge"


def test_missing_required_mlflow_value_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mlflow.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tracking": {
                    "uri": "sqlite:///test.db",
                    "artifact_root": "artifacts",
                    "experiment_name": "",
                    "registry_uri": "sqlite:///test.db",
                },
                "registry": {"model_name": "Model"},
                "quality_gates": {
                    "min_roc_auc": 0.8,
                    "min_average_precision": 0.5,
                    "min_recall": 0.5,
                },
                "tags": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="experiment_name"):
        load_mlflow_config(path)


def test_non_string_mlflow_tags_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mlflow.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tracking": {
                    "uri": "sqlite:///test.db",
                    "artifact_root": "artifacts",
                    "experiment_name": "test",
                    "registry_uri": "sqlite:///test.db",
                },
                "registry": {"model_name": "Model"},
                "quality_gates": {
                    "min_roc_auc": 0.8,
                    "min_average_precision": 0.5,
                    "min_recall": 0.5,
                },
                "tags": {"bad": 42},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tags"):
        load_mlflow_config(path)
