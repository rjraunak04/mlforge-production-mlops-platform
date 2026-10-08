from pathlib import Path

import pytest

from mlforge.serving.config import load_serving_config


CONFIG = """
service:
  name: mlforge-api
  host: 0.0.0.0
  port: 8000
model:
  name: MLForgeChurnClassifier
  alias: champion
  classification_threshold: 0.5
mlflow:
  tracking_uri: sqlite:///mlruns/mlflow.db
  registry_uri: sqlite:///mlruns/mlflow.db
runtime:
  log_level: INFO
  request_timeout_seconds: 30
"""


def test_load_serving_config(tmp_path: Path) -> None:
    path = tmp_path / "serving.yaml"
    path.write_text(CONFIG, encoding="utf-8")

    config = load_serving_config(path)

    assert config.model.name == "MLForgeChurnClassifier"
    assert config.model.alias == "champion"
    assert config.model.classification_threshold == 0.5
    assert config.service.port == 8000


def test_serving_config_rejects_invalid_threshold(tmp_path: Path) -> None:
    path = tmp_path / "serving.yaml"
    path.write_text(
        CONFIG.replace("classification_threshold: 0.5", "classification_threshold: 1.5"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="classification_threshold"):
        load_serving_config(path)
