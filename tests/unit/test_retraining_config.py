from pathlib import Path

import pytest

from mlforge.retraining.config import load_retraining_config


def test_load_retraining_config(tmp_path: Path) -> None:
    source = tmp_path / "retraining.yaml"
    source.write_text(
        """retraining:
  enabled: true
  require_monitoring_signal: true
  data_path: data.csv
  training_config_path: training.yaml
  mlflow_config_path: mlflow.yaml
  evidence_path: evidence.json
""",
        encoding="utf-8",
    )
    config = load_retraining_config(source)
    assert config.enabled
    assert config.data_path == Path("data.csv")


def test_retraining_config_rejects_non_boolean_flag(tmp_path: Path) -> None:
    source = tmp_path / "retraining.yaml"
    source.write_text(
        """retraining:
  enabled: yes
  require_monitoring_signal: "true"
  data_path: data.csv
  training_config_path: training.yaml
  mlflow_config_path: mlflow.yaml
  evidence_path: evidence.json
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="require_monitoring_signal"):
        load_retraining_config(source)
