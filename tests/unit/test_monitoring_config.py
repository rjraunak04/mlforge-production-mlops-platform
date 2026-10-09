from pathlib import Path

import pytest

from mlforge.monitoring.config import load_monitoring_config


def test_load_monitoring_config(tmp_path: Path) -> None:
    path = tmp_path / "monitoring.yaml"
    path.write_text(
        """monitoring:
  reference_path: data/reference/ref.csv
  production_log_path: monitoring/predictions.jsonl
  report_dir: reports/monitoring
  min_batch_size: 50
  drift_share_threshold: 0.3
  prediction_drift_threshold: 0.1
  performance:
    min_roc_auc: 0.75
    min_accuracy: 0.7
    min_recall: 0.45
""",
        encoding="utf-8",
    )
    config = load_monitoring_config(path)
    assert config.min_batch_size == 50
    assert config.performance.min_roc_auc == 0.75


def test_monitoring_config_rejects_invalid_threshold(tmp_path: Path) -> None:
    path = tmp_path / "monitoring.yaml"
    path.write_text(
        """monitoring:
  reference_path: ref.csv
  production_log_path: predictions.jsonl
  report_dir: reports
  min_batch_size: 1
  drift_share_threshold: 2
  prediction_drift_threshold: 0.1
  performance: {min_roc_auc: 0.7, min_accuracy: 0.7, min_recall: 0.4}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="drift_share_threshold"):
        load_monitoring_config(path)
