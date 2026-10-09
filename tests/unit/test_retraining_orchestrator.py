from pathlib import Path
from unittest.mock import patch

from mlforge.monitoring.drift import DriftSummary
from mlforge.monitoring.report import build_monitoring_report
from mlforge.retraining.config import RetrainingConfig
from mlforge.retraining.orchestrator import run_retraining


def _config() -> RetrainingConfig:
    return RetrainingConfig(
        enabled=True,
        require_monitoring_signal=True,
        data_path=Path("data.csv"),
        training_config_path=Path("training.yaml"),
        mlflow_config_path=Path("mlflow.yaml"),
        evidence_path=Path("report.json"),
    )


def test_healthy_monitoring_does_not_train() -> None:
    report = build_monitoring_report(
        rows=100,
        feature_drift=DriftSummary((), 0.0, False),
        prediction_drift_score=0.0,
        prediction_drift_threshold=0.1,
    )
    with patch(
        "mlforge.retraining.orchestrator.run_governed_training"
    ) as governed:
        result = run_retraining(_config(), report)
    governed.assert_not_called()
    assert result.governance is None
