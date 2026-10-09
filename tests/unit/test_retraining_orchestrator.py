from pathlib import Path
from unittest.mock import Mock, patch

from mlforge.monitoring.drift import DriftSummary
from mlforge.monitoring.report import build_monitoring_report
from mlforge.retraining.config import RetrainingConfig
from mlforge.retraining.orchestrator import (
    run_retraining,
    write_retraining_evidence,
)


def _config() -> RetrainingConfig:
    return RetrainingConfig(
        enabled=True,
        require_monitoring_signal=True,
        data_path=Path("data.csv"),
        training_config_path=Path("training.yaml"),
        mlflow_config_path=Path("mlflow.yaml"),
        evidence_path=Path("report.json"),
    )


def _report(drifted: bool):
    return build_monitoring_report(
        rows=100,
        feature_drift=DriftSummary((), 0.5 if drifted else 0.0, drifted),
        prediction_drift_score=0.0,
        prediction_drift_threshold=0.1,
    )


def test_healthy_monitoring_does_not_train() -> None:
    with patch("mlforge.retraining.orchestrator.run_governed_training") as governed:
        result = run_retraining(_config(), _report(False))
    governed.assert_not_called()
    assert result.governance is None


def test_drift_invokes_existing_governed_training() -> None:
    governed_run = Mock()
    with patch(
        "mlforge.retraining.orchestrator.run_governed_training",
        return_value=governed_run,
    ) as governed:
        result = run_retraining(_config(), _report(True))
    governed.assert_called_once_with(
        Path("data.csv"),
        training_config_path=Path("training.yaml"),
        mlflow_config_path=Path("mlflow.yaml"),
    )
    assert result.governance is governed_run


def test_retraining_evidence_records_skipped_run(tmp_path: Path) -> None:
    result = run_retraining(_config(), _report(False))
    destination = write_retraining_evidence(result, tmp_path / "evidence.json")
    content = destination.read_text(encoding="utf-8")
    assert '"should_retrain": false' in content
    assert '"candidate_version": null' in content
