from pathlib import Path

from mlforge.monitoring.drift import DriftSummary
from mlforge.monitoring.report import build_monitoring_report
from mlforge.retraining.config import RetrainingConfig
from mlforge.retraining.trigger import decide_retraining


def _config(enabled: bool = True) -> RetrainingConfig:
    return RetrainingConfig(
        enabled=enabled,
        require_monitoring_signal=True,
        data_path=Path("data.csv"),
        training_config_path=Path("training.yaml"),
        mlflow_config_path=Path("mlflow.yaml"),
        evidence_path=Path("report.json"),
    )


def test_retraining_starts_when_monitoring_recommends_it() -> None:
    report = build_monitoring_report(
        rows=100,
        feature_drift=DriftSummary((), 0.5, True),
        prediction_drift_score=0.0,
        prediction_drift_threshold=0.1,
    )
    assert decide_retraining(report, _config()).should_retrain


def test_retraining_skips_healthy_monitoring() -> None:
    report = build_monitoring_report(
        rows=100,
        feature_drift=DriftSummary((), 0.0, False),
        prediction_drift_score=0.0,
        prediction_drift_threshold=0.1,
    )
    assert not decide_retraining(report, _config()).should_retrain


def test_disabled_retraining_always_skips() -> None:
    assert not decide_retraining(None, _config(enabled=False)).should_retrain
