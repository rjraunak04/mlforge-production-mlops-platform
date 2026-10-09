from mlforge.monitoring.drift import DriftSummary
from mlforge.monitoring.report import build_monitoring_report


def test_monitoring_report_recommends_retraining_on_prediction_drift() -> None:
    report = build_monitoring_report(
        rows=100,
        feature_drift=DriftSummary((), 0.0, False),
        prediction_drift_score=0.2,
        prediction_drift_threshold=0.1,
    )
    assert report.prediction_drifted
    assert report.retraining_recommended
