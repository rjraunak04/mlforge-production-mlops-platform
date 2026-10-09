from mlforge.monitoring.config import PerformanceThresholds
from mlforge.monitoring.performance import evaluate_performance


def test_performance_report_flags_degradation() -> None:
    report = evaluate_performance(
        [0, 0, 1, 1],
        [0.9, 0.8, 0.2, 0.1],
        threshold=0.5,
        limits=PerformanceThresholds(0.75, 0.70, 0.45),
    )
    assert report.degraded
    assert report.recall == 0.0
