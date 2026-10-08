"""Serializable production monitoring reports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mlforge.monitoring.drift import DriftSummary
from mlforge.monitoring.performance import PerformanceReport


@dataclass(frozen=True)
class MonitoringReport:
    rows: int
    feature_drift: DriftSummary
    prediction_drift_score: float
    prediction_drifted: bool
    performance: PerformanceReport | None
    retraining_recommended: bool


def build_monitoring_report(
    *,
    rows: int,
    feature_drift: DriftSummary,
    prediction_drift_score: float,
    prediction_drift_threshold: float,
    performance: PerformanceReport | None = None,
) -> MonitoringReport:
    prediction_drifted = prediction_drift_score >= prediction_drift_threshold
    degraded = performance.degraded if performance is not None else False
    return MonitoringReport(
        rows=rows,
        feature_drift=feature_drift,
        prediction_drift_score=prediction_drift_score,
        prediction_drifted=prediction_drifted,
        performance=performance,
        retraining_recommended=feature_drift.dataset_drifted or prediction_drifted or degraded,
    )


def save_monitoring_report(report: MonitoringReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return destination
