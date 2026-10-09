"""Translate monitoring evidence into a safe retraining decision."""

from dataclasses import dataclass

from mlforge.monitoring.report import MonitoringReport
from mlforge.retraining.config import RetrainingConfig


@dataclass(frozen=True)
class RetrainingDecision:
    should_retrain: bool
    reason: str


def decide_retraining(
    monitoring: MonitoringReport | None,
    config: RetrainingConfig,
) -> RetrainingDecision:
    if not config.enabled:
        return RetrainingDecision(False, "Automated retraining is disabled.")
    if monitoring is None:
        if config.require_monitoring_signal:
            return RetrainingDecision(False, "Monitoring evidence is required.")
        return RetrainingDecision(True, "Scheduled retraining is allowed.")
    if monitoring.retraining_recommended:
        return RetrainingDecision(True, "Monitoring recommends candidate retraining.")
    return RetrainingDecision(
        False, "Monitoring evidence is healthy; retraining skipped."
    )
