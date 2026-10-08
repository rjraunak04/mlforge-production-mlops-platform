"""Delayed-label production performance monitoring."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from mlforge.monitoring.config import PerformanceThresholds


@dataclass(frozen=True)
class PerformanceReport:
    rows: int
    roc_auc: float
    accuracy: float
    recall: float
    degraded: bool


def evaluate_performance(
    y_true: list[int],
    probabilities: list[float],
    *,
    threshold: float,
    limits: PerformanceThresholds,
) -> PerformanceReport:
    if len(y_true) != len(probabilities) or not y_true:
        raise ValueError("Labels and probabilities must be non-empty and aligned.")
    labels = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Performance monitoring requires both binary classes.")
    predicted = (scores >= threshold).astype(int)
    auc = float(roc_auc_score(labels, scores))
    accuracy = float(accuracy_score(labels, predicted))
    recall = float(recall_score(labels, predicted))
    return PerformanceReport(
        rows=len(labels),
        roc_auc=auc,
        accuracy=accuracy,
        recall=recall,
        degraded=(
            auc < limits.min_roc_auc
            or accuracy < limits.min_accuracy
            or recall < limits.min_recall
        ),
    )
