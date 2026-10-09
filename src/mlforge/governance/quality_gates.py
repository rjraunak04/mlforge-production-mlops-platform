"""Config-driven validation quality gates for model governance."""

from __future__ import annotations

from dataclasses import dataclass

from mlforge.training.evaluate import EvaluationResult


@dataclass(frozen=True)
class QualityGateThresholds:
    """Minimum validation metrics required before model registration."""

    min_roc_auc: float
    min_average_precision: float
    min_recall: float

    def __post_init__(self) -> None:
        for name, value in (
            ("min_roc_auc", self.min_roc_auc),
            ("min_average_precision", self.min_average_precision),
            ("min_recall", self.min_recall),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")


@dataclass(frozen=True)
class QualityGateFailure:
    """One failed metric requirement."""

    metric: str
    actual: float
    minimum: float


@dataclass(frozen=True)
class QualityGateResult:
    """Machine-readable model acceptance decision."""

    model_name: str
    passed: bool
    failures: tuple[QualityGateFailure, ...]


def evaluate_quality_gates(
    result: EvaluationResult,
    thresholds: QualityGateThresholds,
) -> QualityGateResult:
    """Evaluate validation metrics against minimum governance thresholds."""
    if result.partition != "validation":
        raise ValueError("Quality gates may only evaluate validation results.")

    requirements = (
        ("roc_auc", result.roc_auc, thresholds.min_roc_auc),
        (
            "average_precision",
            result.average_precision,
            thresholds.min_average_precision,
        ),
        ("recall", result.recall, thresholds.min_recall),
    )
    failures = tuple(
        QualityGateFailure(metric=metric, actual=actual, minimum=minimum)
        for metric, actual, minimum in requirements
        if actual < minimum
    )
    return QualityGateResult(
        model_name=result.model_name,
        passed=not failures,
        failures=failures,
    )
