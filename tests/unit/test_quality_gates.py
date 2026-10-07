"""Tests for validation model quality gates."""

import pytest

from mlforge.governance.quality_gates import (
    QualityGateThresholds,
    evaluate_quality_gates,
)
from mlforge.training.evaluate import EvaluationResult


def _result(
    *,
    partition: str = "validation",
    roc_auc: float = 0.8453,
    average_precision: float = 0.6310,
    recall: float = 0.5929,
) -> EvaluationResult:
    return EvaluationResult(
        model_name="logistic_regression",
        partition=partition,
        roc_auc=roc_auc,
        average_precision=average_precision,
        f1=0.6182,
        precision=0.6459,
        recall=recall,
        accuracy=0.8059,
        confusion_matrix=((700, 80), (114, 162)),
    )


def _thresholds() -> QualityGateThresholds:
    return QualityGateThresholds(
        min_roc_auc=0.80,
        min_average_precision=0.55,
        min_recall=0.50,
    )


def test_day3_validation_winner_passes_quality_gates() -> None:
    decision = evaluate_quality_gates(_result(), _thresholds())

    assert decision.passed is True
    assert decision.failures == ()


def test_quality_gate_reports_every_failed_requirement() -> None:
    decision = evaluate_quality_gates(
        _result(roc_auc=0.79, average_precision=0.54, recall=0.49),
        _thresholds(),
    )

    assert decision.passed is False
    assert [failure.metric for failure in decision.failures] == [
        "roc_auc",
        "average_precision",
        "recall",
    ]
    assert decision.failures[0].actual == pytest.approx(0.79)
    assert decision.failures[0].minimum == pytest.approx(0.80)


def test_quality_gate_accepts_metrics_exactly_at_threshold() -> None:
    decision = evaluate_quality_gates(
        _result(roc_auc=0.80, average_precision=0.55, recall=0.50),
        _thresholds(),
    )

    assert decision.passed is True


def test_quality_gate_rejects_non_validation_partition() -> None:
    with pytest.raises(ValueError, match="only evaluate validation"):
        evaluate_quality_gates(_result(partition="test"), _thresholds())


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_quality_gate_thresholds_must_be_probabilities(value: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        QualityGateThresholds(
            min_roc_auc=value,
            min_average_precision=0.55,
            min_recall=0.50,
        )
