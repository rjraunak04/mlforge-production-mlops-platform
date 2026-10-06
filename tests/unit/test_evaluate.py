"""Tests for validation-only model evaluation and comparison."""

from __future__ import annotations

import pandas as pd
import pytest

from mlforge.features.prepare import prepare_model_data
from mlforge.training.baseline import train_logistic_baseline
from mlforge.training.evaluate import (
    EvaluationResult,
    compare_validation_results,
    evaluate_classifier,
)


def _partition(rows: int = 40) -> pd.DataFrame:
    records = []
    for index in range(rows):
        records.append(
            {
                "customerID": f"C{index:03d}",
                "gender": "Female" if index % 2 == 0 else "Male",
                "SeniorCitizen": index % 2,
                "Partner": "Yes" if index % 3 == 0 else "No",
                "Dependents": "Yes" if index % 5 == 0 else "No",
                "tenure": index + 1,
                "PhoneService": "Yes",
                "MultipleLines": "Yes" if index % 2 else "No",
                "InternetService": "Fiber optic" if index % 3 else "DSL",
                "OnlineSecurity": "Yes" if index % 4 else "No",
                "OnlineBackup": "No" if index % 3 else "Yes",
                "DeviceProtection": "Yes" if index % 2 else "No",
                "TechSupport": "No" if index % 4 else "Yes",
                "StreamingTV": "Yes" if index % 2 else "No",
                "StreamingMovies": "No" if index % 2 else "Yes",
                "Contract": "Month-to-month" if index % 3 else "One year",
                "PaperlessBilling": "Yes" if index % 2 else "No",
                "PaymentMethod": ("Electronic check" if index % 2 else "Mailed check"),
                "MonthlyCharges": 20.0 + index,
                "TotalCharges": (20.0 + index) * (index + 1),
                "Churn": "Yes" if index % 4 == 0 else "No",
            }
        )
    return pd.DataFrame.from_records(records)


def _result(name: str, roc_auc: float, partition: str = "validation"):
    return EvaluationResult(
        model_name=name,
        partition=partition,
        roc_auc=roc_auc,
        average_precision=roc_auc,
        f1=0.5,
        precision=0.5,
        recall=0.5,
        accuracy=0.5,
        confusion_matrix=((10, 2), (3, 5)),
    )


def test_evaluation_returns_required_metrics() -> None:
    train = prepare_model_data(_partition(40))
    validation = prepare_model_data(_partition(20).assign(
        customerID=lambda frame: "V" + frame["customerID"]
    ))
    trained = train_logistic_baseline(train)

    result = evaluate_classifier(
        "logistic_regression",
        trained.pipeline,
        validation,
    )

    assert result.partition == "validation"
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.average_precision <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.accuracy <= 1.0
    assert sum(sum(row) for row in result.confusion_matrix) == 20


def test_comparison_selects_highest_validation_metric() -> None:
    comparison = compare_validation_results(
        [_result("model_b", 0.75), _result("model_a", 0.82)]
    )

    assert comparison.selected_model == "model_a"
    assert [item.model_name for item in comparison.ranking] == ["model_a", "model_b"]


def test_comparison_uses_model_name_as_deterministic_tie_breaker() -> None:
    comparison = compare_validation_results(
        [_result("model_b", 0.80), _result("model_a", 0.80)]
    )

    assert comparison.selected_model == "model_a"


def test_test_partition_cannot_be_used_for_model_selection() -> None:
    with pytest.raises(ValueError, match="restricted to validation"):
        compare_validation_results([_result("model_a", 0.90, partition="test")])


def test_invalid_threshold_is_rejected() -> None:
    train = prepare_model_data(_partition())
    trained = train_logistic_baseline(train)

    with pytest.raises(ValueError, match="threshold"):
        evaluate_classifier(
            "logistic_regression",
            trained.pipeline,
            train,
            threshold=1.1,
        )
