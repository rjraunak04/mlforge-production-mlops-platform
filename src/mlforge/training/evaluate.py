"""Validation-only model evaluation and deterministic comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from mlforge.features.prepare import ModelData


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics for one model on a named evaluation partition."""

    model_name: str
    partition: str
    roc_auc: float
    average_precision: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class ModelComparison:
    """Validation-only ranking and selected model name."""

    selection_metric: str
    selected_model: str
    ranking: tuple[EvaluationResult, ...]


def evaluate_classifier(
    model_name: str,
    pipeline: Pipeline,
    evaluation_data: ModelData,
    *,
    threshold: float = 0.50,
    partition: str = "validation",
) -> EvaluationResult:
    """Evaluate a fitted binary classifier without refitting it."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Classification threshold must be between 0 and 1.")
    if evaluation_data.target.nunique() != 2:
        raise ValueError("Evaluation requires both target classes.")

    probabilities = pipeline.predict_proba(evaluation_data.features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(evaluation_data.target, predictions, labels=[0, 1])

    return EvaluationResult(
        model_name=model_name,
        partition=partition,
        roc_auc=float(roc_auc_score(evaluation_data.target, probabilities)),
        average_precision=float(
            average_precision_score(evaluation_data.target, probabilities)
        ),
        f1=float(f1_score(evaluation_data.target, predictions, zero_division=0)),
        precision=float(
            precision_score(evaluation_data.target, predictions, zero_division=0)
        ),
        recall=float(recall_score(evaluation_data.target, predictions, zero_division=0)),
        accuracy=float(accuracy_score(evaluation_data.target, predictions)),
        confusion_matrix=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
    )


def compare_validation_results(
    results: list[EvaluationResult],
    *,
    selection_metric: str = "roc_auc",
) -> ModelComparison:
    """Rank validation results and select the best model deterministically."""
    if not results:
        raise ValueError("At least one evaluation result is required.")
    if any(result.partition != "validation" for result in results):
        raise ValueError("Model selection is restricted to validation results.")

    allowed_metrics = {
        "roc_auc",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "accuracy",
    }
    if selection_metric not in allowed_metrics:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")

    ranking = tuple(
        sorted(
            results,
            key=lambda result: (-getattr(result, selection_metric), result.model_name),
        )
    )
    return ModelComparison(
        selection_metric=selection_metric,
        selected_model=ranking[0].model_name,
        ranking=ranking,
    )
