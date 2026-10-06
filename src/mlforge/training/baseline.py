"""Reproducible Logistic Regression baseline training."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlforge.features.prepare import ModelData
from mlforge.features.preprocessing import build_preprocessor


@dataclass(frozen=True)
class TrainedBaseline:
    """Fitted baseline pipeline and basic training metadata."""

    pipeline: Pipeline
    training_rows: int
    positive_rows: int


def build_logistic_baseline(
    *,
    max_iter: int = 1000,
    class_weight: str | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build an unfitted preprocessing + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_logistic_baseline(
    train_data: ModelData,
    *,
    max_iter: int = 1000,
    class_weight: str | None = None,
    random_state: int = 42,
) -> TrainedBaseline:
    """Fit the baseline on training data only."""
    if len(train_data.features) != len(train_data.target):
        raise ValueError("Training features and target have different row counts.")
    if train_data.target.nunique() != 2:
        raise ValueError("Baseline training requires both target classes.")

    pipeline = build_logistic_baseline(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )
    pipeline.fit(train_data.features, train_data.target)

    return TrainedBaseline(
        pipeline=pipeline,
        training_rows=len(train_data.target),
        positive_rows=int(train_data.target.sum()),
    )
