"""Candidate model builders and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from mlforge.features.prepare import ModelData
from mlforge.features.preprocessing import build_preprocessor

SUPPORTED_CANDIDATES = ("random_forest", "hist_gradient_boosting")


@dataclass(frozen=True)
class TrainedCandidate:
    """A fitted candidate pipeline with training metadata."""

    name: str
    pipeline: Pipeline
    training_rows: int
    positive_rows: int


def build_random_forest_candidate(
    *,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    class_weight: str | None = "balanced",
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    """Build an unfitted Random Forest candidate pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                    n_jobs=n_jobs,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting_candidate(
    *,
    max_iter: int = 200,
    learning_rate: float = 0.1,
    max_leaf_nodes: int = 31,
    random_state: int = 42,
) -> Pipeline:
    """Build an unfitted histogram gradient boosting candidate pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_candidate(name: str, **params: Any) -> Pipeline:
    """Build a supported candidate by stable registry name."""
    builders = {
        "random_forest": build_random_forest_candidate,
        "hist_gradient_boosting": build_hist_gradient_boosting_candidate,
    }
    try:
        builder = builders[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported candidate '{name}'. Expected one of {SUPPORTED_CANDIDATES}."
        ) from exc
    return builder(**params)


def train_candidate(
    name: str,
    train_data: ModelData,
    **params: Any,
) -> TrainedCandidate:
    """Fit one candidate on the training partition only."""
    if len(train_data.features) != len(train_data.target):
        raise ValueError("Training features and target have different row counts.")
    if train_data.target.nunique() != 2:
        raise ValueError("Candidate training requires both target classes.")

    pipeline = build_candidate(name, **params)
    pipeline.fit(train_data.features, train_data.target)

    return TrainedCandidate(
        name=name,
        pipeline=pipeline,
        training_rows=len(train_data.target),
        positive_rows=int(train_data.target.sum()),
    )
