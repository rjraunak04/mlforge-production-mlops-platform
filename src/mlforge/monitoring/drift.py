"""Deterministic feature and prediction drift diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    score: float
    drifted: bool


@dataclass(frozen=True)
class DriftSummary:
    features: tuple[FeatureDrift, ...]
    drifted_share: float
    dataset_drifted: bool


def _numeric_score(reference: pd.Series, current: pd.Series) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return 0.0
    scale = max(float(ref.std(ddof=0)), 1e-9)
    return abs(float(cur.mean()) - float(ref.mean())) / scale


def _categorical_score(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.fillna("__missing__").astype(str).value_counts(normalize=True)
    cur = current.fillna("__missing__").astype(str).value_counts(normalize=True)
    categories = ref.index.union(cur.index)
    return 0.5 * sum(
        abs(float(ref.get(x, 0.0)) - float(cur.get(x, 0.0))) for x in categories
    )


def detect_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    dataset_threshold: float = 0.30,
    numeric_threshold: float = 0.50,
    categorical_threshold: float = 0.20,
) -> DriftSummary:
    if tuple(reference.columns) != tuple(current.columns):
        raise ValueError("Reference and current feature contracts must match.")
    results = []
    for column in reference.columns:
        numeric = pd.api.types.is_numeric_dtype(reference[column])
        score = (
            _numeric_score(reference[column], current[column])
            if numeric
            else _categorical_score(reference[column], current[column])
        )
        threshold = numeric_threshold if numeric else categorical_threshold
        results.append(FeatureDrift(column, score, score >= threshold))
    share = float(np.mean([item.drifted for item in results])) if results else 0.0
    return DriftSummary(tuple(results), share, share >= dataset_threshold)


def prediction_drift(
    reference_probability: pd.Series,
    current_probability: pd.Series,
) -> float:
    if reference_probability.empty or current_probability.empty:
        raise ValueError("Prediction distributions must not be empty.")
    return abs(float(current_probability.mean()) - float(reference_probability.mean()))
