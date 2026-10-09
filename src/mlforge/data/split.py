"""Leakage-safe dataset splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_churn_data(
    data: pd.DataFrame,
    *,
    target_column: str = "Churn",
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplits:
    """Create deterministic, stratified train/validation/test partitions."""
    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Split proportions must sum to 1.0.")

    train, holdout = train_test_split(
        data,
        train_size=train_size,
        random_state=random_state,
        stratify=data[target_column],
    )

    validation_fraction = validation_size / (validation_size + test_size)
    validation, test = train_test_split(
        holdout,
        train_size=validation_fraction,
        random_state=random_state,
        stratify=holdout[target_column],
    )

    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
