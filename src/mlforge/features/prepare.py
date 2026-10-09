"""Model-ready feature and target preparation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlforge.features.preprocessing import select_model_features

TARGET_MAPPING = {"No": 0, "Yes": 1}


@dataclass(frozen=True)
class ModelData:
    """Features, binary target, and traceability identifiers."""

    features: pd.DataFrame
    target: pd.Series
    customer_ids: pd.Series


def encode_target(
    target: pd.Series,
    *,
    negative_label: str = "No",
    positive_label: str = "Yes",
) -> pd.Series:
    """Encode the binary churn target as 0/1 with strict label validation."""
    mapping = {negative_label: 0, positive_label: 1}
    observed = set(target.dropna().unique())
    unexpected = observed - set(mapping)

    if target.isna().any():
        raise ValueError("Target contains missing values.")
    if unexpected:
        raise ValueError(f"Unexpected target labels: {sorted(unexpected)}")

    encoded = target.map(mapping)
    if encoded.isna().any():
        raise ValueError("Target encoding produced missing values.")

    return encoded.astype("int8").rename(target.name)


def prepare_model_data(
    data: pd.DataFrame,
    *,
    target_column: str = "Churn",
    id_column: str = "customerID",
    negative_label: str = "No",
    positive_label: str = "Yes",
) -> ModelData:
    """Prepare one already-split partition for modeling.

    This function performs deterministic column/target preparation only. It
    does not fit preprocessing statistics, preserving the train-only fit rule.
    """
    required = [target_column, id_column]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required model-data columns: {missing}")

    customer_ids = data[id_column].copy()
    if customer_ids.isna().any():
        raise ValueError("Customer identifiers contain missing values.")
    if customer_ids.duplicated().any():
        raise ValueError("Customer identifiers must be unique within a partition.")

    features = select_model_features(data)
    target = encode_target(
        data[target_column],
        negative_label=negative_label,
        positive_label=positive_label,
    )

    return ModelData(
        features=features.reset_index(drop=True),
        target=target.reset_index(drop=True),
        customer_ids=customer_ids.reset_index(drop=True),
    )
