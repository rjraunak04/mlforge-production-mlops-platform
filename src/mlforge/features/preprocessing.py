"""Leakage-safe preprocessing for the churn training pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


@dataclass(frozen=True)
class FeatureColumns:
    """Explicit model feature groups."""

    numeric: tuple[str, ...] = tuple(NUMERIC_FEATURES)
    categorical: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)

    @property
    def all(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


def build_preprocessor(
    *,
    numeric_imputation: str = "median",
    categorical_imputation: str = "most_frequent",
    handle_unknown: str = "ignore",
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build an unfitted preprocessing graph.

    The returned transformer must be fitted on training features only.
    Validation and test data may only be transformed with that fitted object.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=numeric_imputation)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_imputation)),
            (
                "one_hot",
                OneHotEncoder(handle_unknown=handle_unknown, sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def select_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Select model inputs in a stable order, excluding ID and target columns."""
    columns = FeatureColumns()
    missing = [column for column in columns.all if column not in data.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}")
    return data.loc[:, list(columns.all)].copy()
