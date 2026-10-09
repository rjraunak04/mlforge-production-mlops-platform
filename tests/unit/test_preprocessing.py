"""Tests for leakage-safe preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from mlforge.features.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    select_model_features,
)


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C", "D"],
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 1],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "Yes"],
            "tenure": [1, 12, 24, 36],
            "PhoneService": ["Yes", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No", "Yes", "No", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No", "Yes", "No", "No internet service"],
            "OnlineBackup": ["Yes", "No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No", "No internet service"],
            "TechSupport": ["No", "Yes", "No", "No internet service"],
            "StreamingTV": ["No", "Yes", "No", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 42.30, 18.95],
            "TotalCharges": [29.85, 684.4, None, 682.2],
            "Churn": ["No", "No", "Yes", "No"],
        }
    )


def test_feature_selection_excludes_id_and_target() -> None:
    selected = select_model_features(_feature_frame())

    assert list(selected.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert "customerID" not in selected.columns
    assert "Churn" not in selected.columns


def test_preprocessor_is_unfitted_when_built() -> None:
    features = select_model_features(_feature_frame())
    preprocessor = build_preprocessor()

    with pytest.raises(NotFittedError):
        preprocessor.transform(features)


def test_preprocessor_fits_training_data_and_handles_missing_numeric() -> None:
    features = select_model_features(_feature_frame())
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == len(features)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)


def test_unknown_category_is_safe_at_transform_time() -> None:
    train = select_model_features(_feature_frame().iloc[:3])
    validation = select_model_features(_feature_frame().iloc[[3]].copy())
    validation.loc[:, "InternetService"] = "Satellite"

    preprocessor = build_preprocessor()
    preprocessor.fit(train)
    transformed = preprocessor.transform(validation)

    assert transformed.shape[0] == 1
