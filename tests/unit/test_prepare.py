"""Tests for model-ready feature and target preparation."""

from __future__ import annotations

import pandas as pd
import pytest

from mlforge.features.prepare import encode_target, prepare_model_data
from mlforge.features.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _partition() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 12],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": [29.85, 684.4],
            "Churn": ["No", "Yes"],
        }
    )


def test_encode_target_maps_no_yes_to_zero_one() -> None:
    encoded = encode_target(pd.Series(["No", "Yes", "No"], name="Churn"))

    assert encoded.tolist() == [0, 1, 0]
    assert str(encoded.dtype) == "int8"


def test_encode_target_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="Unexpected target labels"):
        encode_target(pd.Series(["No", "Maybe"], name="Churn"))


def test_encode_target_rejects_missing_value() -> None:
    with pytest.raises(ValueError, match="Target contains missing values"):
        encode_target(pd.Series(["No", None], name="Churn"))


def test_prepare_model_data_separates_features_target_and_id() -> None:
    prepared = prepare_model_data(_partition())

    assert list(prepared.features.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert prepared.target.tolist() == [0, 1]
    assert prepared.customer_ids.tolist() == ["A", "B"]
    assert "customerID" not in prepared.features.columns
    assert "Churn" not in prepared.features.columns


def test_prepare_model_data_rejects_duplicate_ids() -> None:
    data = _partition()
    data.loc[1, "customerID"] = "A"

    with pytest.raises(ValueError, match="must be unique"):
        prepare_model_data(data)
