"""Tests for reproducible Logistic Regression baseline training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mlforge.features.prepare import prepare_model_data
from mlforge.training.baseline import (
    build_logistic_baseline,
    train_logistic_baseline,
)


def _training_partition(rows: int = 40) -> pd.DataFrame:
    records = []
    for index in range(rows):
        churn = "Yes" if index % 4 == 0 else "No"
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
                "Churn": churn,
            }
        )
    return pd.DataFrame.from_records(records)


def test_build_logistic_baseline_returns_unfitted_pipeline() -> None:
    pipeline = build_logistic_baseline()

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]
    assert not hasattr(pipeline.named_steps["classifier"], "classes_")


def test_train_logistic_baseline_fits_complete_pipeline() -> None:
    prepared = prepare_model_data(_training_partition())

    trained = train_logistic_baseline(prepared)

    assert trained.training_rows == 40
    assert trained.positive_rows == 10
    assert trained.pipeline.named_steps["classifier"].classes_.tolist() == [0, 1]


def test_baseline_produces_probabilities() -> None:
    prepared = prepare_model_data(_training_partition())
    trained = train_logistic_baseline(prepared)

    probabilities = trained.pipeline.predict_proba(prepared.features)[:, 1]

    assert probabilities.shape == (40,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_baseline_is_reproducible_for_same_training_data() -> None:
    prepared = prepare_model_data(_training_partition())

    first = train_logistic_baseline(prepared, random_state=42)
    second = train_logistic_baseline(prepared, random_state=42)

    first_probabilities = first.pipeline.predict_proba(prepared.features)[:, 1]
    second_probabilities = second.pipeline.predict_proba(prepared.features)[:, 1]

    np.testing.assert_allclose(first_probabilities, second_probabilities)
