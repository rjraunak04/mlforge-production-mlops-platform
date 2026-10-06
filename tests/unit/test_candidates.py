"""Tests for candidate model training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlforge.features.prepare import prepare_model_data
from mlforge.training.candidates import (
    SUPPORTED_CANDIDATES,
    build_candidate,
    train_candidate,
)


def _training_partition(rows: int = 48) -> pd.DataFrame:
    records = []
    for index in range(rows):
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
                "Churn": "Yes" if index % 4 == 0 else "No",
            }
        )
    return pd.DataFrame.from_records(records)


@pytest.mark.parametrize("name", SUPPORTED_CANDIDATES)
def test_candidate_builders_return_unfitted_pipelines(name: str) -> None:
    pipeline = build_candidate(name)

    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]
    assert not hasattr(pipeline.named_steps["classifier"], "classes_")


@pytest.mark.parametrize("name", SUPPORTED_CANDIDATES)
def test_candidates_train_and_produce_probabilities(name: str) -> None:
    prepared = prepare_model_data(_training_partition())

    trained = train_candidate(name, prepared)
    probabilities = trained.pipeline.predict_proba(prepared.features)[:, 1]

    assert trained.name == name
    assert trained.training_rows == 48
    assert trained.positive_rows == 12
    assert probabilities.shape == (48,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_candidate_training_is_reproducible() -> None:
    prepared = prepare_model_data(_training_partition())

    first = train_candidate(
        "random_forest",
        prepared,
        n_estimators=25,
        n_jobs=1,
        random_state=42,
    )
    second = train_candidate(
        "random_forest",
        prepared,
        n_estimators=25,
        n_jobs=1,
        random_state=42,
    )

    first_probabilities = first.pipeline.predict_proba(prepared.features)[:, 1]
    second_probabilities = second.pipeline.predict_proba(prepared.features)[:, 1]
    np.testing.assert_allclose(first_probabilities, second_probabilities)


def test_unknown_candidate_is_rejected() -> None:
    prepared = prepare_model_data(_training_partition())

    with pytest.raises(ValueError, match="Unsupported candidate"):
        train_candidate("unknown_model", prepared)
