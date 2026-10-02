"""Dataset contract for the MLForge reference churn dataset."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

CHURN_SCHEMA = pa.DataFrameSchema(
    {
        "customerID": pa.Column(str, nullable=False, unique=True),
        "gender": pa.Column(str, pa.Check.isin(["Female", "Male"])),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1])),
        "Partner": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "Dependents": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "tenure": pa.Column(int, pa.Check.ge(0)),
        "PhoneService": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "MultipleLines": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No phone service"])
        ),
        "InternetService": pa.Column(
            str, pa.Check.isin(["DSL", "Fiber optic", "No"])
        ),
        "OnlineSecurity": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"])
        ),
        "OnlineBackup": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"])
        ),
        "DeviceProtection": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"])
        ),
        "TechSupport": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingTV": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingMovies": pa.Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"])
        ),
        "Contract": pa.Column(
            str, pa.Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "PaymentMethod": pa.Column(
            str,
            pa.Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": pa.Column(float, pa.Check.ge(0), coerce=True),
        "TotalCharges": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "Churn": pa.Column(str, pa.Check.isin(["Yes", "No"])),
    },
    strict=True,
    ordered=True,
    coerce=False,
)


def validate_churn_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate a prepared churn dataframe against the Day 1 contract."""
    if data.duplicated().any():
        raise ValueError("Dataset contains duplicate rows.")
    return CHURN_SCHEMA.validate(data, lazy=True)
