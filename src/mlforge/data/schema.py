"""Dataset contract for the MLForge reference churn dataset."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column, DataFrameSchema

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

CHURN_SCHEMA = DataFrameSchema(
    {
        "customerID": Column(str, nullable=False, unique=True),
        "gender": Column(str, Check.isin(["Female", "Male"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(["Yes", "No"])),
        "Dependents": Column(str, Check.isin(["Yes", "No"])),
        "tenure": Column(int, Check.ge(0)),
        "PhoneService": Column(str, Check.isin(["Yes", "No"])),
        "MultipleLines": Column(
            str, Check.isin(["Yes", "No", "No phone service"])
        ),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "OnlineBackup": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "DeviceProtection": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "TechSupport": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingTV": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingMovies": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(
            str,
            Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": Column(float, Check.ge(0), coerce=True),
        "TotalCharges": Column(float, Check.ge(0), nullable=True, coerce=True),
        "Churn": Column(str, Check.isin(["Yes", "No"])),
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
