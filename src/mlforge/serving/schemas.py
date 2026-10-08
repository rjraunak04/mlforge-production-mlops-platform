"""Pydantic contracts for production churn inference."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlforge.features.preprocessing import FeatureColumns


class ChurnPredictionRequest(BaseModel):
    """Validated model features for one churn prediction."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_service_dependencies(self) -> ChurnPredictionRequest:
        """Reject internally inconsistent telecom service combinations."""
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError(
                "MultipleLines must be 'No phone service' when PhoneService is 'No'."
            )
        if self.PhoneService == "Yes" and self.MultipleLines == "No phone service":
            raise ValueError(
                "MultipleLines cannot be 'No phone service' when PhoneService is 'Yes'."
            )

        internet_features = (
            self.OnlineSecurity,
            self.OnlineBackup,
            self.DeviceProtection,
            self.TechSupport,
            self.StreamingTV,
            self.StreamingMovies,
        )
        if self.InternetService == "No" and any(
            value != "No internet service" for value in internet_features
        ):
            raise ValueError(
                "Internet add-ons must be 'No internet service' when "
                "InternetService is 'No'."
            )
        if self.InternetService != "No" and any(
            value == "No internet service" for value in internet_features
        ):
            raise ValueError(
                "Internet add-ons cannot be 'No internet service' when "
                "internet is active."
            )
        return self

    def to_frame(self) -> pd.DataFrame:
        """Convert one validated request to the training feature order."""
        payload = self.model_dump()
        columns = FeatureColumns().all
        return pd.DataFrame([[payload[column] for column in columns]], columns=columns)


class ChurnPredictionResponse(BaseModel):
    """Stable public response contract for one model prediction."""

    model_config = ConfigDict(extra="forbid")

    prediction: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
