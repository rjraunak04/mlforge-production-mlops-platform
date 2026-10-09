import pytest
from pydantic import ValidationError

from mlforge.features.preprocessing import FeatureColumns
from mlforge.serving.schemas import ChurnPredictionRequest, ChurnPredictionResponse


def _payload() -> dict[str, object]:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0,
    }


def test_prediction_request_matches_training_feature_contract() -> None:
    request = ChurnPredictionRequest(**_payload())

    frame = request.to_frame()

    assert tuple(frame.columns) == FeatureColumns().all
    assert frame.shape == (1, 19)


def test_prediction_request_allows_missing_total_charges() -> None:
    payload = _payload()
    payload["TotalCharges"] = None

    request = ChurnPredictionRequest(**payload)

    assert request.TotalCharges is None


def test_prediction_request_forbids_unknown_fields() -> None:
    payload = _payload()
    payload["customerID"] = "customer-001"

    with pytest.raises(ValidationError, match="customerID"):
        ChurnPredictionRequest(**payload)


def test_prediction_request_rejects_invalid_category() -> None:
    payload = _payload()
    payload["Contract"] = "Three year"

    with pytest.raises(ValidationError, match="Contract"):
        ChurnPredictionRequest(**payload)


def test_prediction_request_rejects_negative_numeric_values() -> None:
    payload = _payload()
    payload["MonthlyCharges"] = -1.0

    with pytest.raises(ValidationError, match="MonthlyCharges"):
        ChurnPredictionRequest(**payload)


def test_prediction_request_rejects_phone_service_inconsistency() -> None:
    payload = _payload()
    payload["PhoneService"] = "No"

    with pytest.raises(ValidationError, match="MultipleLines"):
        ChurnPredictionRequest(**payload)


def test_prediction_request_rejects_internet_service_inconsistency() -> None:
    payload = _payload()
    payload["InternetService"] = "No"

    with pytest.raises(ValidationError, match="Internet add-ons"):
        ChurnPredictionRequest(**payload)


def test_prediction_response_enforces_probability_bounds() -> None:
    with pytest.raises(ValidationError, match="churn_probability"):
        ChurnPredictionResponse(
            prediction="Yes",
            churn_probability=1.1,
            threshold=0.5,
            model_name="MLForgeChurnClassifier",
            model_version="1",
        )
