import pandas as pd
import pandera.errors as pa_errors
import pytest

from mlforge.data.schema import EXPECTED_COLUMNS, validate_churn_data


def valid_frame() -> pd.DataFrame:
    row = {
        "customerID": "0001-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
        "Churn": "No",
    }
    return pd.DataFrame([row], columns=EXPECTED_COLUMNS)


def test_valid_data_passes_contract() -> None:
    result = validate_churn_data(valid_frame())
    assert len(result) == 1


def test_missing_target_fails_contract() -> None:
    data = valid_frame().drop(columns=["Churn"])
    with pytest.raises(pa_errors.SchemaErrors):
        validate_churn_data(data)


def test_invalid_target_fails_contract() -> None:
    data = valid_frame()
    data.loc[0, "Churn"] = "Maybe"
    with pytest.raises(pa_errors.SchemaErrors):
        validate_churn_data(data)


def test_duplicate_customer_id_fails_contract() -> None:
    data = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
    with pytest.raises((ValueError, pa_errors.SchemaErrors)):
        validate_churn_data(data)
