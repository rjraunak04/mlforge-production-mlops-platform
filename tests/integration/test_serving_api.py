from unittest.mock import Mock

from fastapi.testclient import TestClient

from mlforge.serving.app import create_app
from mlforge.serving.config import (
    RuntimeConfig,
    ServiceConfig,
    ServingConfig,
    ServingMLflowConfig,
    ServingModelConfig,
)
from mlforge.serving.model_loader import LoadedModel
from mlforge.serving.schemas import ChurnPredictionResponse


def _config() -> ServingConfig:
    return ServingConfig(
        service=ServiceConfig("api", "0.0.0.0", 8000),
        model=ServingModelConfig("MLForgeChurnClassifier", "champion", 0.5),
        mlflow=ServingMLflowConfig("sqlite:///tracking.db", "sqlite:///registry.db"),
        runtime=RuntimeConfig("INFO", 30),
    )


def _loaded() -> LoadedModel:
    return LoadedModel(
        pipeline=Mock(),
        model_name="MLForgeChurnClassifier",
        alias="champion",
        version="7",
        run_id="run-123",
        model_uri="models:/MLForgeChurnClassifier/7",
    )


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


def test_serving_stack_health_ready_predict_and_request_id(monkeypatch) -> None:
    service = Mock()
    service.loaded_model = _loaded()
    service.predict.return_value = ChurnPredictionResponse(
        prediction="Yes",
        churn_probability=0.73,
        threshold=0.5,
        model_name="MLForgeChurnClassifier",
        model_version="7",
    )
    monkeypatch.setattr("mlforge.serving.app.load_serving_config", lambda _: _config())
    monkeypatch.setattr("mlforge.serving.app.load_registry_model", lambda _: _loaded())
    monkeypatch.setattr("mlforge.serving.app._build_service", lambda *_: service)

    with TestClient(create_app("test-serving.yaml")) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        prediction = client.post(
            "/predict", json=_payload(), headers={"X-Request-ID": "e2e-123"}
        )

    assert health.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["model_version"] == "7"
    assert prediction.status_code == 200
    assert prediction.json()["churn_probability"] == 0.73
    assert prediction.headers["X-Request-ID"] == "e2e-123"
