from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from mlforge.serving.app import create_app
from mlforge.serving.config import (
    RuntimeConfig,
    ServiceConfig,
    ServingConfig,
    ServingMLflowConfig,
    ServingModelConfig,
)
from mlforge.serving.inference import InferenceError
from mlforge.serving.model_loader import LoadedModel, ModelLoadError
from mlforge.serving.schemas import ChurnPredictionResponse


def _config() -> ServingConfig:
    return ServingConfig(
        service=ServiceConfig("api", "0.0.0.0", 8000),
        model=ServingModelConfig("MLForgeChurnClassifier", "champion", 0.5),
        mlflow=ServingMLflowConfig("sqlite:///tracking.db", "sqlite:///registry.db"),
        runtime=RuntimeConfig("INFO", 30),
    )


def _loaded_model() -> LoadedModel:
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


def _client(monkeypatch, service: Mock | None = None) -> TestClient:
    monkeypatch.setattr("mlforge.serving.app.load_serving_config", lambda _: _config())
    monkeypatch.setattr(
        "mlforge.serving.app.load_registry_model", lambda _: _loaded_model()
    )
    if service is not None:
        monkeypatch.setattr("mlforge.serving.app._build_service", lambda *_: service)
    return TestClient(create_app("test-serving.yaml"))


def test_health_and_readiness_expose_service_state(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        assert client.get("/health").json() == {"status": "ok"}
        readiness = client.get("/ready")

    assert readiness.status_code == 200
    assert readiness.json() == {
        "status": "ready",
        "model_name": "MLForgeChurnClassifier",
        "model_version": "7",
    }


def test_predict_returns_inference_response(monkeypatch) -> None:
    service = Mock()
    service.loaded_model = _loaded_model()
    service.predict.return_value = ChurnPredictionResponse(
        prediction="Yes",
        churn_probability=0.73,
        threshold=0.5,
        model_name="MLForgeChurnClassifier",
        model_version="7",
    )

    with _client(monkeypatch, service) as client:
        response = client.post("/predict", json=_payload())

    assert response.status_code == 200
    assert response.json()["churn_probability"] == 0.73
    service.predict.assert_called_once()


def test_predict_rejects_invalid_request_with_422(monkeypatch) -> None:
    payload = _payload()
    payload["MonthlyCharges"] = -1

    with _client(monkeypatch) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_hides_internal_inference_error(monkeypatch) -> None:
    service = Mock()
    service.loaded_model = _loaded_model()
    service.predict.side_effect = InferenceError("secret internal detail")

    with _client(monkeypatch, service) as client:
        response = client.post("/predict", json=_payload())

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction could not be completed."}
    assert "secret internal detail" not in response.text


def test_startup_fails_when_champion_cannot_load(monkeypatch) -> None:
    monkeypatch.setattr("mlforge.serving.app.load_serving_config", lambda _: _config())
    monkeypatch.setattr(
        "mlforge.serving.app.load_registry_model",
        Mock(side_effect=ModelLoadError("registry unavailable")),
    )
    client = TestClient(create_app("test-serving.yaml"))

    with pytest.raises(RuntimeError, match="startup failed"), client:
        pass


def test_openapi_documents_prediction_endpoint() -> None:
    schema = create_app("test-serving.yaml").openapi()

    assert "/predict" in schema["paths"]
    assert "ChurnPredictionRequest" in schema["components"]["schemas"]
    assert "ChurnPredictionResponse" in schema["components"]["schemas"]
