import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlforge.serving.observability import REQUEST_ID_HEADER, RequestObservabilityMiddleware


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestObservabilityMiddleware)

    @app.post("/predict")
    async def predict() -> dict[str, str]:
        return {"status": "ok"}

    return app


def test_middleware_preserves_valid_request_id(caplog) -> None:
    caplog.set_level(logging.INFO, logger="mlforge.serving")
    request_id = "client-request-123"

    response = TestClient(_app()).post(
        "/predict",
        headers={REQUEST_ID_HEADER: request_id},
        json={"MonthlyCharges": 999.99, "secret": "must-not-be-logged"},
    )

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == request_id
    event = json.loads(caplog.records[-1].message)
    assert event["event"] == "http_request"
    assert event["request_id"] == request_id
    assert event["method"] == "POST"
    assert event["path"] == "/predict"
    assert event["status_code"] == 200
    assert event["latency_ms"] >= 0
    assert "999.99" not in caplog.text
    assert "must-not-be-logged" not in caplog.text


def test_middleware_generates_request_id() -> None:
    response = TestClient(_app()).post("/predict", json={})

    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id
    assert len(request_id) <= 128


def test_middleware_replaces_oversized_request_id() -> None:
    supplied = "x" * 129

    response = TestClient(_app()).post(
        "/predict", headers={REQUEST_ID_HEADER: supplied}, json={}
    )

    assert response.headers[REQUEST_ID_HEADER] != supplied
    assert len(response.headers[REQUEST_ID_HEADER]) <= 128
