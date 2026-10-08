"""FastAPI application exposing the production churn inference service."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from mlforge.serving.config import ServingConfig, load_serving_config
from mlforge.serving.inference import ChurnInferenceService, InferenceError
from mlforge.serving.observability import (
    RequestObservabilityMiddleware,
    configure_logging,
)
from mlforge.serving.model_loader import (
    LoadedModel,
    ModelLoadError,
    load_registry_model,
)
from mlforge.serving.schemas import ChurnPredictionRequest, ChurnPredictionResponse


class HealthResponse(BaseModel):
    """Liveness contract independent of model readiness."""

    model_config = ConfigDict(extra="forbid")
    status: str


class ReadinessResponse(BaseModel):
    """Readiness contract for the pinned production model."""

    model_config = ConfigDict(extra="forbid")
    status: str
    model_name: str
    model_version: str


def _config_path() -> str:
    return os.getenv("MLFORGE_SERVING_CONFIG", "configs/serving.yaml")


def _build_service(config: ServingConfig, loaded: LoadedModel) -> ChurnInferenceService:
    return ChurnInferenceService(
        loaded_model=loaded,
        threshold=config.model.classification_threshold,
    )


def create_app(config_path: str | None = None) -> FastAPI:
    """Create an application whose startup requires a loadable champion model."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            config = load_serving_config(config_path or _config_path())
            loaded = load_registry_model(config)
        except (FileNotFoundError, ValueError, ModelLoadError) as exc:
            raise RuntimeError("Inference service startup failed.") from exc

        configure_logging(config.runtime.log_level)
        app.state.inference_service = _build_service(config, loaded)
        yield
        app.state.inference_service = None

    application = FastAPI(
        title="MLForge Churn Inference API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(RequestObservabilityMiddleware)

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/ready", response_model=ReadinessResponse)
    async def ready(request: Request) -> ReadinessResponse:
        service = getattr(request.app.state, "inference_service", None)
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference service is not ready.",
            )
        return ReadinessResponse(
            status="ready",
            model_name=service.loaded_model.model_name,
            model_version=service.loaded_model.version,
        )

    @application.post("/predict", response_model=ChurnPredictionResponse)
    async def predict(
        payload: ChurnPredictionRequest, request: Request
    ) -> ChurnPredictionResponse:
        service = getattr(request.app.state, "inference_service", None)
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference service is not ready.",
            )
        try:
            return service.predict(payload)
        except InferenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction could not be completed.",
            ) from exc

    return application


app = create_app()
