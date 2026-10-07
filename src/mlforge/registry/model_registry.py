"""Governed MLflow Model Registry operations."""

from __future__ import annotations

from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mlforge.governance.quality_gates import QualityGateResult
from mlforge.tracking.config import MLflowConfig
from mlforge.tracking.experiments import TrackedRun


@dataclass(frozen=True)
class RegisteredCandidate:
    """Identity of one quality-approved registered model version."""

    model_name: str
    version: str
    run_id: str
    source: str


def register_approved_candidate(
    *,
    config: MLflowConfig,
    tracked_run: TrackedRun,
    gate_result: QualityGateResult,
) -> RegisteredCandidate:
    """Register a model version only after its validation quality gates pass."""
    if gate_result.model_name != tracked_run.model_name:
        raise ValueError("Quality-gate model does not match the tracked run.")
    if not gate_result.passed:
        raise ValueError("Quality-gate failure blocks model registration.")
    if not tracked_run.model_uri:
        raise ValueError("A logged model artifact is required for registration.")

    client = MlflowClient(
        tracking_uri=config.tracking.uri,
        registry_uri=config.tracking.registry_uri,
    )
    model_name = config.registry.model_name

    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(
            model_name,
            tags={"project": config.tags.get("project", "mlforge")},
            description="Quality-governed MLForge churn classifier.",
        )

    version = client.create_model_version(
        name=model_name,
        source=tracked_run.model_uri,
        run_id=tracked_run.run_id,
        tags={
            "governance.status": "approved",
            "validation.model_name": tracked_run.model_name,
        },
        description="Candidate passed configured validation quality gates.",
    )

    return RegisteredCandidate(
        model_name=model_name,
        version=str(version.version),
        run_id=tracked_run.run_id,
        source=tracked_run.model_uri,
    )
