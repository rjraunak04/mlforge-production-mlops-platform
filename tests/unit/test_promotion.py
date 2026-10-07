"""Tests for champion/challenger model promotion governance."""

from pathlib import Path

import mlflow
from mlflow import MlflowClient

from mlforge.governance.promotion import apply_promotion_decision, decide_promotion
from mlforge.registry.model_registry import RegisteredCandidate
from mlforge.tracking.config import (
    MLflowConfig,
    MLflowTrackingConfig,
    ModelRegistryConfig,
)


def _config(tmp_path: Path) -> MLflowConfig:
    database = (tmp_path / "mlflow.db").as_posix()
    uri = f"sqlite:///{database}"
    return MLflowConfig(
        tracking=MLflowTrackingConfig(
            uri=uri,
            artifact_root=(tmp_path / "artifacts").resolve().as_uri(),
            experiment_name="promotion-test",
            registry_uri=uri,
        ),
        registry=ModelRegistryConfig(model_name="MLForgeTestClassifier"),
        tags={"project": "mlforge"},
    )


def _candidate(config: MLflowConfig, version: str = "1") -> RegisteredCandidate:
    mlflow.set_tracking_uri(config.tracking.uri)
    mlflow.set_registry_uri(config.tracking.registry_uri)
    client = MlflowClient(
        tracking_uri=config.tracking.uri,
        registry_uri=config.tracking.registry_uri,
    )
    try:
        client.create_registered_model(config.registry.model_name)
    except Exception:
        pass
    model_version = client.create_model_version(
        name=config.registry.model_name,
        source=f"runs:/run-{version}/model",
        run_id=f"run-{version}",
    )
    return RegisteredCandidate(
        model_name=config.registry.model_name,
        version=str(model_version.version),
        run_id=f"run-{version}",
        source=f"runs:/run-{version}/model",
    )


def test_first_approved_candidate_is_promoted() -> None:
    decision = decide_promotion(
        model_name="MLForgeChurnClassifier",
        candidate_version="1",
        candidate_score=0.8453,
        champion_version=None,
        champion_score=None,
    )

    assert decision.promoted is True
    assert "initial champion" in decision.reason


def test_candidate_must_strictly_beat_existing_champion() -> None:
    improved = decide_promotion(
        model_name="MLForgeChurnClassifier",
        candidate_version="2",
        candidate_score=0.86,
        champion_version="1",
        champion_score=0.8453,
    )
    tied = decide_promotion(
        model_name="MLForgeChurnClassifier",
        candidate_version="2",
        candidate_score=0.8453,
        champion_version="1",
        champion_score=0.8453,
    )

    assert improved.promoted is True
    assert tied.promoted is False


def test_promoted_candidate_receives_champion_alias(tmp_path: Path) -> None:
    config = _config(tmp_path)
    candidate = _candidate(config)
    decision = decide_promotion(
        model_name=candidate.model_name,
        candidate_version=candidate.version,
        candidate_score=0.8453,
        champion_version=None,
        champion_score=None,
    )

    apply_promotion_decision(config=config, candidate=candidate, decision=decision)

    client = MlflowClient(
        tracking_uri=config.tracking.uri,
        registry_uri=config.tracking.registry_uri,
    )
    champion = client.get_model_version_by_alias(candidate.model_name, "champion")
    version = client.get_model_version(candidate.model_name, candidate.version)
    assert str(champion.version) == candidate.version
    assert version.tags["governance.status"] == "champion"
    assert version.tags["governance.promotion_decision"] == "promote"


def test_rejected_candidate_remains_challenger(tmp_path: Path) -> None:
    config = _config(tmp_path)
    champion = _candidate(config)
    client = MlflowClient(
        tracking_uri=config.tracking.uri,
        registry_uri=config.tracking.registry_uri,
    )
    client.set_registered_model_alias(champion.model_name, "champion", champion.version)

    candidate = _candidate(config, version="2")
    decision = decide_promotion(
        model_name=candidate.model_name,
        candidate_version=candidate.version,
        candidate_score=0.82,
        champion_version=champion.version,
        champion_score=0.8453,
    )
    apply_promotion_decision(config=config, candidate=candidate, decision=decision)

    current = client.get_model_version_by_alias(candidate.model_name, "champion")
    challenger = client.get_model_version_by_alias(candidate.model_name, "challenger")
    version = client.get_model_version(candidate.model_name, candidate.version)
    assert str(current.version) == champion.version
    assert str(challenger.version) == candidate.version
    assert version.tags["governance.status"] == "challenger"
    assert version.tags["governance.promotion_decision"] == "reject"
