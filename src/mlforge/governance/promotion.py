"""Champion/challenger promotion decisions for registered models."""

from __future__ import annotations

from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mlforge.registry.model_registry import RegisteredCandidate
from mlforge.tracking.config import MLflowConfig


@dataclass(frozen=True)
class PromotionDecision:
    """Machine-readable champion promotion outcome."""

    model_name: str
    candidate_version: str
    champion_version: str | None
    comparison_metric: str
    candidate_score: float
    champion_score: float | None
    promoted: bool
    reason: str


def decide_promotion(
    *,
    model_name: str,
    candidate_version: str,
    candidate_score: float,
    champion_version: str | None,
    champion_score: float | None,
    comparison_metric: str = "roc_auc",
) -> PromotionDecision:
    """Promote first candidate or require strict improvement over champion."""
    if not 0.0 <= candidate_score <= 1.0:
        raise ValueError("candidate_score must be between 0 and 1.")
    if champion_version is not None and champion_score is None:
        raise ValueError("champion_score is required when a champion exists.")
    if champion_score is not None and not 0.0 <= champion_score <= 1.0:
        raise ValueError("champion_score must be between 0 and 1.")

    if champion_version is None:
        promoted = True
        reason = "No existing champion; approved candidate becomes initial champion."
    else:
        promoted = candidate_score > champion_score
        reason = (
            f"Candidate improved {comparison_metric} over champion."
            if promoted
            else f"Candidate did not improve {comparison_metric} over champion."
        )

    return PromotionDecision(
        model_name=model_name,
        candidate_version=candidate_version,
        champion_version=champion_version,
        comparison_metric=comparison_metric,
        candidate_score=candidate_score,
        champion_score=champion_score,
        promoted=promoted,
        reason=reason,
    )


def apply_promotion_decision(
    *,
    config: MLflowConfig,
    candidate: RegisteredCandidate,
    decision: PromotionDecision,
) -> PromotionDecision:
    """Apply challenger/champion aliases and governance tags in MLflow."""
    if candidate.model_name != decision.model_name:
        raise ValueError("Promotion decision model does not match candidate.")
    if candidate.version != decision.candidate_version:
        raise ValueError("Promotion decision version does not match candidate.")

    client = MlflowClient(
        tracking_uri=config.tracking.uri,
        registry_uri=config.tracking.registry_uri,
    )
    client.set_registered_model_alias(
        candidate.model_name,
        "challenger",
        candidate.version,
    )
    client.set_model_version_tag(
        candidate.model_name,
        candidate.version,
        "governance.promotion_decision",
        "promote" if decision.promoted else "reject",
    )
    client.set_model_version_tag(
        candidate.model_name,
        candidate.version,
        "governance.promotion_reason",
        decision.reason,
    )
    client.set_model_version_tag(
        candidate.model_name,
        candidate.version,
        f"validation.{decision.comparison_metric}",
        str(decision.candidate_score),
    )

    if decision.promoted:
        client.set_registered_model_alias(
            candidate.model_name,
            "champion",
            candidate.version,
        )
        client.set_model_version_tag(
            candidate.model_name,
            candidate.version,
            "governance.status",
            "champion",
        )
        try:
            client.delete_registered_model_alias(candidate.model_name, "challenger")
        except MlflowException:
            pass
    else:
        client.set_model_version_tag(
            candidate.model_name,
            candidate.version,
            "governance.status",
            "challenger",
        )

    return decision
