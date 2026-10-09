"""End-to-end Day 3 experiment-governance orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mlflow import MlflowClient

from mlforge.governance.promotion import (
    PromotionDecision,
    apply_promotion_decision,
    decide_promotion,
)
from mlforge.governance.quality_gates import (
    QualityGateResult,
    QualityGateThresholds,
    evaluate_quality_gates,
)
from mlforge.registry.model_registry import (
    RegisteredCandidate,
    register_approved_candidate,
)
from mlforge.tracking.config import MLflowConfig, load_mlflow_config
from mlforge.tracking.experiments import (
    TrackedRun,
    configure_mlflow,
    track_validation_run,
)
from mlforge.training.runner import (
    TrainingRun,
    load_training_config,
    run_training_pipeline,
)


@dataclass(frozen=True)
class GovernanceRun:
    """Evidence from one complete validation-governed training lifecycle."""

    training: TrainingRun
    tracked_runs: tuple[TrackedRun, ...]
    quality_gate: QualityGateResult
    registered_candidate: RegisteredCandidate | None
    promotion: PromotionDecision | None


def _champion_evidence(
    config: MLflowConfig,
    *,
    metric: str,
) -> tuple[str | None, float | None]:
    client = MlflowClient(
        tracking_uri=config.tracking.uri,
        registry_uri=config.tracking.registry_uri,
    )
    try:
        champion = client.get_model_version_by_alias(
            config.registry.model_name, "champion"
        )
    except Exception:
        return None, None

    score = champion.tags.get(f"validation.{metric}")
    if score is None:
        raise ValueError(
            "Existing champion is missing its validation comparison score."
        )
    return str(champion.version), float(score)


def run_governed_training(
    data_path: str | Path,
    *,
    training_config_path: str | Path = "configs/training.yaml",
    mlflow_config_path: str | Path = "configs/mlflow.yaml",
) -> GovernanceRun:
    """Train, track, gate, register, and promote using validation evidence only."""
    training_config = load_training_config(training_config_path)
    mlflow_config = load_mlflow_config(mlflow_config_path)
    training = run_training_pipeline(
        data_path,
        training_config_path=training_config_path,
    )

    experiment_id = configure_mlflow(mlflow_config)
    settings = training_config["training"]
    evaluation = training_config["evaluation"]
    tracked: list[TrackedRun] = []

    results = {result.model_name: result for result in training.validation_results}
    for model_name, pipeline in training.models.items():
        tracked.append(
            track_validation_run(
                config=mlflow_config,
                experiment_id=experiment_id,
                result=results[model_name],
                model_config=training_config["models"][model_name],
                random_state=int(settings["random_state"]),
                classification_threshold=float(evaluation["classification_threshold"]),
                training_rows=len(training.train.target),
                validation_rows=len(training.validation.target),
                pipeline=pipeline,
                selection_metric=training.comparison.selection_metric,
            )
        )

    winner_name = training.comparison.selected_model
    winner_result = results[winner_name]
    thresholds = QualityGateThresholds(
        min_roc_auc=mlflow_config.quality_gates.min_roc_auc,
        min_average_precision=mlflow_config.quality_gates.min_average_precision,
        min_recall=mlflow_config.quality_gates.min_recall,
    )
    gate = evaluate_quality_gates(winner_result, thresholds)
    if not gate.passed:
        return GovernanceRun(training, tuple(tracked), gate, None, None)

    winner_run = next(run for run in tracked if run.model_name == winner_name)
    candidate = register_approved_candidate(
        config=mlflow_config,
        tracked_run=winner_run,
        gate_result=gate,
    )

    metric = training.comparison.selection_metric
    champion_version, champion_score = _champion_evidence(mlflow_config, metric=metric)
    decision = decide_promotion(
        model_name=candidate.model_name,
        candidate_version=candidate.version,
        candidate_score=float(getattr(winner_result, metric)),
        champion_version=champion_version,
        champion_score=champion_score,
        comparison_metric=metric,
    )
    apply_promotion_decision(
        config=mlflow_config,
        candidate=candidate,
        decision=decision,
    )
    return GovernanceRun(training, tuple(tracked), gate, candidate, decision)
