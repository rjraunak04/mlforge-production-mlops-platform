"""Safe retraining orchestration that reuses Day 03 governance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mlforge.governance.workflow import GovernanceRun, run_governed_training
from mlforge.monitoring.report import MonitoringReport
from mlforge.retraining.config import RetrainingConfig
from mlforge.retraining.trigger import RetrainingDecision, decide_retraining


@dataclass(frozen=True)
class RetrainingRun:
    decision: RetrainingDecision
    governance: GovernanceRun | None


def run_retraining(
    config: RetrainingConfig,
    monitoring: MonitoringReport | None,
) -> RetrainingRun:
    decision = decide_retraining(monitoring, config)
    if not decision.should_retrain:
        return RetrainingRun(decision, None)
    governance = run_governed_training(
        config.data_path,
        training_config_path=config.training_config_path,
        mlflow_config_path=config.mlflow_config_path,
    )
    return RetrainingRun(decision, governance)


def write_retraining_evidence(run: RetrainingRun, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    governance = run.governance
    evidence = {
        "decision": asdict(run.decision),
        "candidate_version": (
            governance.registered_candidate.version
            if governance and governance.registered_candidate
            else None
        ),
        "quality_gate_passed": (
            governance.quality_gate.passed if governance else None
        ),
        "promoted": (
            governance.promotion.promoted
            if governance and governance.promotion
            else None
        ),
    }
    destination.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return destination
