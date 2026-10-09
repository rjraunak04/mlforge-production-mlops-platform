"""Configuration contracts for Day 05 monitoring."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PerformanceThresholds:
    min_roc_auc: float
    min_accuracy: float
    min_recall: float


@dataclass(frozen=True)
class MonitoringConfig:
    reference_path: Path
    production_log_path: Path
    report_dir: Path
    min_batch_size: int
    drift_share_threshold: float
    prediction_drift_threshold: float
    performance: PerformanceThresholds


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


def load_monitoring_config(\n    path: str | Path = "configs/monitoring.yaml",\n) -> MonitoringConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Monitoring configuration not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    section = raw.get("monitoring") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        raise ValueError("monitoring must be a mapping.")
    performance = section.get("performance")
    if not isinstance(performance, dict):
        raise ValueError("monitoring.performance must be a mapping.")
    size = section.get("min_batch_size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("monitoring.min_batch_size must be a positive integer.")
    return MonitoringConfig(
        reference_path=Path(section["reference_path"]),
        production_log_path=Path(section["production_log_path"]),
        report_dir=Path(section["report_dir"]),
        min_batch_size=size,
        drift_share_threshold=_probability(\n            section["drift_share_threshold"], "drift_share_threshold"\n        ),
        prediction_drift_threshold=_probability(\n            section["prediction_drift_threshold"], "prediction_drift_threshold"\n        ),
        performance=PerformanceThresholds(
            min_roc_auc=_probability(performance["min_roc_auc"], "min_roc_auc"),
            min_accuracy=_probability(performance["min_accuracy"], "min_accuracy"),
            min_recall=_probability(performance["min_recall"], "min_recall"),
        ),
    )
