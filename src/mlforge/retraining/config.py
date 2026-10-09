"""Configuration for governed retraining."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RetrainingConfig:
    enabled: bool
    require_monitoring_signal: bool
    data_path: Path
    training_config_path: Path
    mlflow_config_path: Path
    evidence_path: Path


def load_retraining_config(
    path: str | Path = "configs/retraining.yaml",
) -> RetrainingConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Retraining configuration not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    section = raw.get("retraining") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        raise ValueError("retraining must be a mapping.")
    for key in ("enabled", "require_monitoring_signal"):
        if not isinstance(section.get(key), bool):
            raise ValueError(f"retraining.{key} must be boolean.")
    return RetrainingConfig(
        enabled=section["enabled"],
        require_monitoring_signal=section["require_monitoring_signal"],
        data_path=Path(section["data_path"]),
        training_config_path=Path(section["training_config_path"]),
        mlflow_config_path=Path(section["mlflow_config_path"]),
        evidence_path=Path(section["evidence_path"]),
    )
