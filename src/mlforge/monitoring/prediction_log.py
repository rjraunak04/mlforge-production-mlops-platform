"""Privacy-conscious production prediction event logging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PredictionEvent:
    timestamp: str
    request_id: str
    model_name: str
    model_version: str
    prediction: str
    churn_probability: float
    features: dict[str, Any]


def build_prediction_event(
    *,
    request_id: str,
    model_name: str,
    model_version: str,
    prediction: str,
    churn_probability: float,
    features: dict[str, Any],
) -> PredictionEvent:
    return PredictionEvent(
        timestamp=datetime.now(UTC).isoformat(),
        request_id=request_id,
        model_name=model_name,
        model_version=model_version,
        prediction=prediction,
        churn_probability=float(churn_probability),
        features=dict(features),
    )


def append_prediction_event(path: str | Path, event: PredictionEvent) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(event), separators=(",", ":")) + "\n")


def read_prediction_events(path: str | Path) -> list[PredictionEvent]:
    source = Path(path)
    if not source.exists():
        return []
    events = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(PredictionEvent(**json.loads(line)))
    return events
