"""Production inference service for the registered churn champion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mlforge.serving.model_loader import LoadedModel
from mlforge.serving.schemas import ChurnPredictionRequest, ChurnPredictionResponse


class InferenceError(RuntimeError):
    """Raised when a loaded model cannot produce a safe prediction."""


@dataclass(frozen=True)
class ChurnInferenceService:
    """Run deterministic binary inference with one pinned registry model."""

    loaded_model: LoadedModel
    threshold: float

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Classification threshold must be between 0 and 1.")

    def predict(self, request: ChurnPredictionRequest) -> ChurnPredictionResponse:
        """Predict churn probability and apply the configured decision threshold."""
        try:
            probabilities = self.loaded_model.pipeline.predict_proba(request.to_frame())
        except Exception as exc:
            raise InferenceError("Model inference failed.") from exc

        probability = _positive_class_probability(
            probabilities,
            getattr(self.loaded_model.pipeline, "classes_", None),
        )
        prediction = "Yes" if probability >= self.threshold else "No"
        return ChurnPredictionResponse(
            prediction=prediction,
            churn_probability=probability,
            threshold=self.threshold,
            model_name=self.loaded_model.model_name,
            model_version=self.loaded_model.version,
        )


def _positive_class_probability(probabilities: object, classes: object) -> float:
    """Extract the probability for encoded positive class 1 safely."""
    array = np.asarray(probabilities)
    if array.shape != (1, 2):
        raise InferenceError("Model returned an invalid probability shape.")

    if classes is None:
        positive_index = 1
    else:
        labels = list(np.asarray(classes).tolist())
        if 1 not in labels:
            raise InferenceError("Model does not expose encoded positive class 1.")
        positive_index = labels.index(1)

    probability = float(array[0, positive_index])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise InferenceError("Model returned an invalid churn probability.")
    return probability
