from unittest.mock import Mock

import numpy as np
import pytest

from mlforge.serving.inference import ChurnInferenceService, InferenceError
from mlforge.serving.model_loader import LoadedModel
from mlforge.serving.schemas import ChurnPredictionRequest


def _request() -> ChurnPredictionRequest:
    return ChurnPredictionRequest(
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=12,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="DSL",
        OnlineSecurity="Yes",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="Yes",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="One year",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=65.5,
        TotalCharges=786.0,
    )


def _loaded_model(probabilities: list[list[float]]) -> LoadedModel:
    pipeline = Mock()
    pipeline.classes_ = np.array([0, 1])
    pipeline.predict_proba.return_value = np.asarray(probabilities)
    return LoadedModel(
        pipeline=pipeline,
        model_name="MLForgeChurnClassifier",
        alias="champion",
        version="7",
        run_id="run-123",
        model_uri="models:/MLForgeChurnClassifier/7",
    )


def test_inference_returns_positive_prediction_at_threshold() -> None:
    loaded = _loaded_model([[0.5, 0.5]])
    service = ChurnInferenceService(loaded_model=loaded, threshold=0.5)

    response = service.predict(_request())

    assert response.prediction == "Yes"
    assert response.churn_probability == 0.5
    assert response.threshold == 0.5
    assert response.model_name == "MLForgeChurnClassifier"
    assert response.model_version == "7"
    loaded.pipeline.predict_proba.assert_called_once()


def test_inference_returns_negative_prediction_below_threshold() -> None:
    service = ChurnInferenceService(
        loaded_model=_loaded_model([[0.8, 0.2]]), threshold=0.5
    )

    response = service.predict(_request())

    assert response.prediction == "No"
    assert response.churn_probability == 0.2


def test_inference_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        ChurnInferenceService(loaded_model=_loaded_model([[0.5, 0.5]]), threshold=1.1)


def test_inference_wraps_model_prediction_failure() -> None:
    loaded = _loaded_model([[0.5, 0.5]])
    loaded.pipeline.predict_proba.side_effect = RuntimeError("internal model failure")
    service = ChurnInferenceService(loaded_model=loaded, threshold=0.5)

    with pytest.raises(InferenceError, match="Model inference failed"):
        service.predict(_request())


def test_inference_rejects_invalid_probability_shape() -> None:
    service = ChurnInferenceService(
        loaded_model=_loaded_model([[0.2, 0.3, 0.5]]), threshold=0.5
    )

    with pytest.raises(InferenceError, match="probability shape"):
        service.predict(_request())


def test_inference_uses_positive_class_position() -> None:
    loaded = _loaded_model([[0.7, 0.3]])
    loaded.pipeline.classes_ = np.array([1, 0])
    service = ChurnInferenceService(loaded_model=loaded, threshold=0.5)

    response = service.predict(_request())

    assert response.prediction == "Yes"
    assert response.churn_probability == 0.7
