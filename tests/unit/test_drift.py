import pandas as pd

from mlforge.monitoring.drift import detect_feature_drift, prediction_drift


def test_feature_drift_detects_shift() -> None:
    reference = pd.DataFrame({"tenure": [10, 11, 12, 13], "Contract": ["One", "One", "Two", "Two"]})
    current = pd.DataFrame({"tenure": [50, 51, 52, 53], "Contract": ["Month", "Month", "Month", "Month"]})
    result = detect_feature_drift(reference, current, dataset_threshold=0.5)
    assert result.dataset_drifted
    assert result.drifted_share == 1.0


def test_prediction_drift_uses_mean_probability_shift() -> None:
    assert prediction_drift(pd.Series([0.2, 0.3]), pd.Series([0.5, 0.6])) == pytest.approx(0.3)
