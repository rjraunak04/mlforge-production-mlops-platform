import pandas as pd
import pytest

from mlforge.features.preprocessing import FeatureColumns
from mlforge.monitoring.reference import build_reference_dataset


def test_reference_dataset_preserves_training_contract() -> None:
    frame = pd.DataFrame([{column: 0 for column in FeatureColumns().all}])
    result = build_reference_dataset(frame)
    assert tuple(result.columns) == FeatureColumns().all


def test_reference_dataset_rejects_missing_feature() -> None:
    with pytest.raises(ValueError, match="missing features"):
        build_reference_dataset(pd.DataFrame({"tenure": [1]}))
