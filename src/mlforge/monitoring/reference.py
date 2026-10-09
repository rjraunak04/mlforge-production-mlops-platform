"""Reference baseline creation for production drift monitoring."""

from pathlib import Path

import pandas as pd

from mlforge.features.preprocessing import FeatureColumns


def build_reference_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(FeatureColumns().all)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Reference data is missing features: {missing}")
    return frame.loc[:, columns].copy()


def save_reference_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_reference_dataset(frame).to_csv(destination, index=False)
    return destination
