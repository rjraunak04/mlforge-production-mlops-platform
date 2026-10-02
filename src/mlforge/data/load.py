"""Validated ingestion for the MLForge reference dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlforge.data.schema import EXPECTED_COLUMNS, validate_churn_data


def load_churn_data(path: str | Path) -> pd.DataFrame:
    """Load, minimally normalize, and validate the raw churn CSV."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")

    data = pd.read_csv(source, dtype={"TotalCharges": "string"})

    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError("Dataset columns do not match the expected contract.")

    data = data.copy()
    total_charges = data["TotalCharges"].str.strip()
    data["TotalCharges"] = pd.to_numeric(
        total_charges.mask(total_charges.eq("")),
        errors="raise",
    )

    return validate_churn_data(data)
