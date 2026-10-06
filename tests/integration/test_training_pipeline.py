"""Integration tests for the end-to-end Day 2 training pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from mlforge.data.schema import EXPECTED_COLUMNS
from mlforge.training.runner import run_training_pipeline


def _dataset(rows: int = 80) -> pd.DataFrame:
    records = []
    for index in range(rows):
        records.append(
            {
                "customerID": f"C{index:04d}",
                "gender": "Female" if index % 2 == 0 else "Male",
                "SeniorCitizen": index % 2,
                "Partner": "Yes" if index % 3 == 0 else "No",
                "Dependents": "Yes" if index % 5 == 0 else "No",
                "tenure": index + 1,
                "PhoneService": "Yes",
                "MultipleLines": "Yes" if index % 2 else "No",
                "InternetService": "Fiber optic" if index % 3 else "DSL",
                "OnlineSecurity": "Yes" if index % 4 else "No",
                "OnlineBackup": "No" if index % 3 else "Yes",
                "DeviceProtection": "Yes" if index % 2 else "No",
                "TechSupport": "No" if index % 4 else "Yes",
                "StreamingTV": "Yes" if index % 2 else "No",
                "StreamingMovies": "No" if index % 2 else "Yes",
                "Contract": "Month-to-month" if index % 3 else "One year",
                "PaperlessBilling": "Yes" if index % 2 else "No",
                "PaymentMethod": ("Electronic check" if index % 2 else "Mailed check"),
                "MonthlyCharges": 20.0 + index,
                "TotalCharges": (20.0 + index) * (index + 1),
                "Churn": "Yes" if index % 4 == 0 else "No",
            }
        )
    return pd.DataFrame.from_records(records, columns=EXPECTED_COLUMNS)


def _config(path: Path) -> Path:
    config = {
        "training": {
            "target_column": "Churn",
            "id_column": "customerID",
            "positive_label": "Yes",
            "negative_label": "No",
            "random_state": 42,
        },
        "models": {
            "logistic_regression": {
                "enabled": True,
                "max_iter": 500,
                "class_weight": None,
            },
            "random_forest": {
                "enabled": True,
                "n_estimators": 20,
                "max_depth": None,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
                "n_jobs": 1,
            },
            "hist_gradient_boosting": {
                "enabled": True,
                "max_iter": 20,
                "learning_rate": 0.1,
                "max_leaf_nodes": 15,
            },
        },
        "evaluation": {
            "selection_metric": "roc_auc",
            "classification_threshold": 0.5,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_end_to_end_training_selects_from_validation_only(tmp_path: Path) -> None:
    data_path = tmp_path / "churn.csv"
    _dataset().to_csv(data_path, index=False)
    config_path = _config(tmp_path / "training.yaml")

    run = run_training_pipeline(data_path, training_config_path=config_path)

    assert len(run.train.target) == 56
    assert len(run.validation.target) == 12
    assert len(run.locked_test.target) == 12
    assert set(run.models) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert all(result.partition == "validation" for result in run.validation_results)
    assert run.comparison.selected_model in run.models


def test_locked_test_is_disjoint_from_training_and_validation(tmp_path: Path) -> None:
    data_path = tmp_path / "churn.csv"
    _dataset().to_csv(data_path, index=False)
    config_path = _config(tmp_path / "training.yaml")

    run = run_training_pipeline(data_path, training_config_path=config_path)

    train_ids = set(run.train.customer_ids)
    validation_ids = set(run.validation.customer_ids)
    test_ids = set(run.locked_test.customer_ids)

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
