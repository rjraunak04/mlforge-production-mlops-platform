"""Integration tests for the complete Day 3 governance workflow."""

from pathlib import Path

import pandas as pd
import yaml
from mlflow import MlflowClient

from mlforge.governance.workflow import run_governed_training


def _dataset(path: Path, rows: int = 120) -> None:
    records = []
    for index in range(rows):
        churn = "Yes" if index % 4 == 0 else "No"
        records.append(
            {
                "customerID": f"C{index:04d}",
                "gender": "Male" if index % 2 else "Female",
                "SeniorCitizen": index % 2,
                "Partner": "Yes" if index % 3 else "No",
                "Dependents": "No",
                "tenure": index % 72,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 30.0 + index % 50,
                "TotalCharges": str(100.0 + index * 20),
                "Churn": churn,
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)


def _configs(tmp_path: Path, *, gate: float = 0.0) -> tuple[Path, Path]:
    training = {
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
                "max_iter": 200,
                "class_weight": None,
            },
            "random_forest": {
                "enabled": False,
                "n_estimators": 10,
                "max_depth": None,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
                "n_jobs": 1,
            },
            "hist_gradient_boosting": {
                "enabled": False,
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
    training_path = tmp_path / "training.yaml"
    training_path.write_text(yaml.safe_dump(training), encoding="utf-8")

    database = (tmp_path / "mlflow.db").as_posix()
    uri = f"sqlite:///{database}"
    mlflow_config = {
        "tracking": {
            "uri": uri,
            "artifact_root": (tmp_path / "artifacts").resolve().as_uri(),
            "experiment_name": "governance-integration",
            "registry_uri": uri,
        },
        "registry": {"model_name": "MLForgeIntegrationClassifier"},
        "quality_gates": {
            "min_roc_auc": gate,
            "min_average_precision": gate,
            "min_recall": gate,
        },
        "tags": {"project": "mlforge", "lifecycle_stage": "validation"},
    }
    mlflow_path = tmp_path / "mlflow.yaml"
    mlflow_path.write_text(yaml.safe_dump(mlflow_config), encoding="utf-8")
    return training_path, mlflow_path


def test_governed_workflow_tracks_registers_and_promotes(tmp_path: Path) -> None:
    data_path = tmp_path / "churn.csv"
    _dataset(data_path)
    training_path, mlflow_path = _configs(tmp_path)

    result = run_governed_training(
        data_path,
        training_config_path=training_path,
        mlflow_config_path=mlflow_path,
    )

    assert len(result.tracked_runs) == 1
    assert result.quality_gate.passed is True
    assert result.registered_candidate is not None
    assert result.promotion is not None and result.promotion.promoted is True
    assert len(result.training.locked_test.target) > 0
    assert all(
        item.partition == "validation" for item in result.training.validation_results
    )

    config = yaml.safe_load(mlflow_path.read_text(encoding="utf-8"))
    client = MlflowClient(
        tracking_uri=config["tracking"]["uri"],
        registry_uri=config["tracking"]["registry_uri"],
    )
    champion = client.get_model_version_by_alias(
        config["registry"]["model_name"], "champion"
    )
    assert str(champion.version) == result.registered_candidate.version


def test_failed_gate_stops_before_registry(tmp_path: Path) -> None:
    data_path = tmp_path / "churn.csv"
    _dataset(data_path)
    training_path, mlflow_path = _configs(tmp_path, gate=1.0)

    result = run_governed_training(
        data_path,
        training_config_path=training_path,
        mlflow_config_path=mlflow_path,
    )

    assert result.quality_gate.passed is False
    assert result.registered_candidate is None
    assert result.promotion is None
