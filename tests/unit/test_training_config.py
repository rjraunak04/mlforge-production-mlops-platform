"""Tests for YAML training configuration semantics."""

from __future__ import annotations

from pathlib import Path

from mlforge.training.runner import load_training_config


def test_target_labels_load_as_strings() -> None:
    config_path = Path("configs/training.yaml")

    config = load_training_config(config_path)

    assert config["training"]["positive_label"] == "Yes"
    assert config["training"]["negative_label"] == "No"
    assert isinstance(config["training"]["positive_label"], str)
    assert isinstance(config["training"]["negative_label"], str)
