"""Lightweight integration tests for the retail demand forecasting pipeline."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_pipeline_modules_import_successfully() -> None:
    """Verify that major pipeline modules can be imported together."""
    modules = [
        "mlops_se489.data.make_dataset",
        "mlops_se489.features.build_features",
        "mlops_se489.models.train_model",
        "mlops_se489.models.predict_model",
        "mlops_se489.models.compare_models",
        "mlops_se489.logging_config",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None


def test_pipeline_project_structure_exists() -> None:
    """Verify the project contains the expected pipeline structure."""
    project_root = Path(__file__).resolve().parents[1]

    expected_paths = [
        project_root / "src" / "mlops_se489" / "data",
        project_root / "src" / "mlops_se489" / "features",
        project_root / "src" / "mlops_se489" / "models",
        project_root / "src" / "mlops_se489" / "logging_config.py",
        project_root / "tests",
        project_root / "Makefile",
        project_root / "requirements.txt",
        project_root / "pyproject.toml",
    ]

    for path in expected_paths:
        assert path.exists(), f"Missing expected project path: {path}"


def test_makefile_pipeline_targets_exist() -> None:
    """Verify Makefile includes expected pipeline targets."""
    project_root = Path(__file__).resolve().parents[1]
    makefile = project_root / "Makefile"

    content = makefile.read_text(encoding="utf-8")

    expected_targets = [
        "data:",
        "features:",
        "train:",
        "predict:",
        "test:",
        "lint:",
        "format:",
    ]

    for target in expected_targets:
        assert target in content, f"Missing Makefile target: {target}"
