"""Tests del módulo `src.models.train_mlp` (helpers no-Keras)."""
from __future__ import annotations

import json

import pytest


def test_load_xgboost_baseline_returns_test_metrics(tmp_path, monkeypatch):
    """Lee f1, auc_roc y threshold desde test_metrics + threshold del JSON."""
    pytest.importorskip("tensorflow")
    from src.models import train_mlp

    classical_dir = tmp_path / "classical"
    classical_dir.mkdir()
    (classical_dir / "xgboost.json").write_text(json.dumps({
        "threshold": 0.5,
        "test_metrics": {"f1": 0.55, "auc_roc": 0.84},
    }))
    monkeypatch.setattr(train_mlp, "REPORTS_DIR", tmp_path)

    assert train_mlp.load_xgboost_baseline() == {
        "f1": 0.55,
        "auc_roc": 0.84,
        "threshold": 0.5,
    }


def test_load_xgboost_baseline_missing_file_explains_remedy(tmp_path, monkeypatch):
    """Si no existe el JSON de XGBoost, falla con mensaje accionable."""
    pytest.importorskip("tensorflow")
    from src.models import train_mlp

    monkeypatch.setattr(train_mlp, "REPORTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="train_classical"):
        train_mlp.load_xgboost_baseline()
