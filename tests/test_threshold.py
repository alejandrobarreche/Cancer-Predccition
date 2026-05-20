"""Tests del ajuste de umbral anti-leakage (`src.models.threshold.tune_threshold`)."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.threshold import tune_threshold


@pytest.fixture
def separable_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Genera y_true / y_proba con buena separación: positivos en [0.6,1.0], negativos en [0.0,0.4]."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.binomial(1, 0.3, size=n)
    y_proba = np.where(
        y_true == 1,
        rng.uniform(0.6, 1.0, size=n),
        rng.uniform(0.0, 0.4, size=n),
    )
    return y_true, y_proba


def test_tune_threshold_finds_high_f1(separable_predictions):
    """Con predicciones bien separadas, F1 óptimo debe ser cercano a 1.0."""
    y_true, y_proba = separable_predictions
    result = tune_threshold(y_true, y_proba)

    assert result["f1_at_threshold"] > 0.95
    assert 0.3 <= result["threshold_f1"] <= 0.7


def test_tune_threshold_returns_youden(separable_predictions):
    """El barrido también debe devolver el umbral por índice de Youden."""
    y_true, y_proba = separable_predictions
    result = tune_threshold(y_true, y_proba)

    assert "threshold_youden" in result
    assert 0.0 < result["threshold_youden"] < 1.0


def test_tune_threshold_sweep_arrays_aligned(separable_predictions):
    """sweep_thresholds y sweep_{f1,precision,recall} deben tener misma longitud."""
    y_true, y_proba = separable_predictions
    result = tune_threshold(y_true, y_proba)

    n = len(result["sweep_thresholds"])
    assert n == len(result["sweep_f1"]) == len(result["sweep_precision"]) == len(result["sweep_recall"])
    assert n > 0
