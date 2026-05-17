"""Smoke tests para la arquitectura MLP (`src.models.mlp.build_model`)."""
from __future__ import annotations

import pytest


def _build(**kwargs):
    pytest.importorskip("tensorflow")
    from src.models.mlp import build_model
    return build_model(**kwargs)


def test_default_architecture_matches_target_params():
    """Con input_dim=39 y la arquitectura por defecto, ~46.881 params (objetivo ~46.913)."""
    model = _build(
        input_dim=39,
        hidden_units=(300, 100, 30),
        dropout_rates=(0.25, 0.25, 0.20),
    )
    assert model.count_params() == 46_881


def test_three_hidden_dense_layers_plus_output():
    """3 Dense ocultas + 1 Dense de salida = 4 capas Dense en total."""
    model = _build(input_dim=39)
    dense = [l for l in model.layers if l.__class__.__name__ == "Dense"]
    assert len(dense) == 4
    # La capa de salida tiene 1 unidad (clasificación binaria)
    assert dense[-1].units == 1


def test_requires_minimum_three_hidden_layers():
    """El enunciado pide al menos 3 capas ocultas."""
    pytest.importorskip("tensorflow")
    from src.models.mlp import build_model
    with pytest.raises(ValueError, match="al menos 3 capas"):
        build_model(input_dim=10, hidden_units=(32, 16), dropout_rates=(0.2, 0.2))


def test_hidden_and_dropout_lengths_must_match():
    """hidden_units y dropout_rates deben tener la misma longitud."""
    pytest.importorskip("tensorflow")
    from src.models.mlp import build_model
    with pytest.raises(ValueError, match="misma longitud"):
        build_model(input_dim=10, hidden_units=(32, 16, 8), dropout_rates=(0.2, 0.2))
