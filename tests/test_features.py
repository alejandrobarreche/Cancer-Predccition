"""Tests de las constantes de selección de features (anti-leakage)."""
from __future__ import annotations

from src.features.build_dataset import (
    BINARY_COLS,
    CATEGORICAL_COLS,
    LEAKAGE_COLS,
    NUMERIC_COLS,
    ZERO_VARIANCE_COLS,
)


def test_feature_groups_disjoint():
    """Numéricas, binarias y categóricas no se solapan entre sí."""
    n, b, c = set(NUMERIC_COLS), set(BINARY_COLS), set(CATEGORICAL_COLS)
    assert n.isdisjoint(b)
    assert n.isdisjoint(c)
    assert b.isdisjoint(c)


def test_features_disjoint_from_excluded():
    """Ninguna feature activa coincide con una columna excluida (leakage o varianza cero)."""
    features = set(NUMERIC_COLS) | set(BINARY_COLS) | set(CATEGORICAL_COLS)
    excluded = set(LEAKAGE_COLS) | set(ZERO_VARIANCE_COLS)
    assert features.isdisjoint(excluded)


def test_known_leakage_cols_documented():
    """Las columnas con leakage post-diagnóstico confirmado están en LEAKAGE_COLS."""
    must_exclude = {"coste_total", "coste_farmaco", "num_ingresos", "dias_hospital", "vive"}
    assert must_exclude.issubset(set(LEAKAGE_COLS))


def test_alcohol_marked_as_zero_variance():
    """`alcohol` es constante en el dataset y debe estar excluida por varianza cero."""
    assert "alcohol" in ZERO_VARIANCE_COLS
