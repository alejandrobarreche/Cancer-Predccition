"""Tests del preprocesador (StandardScaler + passthrough + OneHotEncoder)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.pipelines import build_preprocessor


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "edad":     [25, 60, 40, 80],
        "glucosa":  [90.0, 130.0, 100.0, 150.0],
        "fumador":  [0, 1, 0, 1],
        "zona":     ["Urbana", "Rural", "Urbana", "Semiurbana"],
    })


def test_preprocessor_output_shape():
    """2 numéricas + 1 binaria + OHE(3 niveles, drop=first) = 5 columnas."""
    pre = build_preprocessor(["edad", "glucosa"], ["fumador"], ["zona"])
    out = pre.fit_transform(_toy_df())
    assert out.shape == (4, 5)


def test_preprocessor_scales_numeric_to_zero_mean():
    """StandardScaler debe centrar las numéricas en media ~0."""
    pre = build_preprocessor(["edad", "glucosa"], ["fumador"], ["zona"])
    out = pre.fit_transform(_toy_df())
    numeric_block = out[:, :2]
    assert np.allclose(numeric_block.mean(axis=0), 0.0, atol=1e-6)


def test_preprocessor_passes_binary_unchanged():
    """Las binarias 0/1 deben llegar sin transformar."""
    pre = build_preprocessor(["edad", "glucosa"], ["fumador"], ["zona"])
    out = pre.fit_transform(_toy_df())
    binary_col = out[:, 2]
    assert set(np.unique(binary_col).tolist()) <= {0.0, 1.0}
