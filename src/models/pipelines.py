"""Construcción del preprocesador ColumnTransformer para modelos clásicos.

El preprocesador se ajusta SOLO sobre train para evitar data leakage.
Lee la clasificación de columnas desde data/processed/feature_groups.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_GROUPS_PATH = PROJECT_ROOT / "data" / "processed" / "feature_groups.json"


def load_feature_groups(path: Path = FEATURE_GROUPS_PATH) -> dict[str, Any]:
    """Carga el manifiesto de grupos de features desde JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_preprocessor(
    numeric_cols: list[str],
    binary_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Construye el ColumnTransformer con:
    - StandardScaler para columnas numéricas.
    - Passthrough para columnas binarias (ya son 0/1).
    - OneHotEncoder (drop='first') para columnas categóricas.

    Args:
        numeric_cols: Columnas numéricas a estandarizar.
        binary_cols: Columnas binarias a pasar sin transformación.
        categorical_cols: Columnas categóricas a codificar con OHE.

    Returns:
        ColumnTransformer listo para fit/transform.
    """
    LOGGER.info(
        "Construyendo preprocesador: %d numéricas, %d binarias, %d categóricas",
        len(numeric_cols),
        len(binary_cols),
        len(categorical_cols),
    )

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )

    transformers = [
        ("numeric", numeric_transformer, numeric_cols),
        ("binary", "passthrough", binary_cols),
        ("categorical", categorical_transformer, categorical_cols),
    ]

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


def build_preprocessor_from_manifest(
    path: Path = FEATURE_GROUPS_PATH,
) -> ColumnTransformer:
    """Construye el preprocesador directamente desde el manifiesto JSON.

    Args:
        path: Ruta al JSON con la clasificación de features.

    Returns:
        ColumnTransformer listo para fit/transform.
    """
    groups = load_feature_groups(path)
    return build_preprocessor(
        numeric_cols=groups["numeric"],
        binary_cols=groups["binary"],
        categorical_cols=groups["categorical"],
    )


def get_feature_names_out(
    preprocessor: ColumnTransformer,
    numeric_cols: list[str],
    binary_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    """Obtiene los nombres de features tras la transformación.

    Args:
        preprocessor: ColumnTransformer ya ajustado.
        numeric_cols: Columnas numéricas.
        binary_cols: Columnas binarias.
        categorical_cols: Columnas categóricas.

    Returns:
        Lista de nombres de features en el orden de transformación.
    """
    feature_names: list[str] = []

    # Numéricas: mismos nombres
    feature_names.extend(numeric_cols)

    # Binarias: mismos nombres
    feature_names.extend(binary_cols)

    # Categóricas: nombres generados por OHE
    ohe = preprocessor.named_transformers_["categorical"]
    if hasattr(ohe, "get_feature_names_out"):
        cat_names = ohe.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_names.tolist())

    return feature_names
