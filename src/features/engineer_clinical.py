"""Feature engineering clínico para el dataset de cáncer.

Añade features derivadas con conocimiento de dominio sobre el dataset de cáncer
(ver metadata_dataset_cancer.md, sección 'Modelo generativo').

Las features son ROW-LEVEL DETERMINISTAS: no requieren fit sobre train, no hay
riesgo de data leakage al aplicarlas también sobre test.csv.

Tier 1 — Umbrales clínicos exactos del modelo generativo (binarias 0/1):
    glucosa_alta         = (glucosa > 130).astype(int)
    hemoglobina_baja     = (hemoglobina < 11).astype(int)
    leucocitos_altos     = (leucocitos > 10).astype(int)
    edad_riesgo          = (edad > 55).astype(int)

Tier 2 — Carga mutacional agregada (numéricas):
    n_mutaciones_total          = suma de las 7 mutaciones (0-7)
    n_mutaciones_alto_riesgo    = mut_BRCA1 + mut_TP53 + mut_KRAS (0-3)

Tier 3 — Recodificación ordinal de actividad_fisica (numérica):
    actividad_score: Baja=0, Moderada=1, Alta=2
    La columna original 'actividad_fisica' se elimina del DataFrame.

Uso desde build_dataset.py:
    from src.features.engineer_clinical import add_clinical_features
    df = add_clinical_features(df)
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Columnas de mutaciones para carga agregada
MUTATION_COLS: tuple[str, ...] = (
    "mut_BRCA1",
    "mut_TP53",
    "mut_EGFR",
    "mut_KRAS",
    "mut_PIK3CA",
    "mut_ALK",
    "mut_BRAF",
)
HIGH_RISK_MUTATION_COLS: tuple[str, ...] = ("mut_BRCA1", "mut_TP53", "mut_KRAS")

ACTIVIDAD_ORDINAL: dict[str, int] = {"Baja": 0, "Moderada": 1, "Alta": 2}


def add_clinical_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Añade features clínicas derivadas al DataFrame.

    Modifica una copia del DataFrame (o el propio si inplace=True).
    No requiere ajuste sobre ningún subconjunto — las transformaciones
    son deterministas por fila.

    Args:
        df: DataFrame con las columnas originales del dataset de cáncer.
        inplace: Si True, modifica el DataFrame original. Por defecto False.

    Returns:
        DataFrame ampliado con las nuevas columnas.

    Raises:
        KeyError: Si alguna columna requerida no está presente en df.
    """
    required: Sequence[str] = [
        "glucosa", "hemoglobina", "leucocitos", "edad",
        "actividad_fisica",
        *MUTATION_COLS,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Columnas requeridas para feature engineering clínico no encontradas: {missing}"
        )

    out = df if inplace else df.copy()

    # --- Tier 1: umbrales clínicos ---
    out["glucosa_alta"] = (out["glucosa"] > 130).astype(int)
    out["hemoglobina_baja"] = (out["hemoglobina"] < 11).astype(int)
    out["leucocitos_altos"] = (out["leucocitos"] > 10).astype(int)
    out["edad_riesgo"] = (out["edad"] > 55).astype(int)

    # --- Tier 2: carga mutacional ---
    out["n_mutaciones_total"] = out[list(MUTATION_COLS)].sum(axis=1).astype(int)
    out["n_mutaciones_alto_riesgo"] = (
        out[list(HIGH_RISK_MUTATION_COLS)].sum(axis=1).astype(int)
    )

    # --- Tier 3: actividad física ordinal ---
    unknown_vals = set(out["actividad_fisica"].dropna().unique()) - set(ACTIVIDAD_ORDINAL)
    if unknown_vals:
        LOGGER.warning(
            "Valores inesperados en actividad_fisica: %s. "
            "Se mapearán a NaN y luego a 0 (Baja).",
            unknown_vals,
        )
    out["actividad_score"] = (
        out["actividad_fisica"].map(ACTIVIDAD_ORDINAL).fillna(0).astype(int)
    )
    out = out.drop(columns=["actividad_fisica"])

    n_new = 7  # glucosa_alta, hemoglobina_baja, leucocitos_altos, edad_riesgo,
               # n_mutaciones_total, n_mutaciones_alto_riesgo, actividad_score
    LOGGER.info(
        "Feature engineering clínico completado: %d columnas añadidas, "
        "'actividad_fisica' eliminada. Shape resultante: %s",
        n_new,
        out.shape,
    )

    return out


# ─── CLI para prueba rápida ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/train.csv")
    df = pd.read_csv(path)
    out = add_clinical_features(df)
    print(f"Shape antes: {df.shape} → después: {out.shape}")
    new_cols = ["glucosa_alta", "hemoglobina_baja", "leucocitos_altos",
                "edad_riesgo", "n_mutaciones_total", "n_mutaciones_alto_riesgo",
                "actividad_score"]
    print(out[new_cols].describe())
