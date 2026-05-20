"""Construye `data/processed/{train,test}.csv` a partir de `data/interim/joined.csv`.

Decisiones (ver `reports/eda/memo.md`):
- Excluye leakage post-diagnostico: coste_total, coste_farmaco, num_ingresos,
  dias_hospital, vive.
- Excluye `alcohol` (varianza cero) y `paciente_id` (identificador).
- Mantiene todas las comorbilidades clinicas, mutaciones geneticas, bioquimicas,
  habitos y sociodemograficas (incluso las de baja senial: los modelos no
  lineales pueden encontrar interacciones).
- Aplica feature engineering clinico (src.features.engineer_clinical):
  * Tier 1: umbrales clinicos exactos del modelo generativo (binarias).
  * Tier 2: carga mutacional agregada (numericas).
  * Tier 3: actividad_fisica recodificada como ordinal actividad_score.

Los CSV resultantes contienen las features en su forma original (sin escalar,
sin OHE), incluidas las nuevas columnas derivadas. Cada modelo (clasico o MLP)
ajusta su propio preprocesado SOLO sobre train para evitar leakage.

Adicionalmente escribe `data/processed/feature_groups.json` con la clasificacion
de columnas (numericas a escalar / binarias / categoricas a OHE) para consumo
de `ml-classical` y `mlp-designer`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.engineer_clinical import add_clinical_features

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "joined.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "cancer"
ID_COL = "paciente_id"

# Excluidas por leakage post-diagnostico (ver eda_memo.md seccion 4).
LEAKAGE_COLS: tuple[str, ...] = (
    "coste_total",
    "coste_farmaco",
    "num_ingresos",
    "dias_hospital",
    "vive",
)

# Excluida por varianza cero (alcohol == 1 en todos los pacientes).
ZERO_VARIANCE_COLS: tuple[str, ...] = ("alcohol",)

NUMERIC_COLS: tuple[str, ...] = (
    "glucosa",
    "colesterol",
    "trigliceridos",
    "hemoglobina",
    "leucocitos",
    "plaquetas",
    "creatinina",
    "edad",
    "num_hijos",
    "distancia_hospital_km",
    # Tier 2: carga mutacional agregada
    "n_mutaciones_total",
    "n_mutaciones_alto_riesgo",
    # Tier 3: actividad fisica ordinal (reemplaza actividad_fisica categorica)
    "actividad_score",
)

BINARY_COLS: tuple[str, ...] = (
    "diabetes",
    "hipertension",
    "obesidad",
    "enfermedad_cardiaca",
    "asma",
    "epoc",
    "mut_BRCA1",
    "mut_TP53",
    "mut_EGFR",
    "mut_KRAS",
    "mut_PIK3CA",
    "mut_ALK",
    "mut_BRAF",
    "fumador",
    # Tier 1: umbrales clinicos exactos del modelo generativo
    "glucosa_alta",
    "hemoglobina_baja",
    "leucocitos_altos",
    "edad_riesgo",
)

CATEGORICAL_COLS: tuple[str, ...] = (
    "tipo_seguro",
    # actividad_fisica eliminada: sustituida por actividad_score (ordinal numerica)
    "nivel_educativo",
    "nivel_ingresos",
    "zona",
    "estado_civil",
)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build() -> dict[str, object]:
    """Genera los CSV de train/test y el manifiesto de feature groups.

    Pasos:
    1. Lee joined.csv desde data/interim/.
    2. Verifica columnas originales (pre-engineering).
    3. Aplica feature engineering clinico (add_clinical_features):
       las nuevas columnas son deterministas por fila, sin fit.
    4. Selecciona feature_cols (ya incluyen las columnas derivadas).
    5. Split 80/20 estratificado.
    6. Escribe train.csv, test.csv y feature_groups.json.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    LOGGER.info("Leyendo %s", INTERIM_PATH)
    df = pd.read_csv(INTERIM_PATH)

    # Columnas originales esperadas en el CSV crudo (antes de engineering).
    # Las columnas derivadas (Tier 1/2/3) se generan en add_clinical_features.
    ORIGINAL_NUMERIC = (
        "glucosa", "colesterol", "trigliceridos", "hemoglobina", "leucocitos",
        "plaquetas", "creatinina", "edad", "num_hijos", "distancia_hospital_km",
    )
    ORIGINAL_BINARY = (
        "diabetes", "hipertension", "obesidad", "enfermedad_cardiaca", "asma",
        "epoc", "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS", "mut_PIK3CA",
        "mut_ALK", "mut_BRAF", "fumador",
    )
    ORIGINAL_CATEGORICAL = (
        "tipo_seguro", "actividad_fisica", "nivel_educativo",
        "nivel_ingresos", "zona", "estado_civil",
    )
    expected = (
        {TARGET, ID_COL}
        | set(LEAKAGE_COLS)
        | set(ZERO_VARIANCE_COLS)
        | set(ORIGINAL_NUMERIC)
        | set(ORIGINAL_BINARY)
        | set(ORIGINAL_CATEGORICAL)
    )
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Columnas esperadas no encontradas: {sorted(missing)}")

    LOGGER.info("Filas: %d | Aplicando feature engineering clinico...", len(df))
    df = add_clinical_features(df)

    feature_cols = list(NUMERIC_COLS) + list(BINARY_COLS) + list(CATEGORICAL_COLS)
    drop_cols = [ID_COL, *LEAKAGE_COLS, *ZERO_VARIANCE_COLS]

    LOGGER.info("Features tras engineering: %d | excluidas: %d",
                len(feature_cols), len(drop_cols))

    X = df[feature_cols].copy()
    y = df[TARGET].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    train_df = X_train.assign(**{TARGET: y_train.values})
    test_df = X_test.assign(**{TARGET: y_test.values})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    feature_groups = {
        "target": TARGET,
        "numeric": list(NUMERIC_COLS),
        "binary": list(BINARY_COLS),
        "categorical": list(CATEGORICAL_COLS),
        "excluded_leakage": list(LEAKAGE_COLS),
        "excluded_zero_variance": list(ZERO_VARIANCE_COLS),
        "excluded_id": [ID_COL],
        "split": {
            "test_size": TEST_SIZE,
            "stratify": TARGET,
            "random_state": RANDOM_STATE,
        },
    }
    manifest_path = PROCESSED_DIR / "feature_groups.json"
    manifest_path.write_text(json.dumps(feature_groups, indent=2, ensure_ascii=False))

    summary = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "prevalence_train": float(y_train.mean()),
        "prevalence_test": float(y_test.mean()),
        "n_features": len(feature_cols),
        "train_path": str(train_path.relative_to(PROJECT_ROOT)),
        "test_path": str(test_path.relative_to(PROJECT_ROOT)),
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
    }
    LOGGER.info("Resumen: %s", json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    build()
