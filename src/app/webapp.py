"""Aplicación web FastAPI — Dashboard del estudio de predicción de cáncer (UAX).

Reemplaza la antigua app de Streamlit (`streamlit_app.py`) por una single-page
app con FastAPI + Jinja2 + Plotly.js. Sirve:

- Dashboard estático con métricas, curvas ROC/PR interactivas y comparativa de
  los 5 modelos del estudio (precomputado al arrancar).
- Endpoint POST /api/predict para inferencia en vivo (XGBoost + MLP).
- Endpoint GET /api/distribution/{feature} para distribuciones por clase.

Ejecutar:
    conda activate uax-tf
    uvicorn src.app.webapp:app --reload --port 8000
"""

from __future__ import annotations

# IMPORTANTE: TensorFlow tiene que importarse ANTES que sklearn/pandas/joblib
# en macOS ARM con TF 2.21 + sklearn 1.5 (ver memoria del proyecto
# `feedback_tf_import_order.md`). De lo contrario, model.predict() entra en
# deadlock indefinido en el primer request al cargar la MLP de forma diferida.
import os as _os

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as _tf  # noqa: F401  — se importa primero por orden

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.features.engineer_clinical import add_clinical_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
LOGGER = logging.getLogger(__name__)

# ─── Rutas del proyecto ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

REPORTS = PROJECT_ROOT / "reports"
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"

# Pre-procesado: orden canónico de columnas que espera la pipeline
RAW_NUMERIC_BASE = [
    "glucosa", "colesterol", "trigliceridos", "hemoglobina",
    "leucocitos", "plaquetas", "creatinina", "edad",
    "num_hijos", "distancia_hospital_km",
]
RAW_BINARY = [
    "diabetes", "hipertension", "obesidad", "enfermedad_cardiaca", "asma", "epoc",
    "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS",
    "mut_PIK3CA", "mut_ALK", "mut_BRAF", "fumador",
]
RAW_CATEGORICAL = ["tipo_seguro", "nivel_educativo", "nivel_ingresos", "zona", "estado_civil"]
ACTIVIDAD_LEVELS = ["Baja", "Moderada", "Alta"]
CATEGORICAL_LEVELS = {
    "tipo_seguro": ["Mixto", "Privado", "Publico"],
    "nivel_educativo": ["Primaria", "Secundaria", "Sin estudios", "Universitario"],
    "nivel_ingresos": ["Alto", "Bajo", "Medio", "Muy bajo"],
    "zona": ["Rural", "Semiurbana", "Urbana"],
    "estado_civil": ["Casado", "Divorciado", "Soltero", "Viudo"],
}

MODEL_FILES = {
    "logistic_regression": "Regresión logística",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "histgradientboosting": "HistGradientBoosting",
}

# ─── Carga de artefactos al arrancar ──────────────────────────────────────────


class Artifacts:
    """Contenedor de modelos y datos cargados una sola vez."""

    def __init__(self) -> None:
        LOGGER.info("Cargando artefactos…")
        self.eda = json.loads((REPORTS / "eda" / "report.json").read_text())
        self.classical: list[dict[str, Any]] = json.loads(
            (REPORTS / "classical" / "results.json").read_text()
        )
        self.mlp_results = json.loads((REPORTS / "mlp" / "results.json").read_text())
        self.feature_groups = json.loads((PROCESSED / "feature_groups.json").read_text())

        # Cargar modelos clásicos (sklearn Pipelines completos)
        self.classical_models = {
            key: joblib.load(MODELS / f"{key}.pkl")
            for key in MODEL_FILES
        }

        # MLP: cargamos eager porque TF está disponible desde el primer import
        # (mantenemos el patrón de atributos privados para no romper el resto).
        self._mlp = None
        self._mlp_preprocessor = None
        self._eager_load_mlp = True  # cargado al final de __init__

        # Test set para curvas ROC/PR
        self.test_df = pd.read_csv(PROCESSED / "test.csv")
        self.train_df = pd.read_csv(PROCESSED / "train.csv")
        self.y_test = self.test_df["cancer"].values
        feature_cols = (
            self.feature_groups["numeric"]
            + self.feature_groups["binary"]
            + self.feature_groups["categorical"]
        )
        self.X_test = self.test_df[feature_cols]

        # Pre-calcular probas + curvas para los 5 modelos
        self.curves: dict[str, dict[str, Any]] = {}
        self._compute_classical_curves()

        # Cargar MLP eager para que el primer request sea rápido
        LOGGER.info("Cargando MLP (TF/Keras)…")
        _ = self.mlp  # dispara load + cálculo de curva
        LOGGER.info("Artefactos cargados ✓ (MLP listo)")

    @property
    def mlp(self):
        if self._mlp is None:
            import tensorflow as tf  # import diferido

            self._mlp = tf.keras.models.load_model(MODELS / "mlp.keras")
            self._mlp_preprocessor = joblib.load(MODELS / "mlp_preprocessor.pkl")
            self._compute_mlp_curve()
        return self._mlp

    @property
    def mlp_preprocessor(self):
        if self._mlp_preprocessor is None:
            _ = self.mlp  # trigger lazy load
        return self._mlp_preprocessor

    def _compute_classical_curves(self) -> None:
        for key, label in MODEL_FILES.items():
            proba = self.classical_models[key].predict_proba(self.X_test)[:, 1]
            self.curves[key] = self._curve_dict(label, proba)

    def _compute_mlp_curve(self) -> None:
        X = self._mlp_preprocessor.transform(self.X_test).astype(np.float32)
        proba = self._mlp.predict(X, batch_size=256, verbose=0).ravel()
        self.curves["mlp"] = self._curve_dict("MLP", proba)

    def _curve_dict(self, label: str, proba: np.ndarray) -> dict[str, Any]:
        fpr, tpr, _ = roc_curve(self.y_test, proba)
        prec, rec, _ = precision_recall_curve(self.y_test, proba)
        # Submuestrear para no enviar 10k puntos por curva (UI sigue suave con ~200)
        idx_roc = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
        idx_pr = np.linspace(0, len(prec) - 1, min(200, len(prec))).astype(int)
        return {
            "label": label,
            "auc_roc": float(roc_auc_score(self.y_test, proba)),
            "ap": float(average_precision_score(self.y_test, proba)),
            "fpr": fpr[idx_roc].tolist(),
            "tpr": tpr[idx_roc].tolist(),
            "precision": prec[idx_pr].tolist(),
            "recall": rec[idx_pr].tolist(),
        }


ARTIFACTS = Artifacts()


# ─── Pre-cómputo: distribuciones para el explorador ───────────────────────────


def precompute_distributions() -> dict[str, dict[str, Any]]:
    """Pre-computa histogramas (por clase) de las features numéricas clave."""
    df = ARTIFACTS.train_df
    y = df["cancer"].values

    features = [
        ("glucosa", "Glucosa (mg/dL)"),
        ("colesterol", "Colesterol (mg/dL)"),
        ("hemoglobina", "Hemoglobina (g/dL)"),
        ("leucocitos", "Leucocitos (×10³/μL)"),
        ("edad", "Edad (años)"),
        ("trigliceridos", "Triglicéridos (mg/dL)"),
        ("plaquetas", "Plaquetas (×10³/μL)"),
        ("creatinina", "Creatinina (mg/dL)"),
        ("n_mutaciones_total", "Carga mutacional total"),
        ("n_mutaciones_alto_riesgo", "Mutaciones alto riesgo (BRCA1+TP53+KRAS)"),
    ]

    out = {}
    for col, label in features:
        if col not in df.columns:
            continue
        vals = df[col].values
        bins = np.linspace(np.min(vals), np.max(vals), 31)
        hist_neg, edges = np.histogram(vals[y == 0], bins=bins, density=True)
        hist_pos, _ = np.histogram(vals[y == 1], bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        out[col] = {
            "label": label,
            "centers": centers.tolist(),
            "hist_negative": hist_neg.tolist(),
            "hist_positive": hist_pos.tolist(),
            "median_negative": float(np.median(vals[y == 0])),
            "median_positive": float(np.median(vals[y == 1])),
        }
    return out


DISTRIBUTIONS = precompute_distributions()


# ─── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Predicción de Cáncer · UAX",
    description="Dashboard del estudio de IA — Universidad Alfonso X el Sabio 2025/26",
    version="1.0.0",
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Payload de inferencia ────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    glucosa: float = Field(..., ge=40, le=300)
    colesterol: float = Field(..., ge=80, le=400)
    trigliceridos: float = Field(..., ge=30, le=500)
    hemoglobina: float = Field(..., ge=5, le=22)
    leucocitos: float = Field(..., ge=1, le=30)
    plaquetas: float = Field(..., ge=50, le=600)
    creatinina: float = Field(..., ge=0.1, le=5)
    edad: float = Field(..., ge=18, le=100)
    num_hijos: int = Field(..., ge=0, le=15)
    distancia_hospital_km: float = Field(..., ge=0, le=500)
    diabetes: int = Field(..., ge=0, le=1)
    hipertension: int = Field(..., ge=0, le=1)
    obesidad: int = Field(..., ge=0, le=1)
    enfermedad_cardiaca: int = Field(..., ge=0, le=1)
    asma: int = Field(..., ge=0, le=1)
    epoc: int = Field(..., ge=0, le=1)
    mut_BRCA1: int = Field(..., ge=0, le=1)
    mut_TP53: int = Field(..., ge=0, le=1)
    mut_EGFR: int = Field(..., ge=0, le=1)
    mut_KRAS: int = Field(..., ge=0, le=1)
    mut_PIK3CA: int = Field(..., ge=0, le=1)
    mut_ALK: int = Field(..., ge=0, le=1)
    mut_BRAF: int = Field(..., ge=0, le=1)
    fumador: int = Field(..., ge=0, le=1)
    tipo_seguro: str
    actividad_fisica: str
    nivel_educativo: str
    nivel_ingresos: str
    zona: str
    estado_civil: str


# ─── Datos que pasamos al template ────────────────────────────────────────────


def build_template_context() -> dict[str, Any]:
    """Empaqueta toda la información que la página necesita para renderizar."""
    eda = ARTIFACTS.eda
    mlp = ARTIFACTS.mlp_results
    classical = ARTIFACTS.classical

    # Ordenar clásicos por F1 óptimo desc
    classical_sorted = sorted(
        classical, key=lambda r: r["test_metrics_optimal"]["f1"], reverse=True
    )

    # Modelos para la comparativa (incluye MLP)
    comparison_rows = []
    for r in classical_sorted:
        comparison_rows.append({
            "key": r["model_key"],
            "name": r["model_name"],
            "type": "Clásico",
            "threshold": r["threshold_optimal_f1"],
            "f1": r["test_metrics_optimal"]["f1"],
            "auc": r["test_metrics_optimal"]["auc_roc"],
            "precision": r["test_metrics_optimal"]["precision"],
            "recall": r["test_metrics_optimal"]["recall"],
            "training_time": r["training_time_seconds"],
        })
    comparison_rows.append({
        "key": "mlp",
        "name": "MLP (TensorFlow)",
        "type": "Red neuronal",
        "threshold": mlp["threshold_optimal_f1"],
        "f1": mlp["test_metrics_optimal"]["f1"],
        "auc": mlp["test_metrics_optimal"]["auc_roc"],
        "precision": mlp["test_metrics_optimal"]["precision"],
        "recall": mlp["test_metrics_optimal"]["recall"],
        "training_time": mlp["training_time_seconds"],
    })
    comparison_rows.sort(key=lambda r: r["f1"], reverse=True)

    # Forzar lazy-load del MLP para que las curvas estén disponibles
    _ = ARTIFACTS.mlp
    curves = ARTIFACTS.curves

    # XGBoost feature importances (top 12)
    xgb_record = next(r for r in classical if r["model_key"] == "xgboost")
    xgb_importances = xgb_record.get("feature_importances_top20", [])[:12]

    # Matriz de confusión del modelo final
    mlp_cm_optimal = mlp["test_metrics_optimal"]["confusion_matrix"]
    mlp_cm_default = mlp["test_metrics_default"]["confusion_matrix"]

    # Training history para el gráfico de aprendizaje
    history = mlp.get("training_history", {})

    context = {
        "title": "Predicción de Cáncer · UAX",
        # KPI hero
        "n_total": eda["n_total"],
        "n_positivos": eda["n_positivos"],
        "n_negativos": eda["n_negativos"],
        "prevalencia": eda["prevalencia_cancer"],
        "ratio": eda["ratio_desbalance_1_a_N"],
        # MLP performance
        "mlp_f1": mlp["test_metrics_optimal"]["f1"],
        "mlp_auc": mlp["test_metrics_optimal"]["auc_roc"],
        "mlp_precision": mlp["test_metrics_optimal"]["precision"],
        "mlp_recall": mlp["test_metrics_optimal"]["recall"],
        "mlp_threshold": mlp["threshold_optimal_f1"],
        "mlp_threshold_youden": mlp["threshold_youden"],
        "mlp_total_params": mlp["hyperparameters"]["total_params"],
        "mlp_epochs": mlp["hyperparameters"]["epochs_actual"],
        "mlp_input_dim": mlp["hyperparameters"]["input_dim"],
        "mlp_training_time": mlp["training_time_seconds"],
        "mlp_cm_optimal": mlp_cm_optimal,
        "mlp_cm_default": mlp_cm_default,
        # Comparison
        "comparison_rows": comparison_rows,
        # XGB feature importances
        "xgb_importances": xgb_importances,
        # Categorical levels for form
        "categorical_levels": CATEGORICAL_LEVELS,
        "actividad_levels": ACTIVIDAD_LEVELS,
        # JSON blob for frontend Plotly
        "data_json": json.dumps({
            "curves": curves,
            "distributions": DISTRIBUTIONS,
            "history": history,
            "comparison": comparison_rows,
            "mlp": {
                "threshold": mlp["threshold_optimal_f1"],
                "cm_optimal": mlp_cm_optimal,
                "cm_default": mlp_cm_default,
            },
            "xgb_importances": xgb_importances,
        }),
    }
    return context


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    context = build_template_context()
    return templates.TemplateResponse(request, "dashboard.html", context)


@app.post("/api/predict")
def api_predict(payload: PredictRequest) -> dict[str, Any]:
    """Inferencia en vivo. Recibe valores RAW (pre-feature-engineering)."""
    row = payload.model_dump()

    # 1. Construir DataFrame con las columnas raw + actividad_fisica
    df_raw = pd.DataFrame([row])

    # 2. Aplicar feature engineering clínico (Tier 1, 2, 3)
    df_eng = add_clinical_features(df_raw)

    # 3. Reordenar columnas según feature_groups (lo que esperan los modelos)
    feature_cols = (
        ARTIFACTS.feature_groups["numeric"]
        + ARTIFACTS.feature_groups["binary"]
        + ARTIFACTS.feature_groups["categorical"]
    )
    X = df_eng[feature_cols]

    # 4. Predicciones
    proba_xgb = float(ARTIFACTS.classical_models["xgboost"].predict_proba(X)[0, 1])
    proba_lr = float(ARTIFACTS.classical_models["logistic_regression"].predict_proba(X)[0, 1])
    proba_rf = float(ARTIFACTS.classical_models["random_forest"].predict_proba(X)[0, 1])
    proba_hgb = float(ARTIFACTS.classical_models["histgradientboosting"].predict_proba(X)[0, 1])

    X_mlp = ARTIFACTS.mlp_preprocessor.transform(X).astype(np.float32)
    proba_mlp = float(ARTIFACTS.mlp.predict(X_mlp, verbose=0).ravel()[0])

    threshold_mlp = ARTIFACTS.mlp_results["threshold_optimal_f1"]
    xgb_record = next(r for r in ARTIFACTS.classical if r["model_key"] == "xgboost")
    threshold_xgb = xgb_record["threshold_optimal_f1"]

    # 5. Banda de riesgo (basada en MLP, el modelo final)
    if proba_mlp < 0.20:
        band = {"label": "Muy bajo", "color": "emerald", "score": 1}
    elif proba_mlp < 0.40:
        band = {"label": "Bajo", "color": "green", "score": 2}
    elif proba_mlp < threshold_mlp:
        band = {"label": "Moderado", "color": "amber", "score": 3}
    elif proba_mlp < 0.85:
        band = {"label": "Elevado", "color": "orange", "score": 4}
    else:
        band = {"label": "Muy elevado", "color": "red", "score": 5}

    # 6. Features clínicas derivadas (devolverlas para mostrarlas en UI)
    derived = {
        "glucosa_alta": int(df_eng["glucosa_alta"].iloc[0]),
        "hemoglobina_baja": int(df_eng["hemoglobina_baja"].iloc[0]),
        "leucocitos_altos": int(df_eng["leucocitos_altos"].iloc[0]),
        "edad_riesgo": int(df_eng["edad_riesgo"].iloc[0]),
        "n_mutaciones_total": int(df_eng["n_mutaciones_total"].iloc[0]),
        "n_mutaciones_alto_riesgo": int(df_eng["n_mutaciones_alto_riesgo"].iloc[0]),
        "actividad_score": int(df_eng["actividad_score"].iloc[0]),
    }

    return {
        "predictions": {
            "mlp": {"proba": proba_mlp, "threshold": threshold_mlp,
                    "positive": proba_mlp >= threshold_mlp},
            "xgboost": {"proba": proba_xgb, "threshold": threshold_xgb,
                        "positive": proba_xgb >= threshold_xgb},
            "logistic_regression": {"proba": proba_lr},
            "random_forest": {"proba": proba_rf},
            "histgradientboosting": {"proba": proba_hgb},
        },
        "band": band,
        "derived_features": derived,
    }


@app.get("/api/distribution/{feature}")
def api_distribution(feature: str) -> JSONResponse:
    if feature not in DISTRIBUTIONS:
        raise HTTPException(status_code=404, detail=f"Feature '{feature}' no encontrada.")
    return JSONResponse(DISTRIBUTIONS[feature])


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return {"status": "ok", "models_loaded": list(ARTIFACTS.classical_models.keys())}
