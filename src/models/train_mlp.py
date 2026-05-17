"""Entrenamiento, threshold-tuning y evaluación de la MLP de predicción de cáncer.

Protocolo anti-leakage estricto:
1. Carga train.csv y test.csv desde data/processed/.
2. Sub-divide train en sub-train (80%) y validación (20%) estratificados.
3. Ajusta el preprocesador SOLO sobre sub-train; aplica a val y test.
4. Entrena MLP con class_weight, EarlyStopping y ReduceLROnPlateau.
5. Ajusta el umbral de decisión SOLO sobre validación (barrido F1).
6. Evalúa sobre test UNA SOLA VEZ.
7. Persiste modelo, preprocesador y artefactos de reporte.

Ejecutar:
    conda activate uax-tf  # o entorno con TF 2.x + Python <=3.12
    python -m src.models.train_mlp

Artefactos generados:
    models/mlp.keras
    models/mlp_preprocessor.pkl
    reports/mlp/results.json
    reports/mlp/train.log
    reports/mlp/figures/loss_curve.png
    reports/mlp/figures/auc_curve.png
    reports/mlp/figures/threshold_sweep.png
    reports/mlp/figures/roc.png
    reports/mlp/figures/pr.png
    reports/mlp/figures/cm_default.png
    reports/mlp/figures/cm_optimal.png
    reports/comparison/figures/mlp_vs_xgboost.png
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ─── Seeds fijas (antes de importar TF) ───────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

import tensorflow as tf  # noqa: E402 — debe ir después del seed de numpy/random

tf.keras.utils.set_random_seed(RANDOM_SEED)

# ─── Rutas del proyecto ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLP_DIR = REPORTS_DIR / "mlp"
FIGURES_DIR = MLP_DIR / "figures"
COMPARISON_FIGURES_DIR = REPORTS_DIR / "comparison" / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(MLP_DIR / "train.log", mode="w"),
    ],
)
LOGGER = logging.getLogger(__name__)

# ─── Constantes ───────────────────────────────────────────────────────────────
TARGET_COL = "cancer"

NUMERIC_COLS = [
    "glucosa", "colesterol", "trigliceridos", "hemoglobina", "leucocitos",
    "plaquetas", "creatinina", "edad", "num_hijos", "distancia_hospital_km",
]
BINARY_COLS = [
    "diabetes", "hipertension", "obesidad", "enfermedad_cardiaca", "asma",
    "epoc", "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS", "mut_PIK3CA",
    "mut_ALK", "mut_BRAF", "fumador",
]
CATEGORICAL_COLS = [
    "tipo_seguro", "actividad_fisica", "nivel_educativo",
    "nivel_ingresos", "zona", "estado_civil",
]

BATCH_SIZE = 256
MAX_EPOCHS = 100
# Arquitectura ajustada para ~46.913 parámetros con input_dim=39:
# (300, 100, 30) → 46.881 params (error < 0.1% del objetivo)
HIDDEN_UNITS = (300, 100, 30)
DROPOUT_RATES = (0.25, 0.25, 0.20)
LEARNING_RATE = 1e-3

# Baseline de referencia (XGBoost @ threshold=0.5)
XGBOOST_TEST_F1 = 0.556996
XGBOOST_TEST_AUC = 0.844706


# ─── Construcción del preprocesador ───────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: StandardScaler (num), passthrough (bin), OHE (cat)."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLS),
            ("binary", "passthrough", BINARY_COLS),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )


# ─── Arquitectura MLP ─────────────────────────────────────────────────────────

def build_model(input_dim: int) -> tf.keras.Model:
    """MLP densa de 3 capas ocultas con BN + Dropout.

    Arquitectura 300-100-30 ajustada para ~46.881 parámetros con input_dim=39
    (objetivo del enunciado: ~46.913 parámetros).

    La arquitectura de referencia del enunciado es 128-64-32 pensada para
    input_dim≈100+. Con input_dim=39 (30 orig + 9 de OHE), se ajustan las
    unidades a (300, 100, 30) para mantener el orden de magnitud objetivo.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="input")

    # Capa 1: 300 unidades (ajustado para ~46.913 params con input_dim=39)
    x = tf.keras.layers.Dense(300, kernel_initializer="he_normal", name="dense_1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu_1")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_1")(x)

    # Capa 2: 100 unidades
    x = tf.keras.layers.Dense(100, kernel_initializer="he_normal", name="dense_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu_2")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_2")(x)

    # Capa 3: 30 unidades
    x = tf.keras.layers.Dense(30, kernel_initializer="he_normal", name="dense_3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_3")(x)
    x = tf.keras.layers.Activation("relu", name="relu_3")(x)
    x = tf.keras.layers.Dropout(0.20, name="dropout_3")(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="mlp_cancer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    total_params = model.count_params()
    LOGGER.info("Modelo MLP construido — input_dim=%d, params=%d", input_dim, total_params)
    target = 46_913
    ratio = total_params / target
    if not (0.5 <= ratio <= 2.0):
        LOGGER.warning(
            "Parámetros (%d) lejos del objetivo ~46.913 (ratio=%.2f).",
            total_params,
            ratio,
        )

    return model


# ─── Threshold tuning (skill: threshold-tuning) ───────────────────────────────

def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Barrido de umbral sobre conjunto de VALIDACIÓN para maximizar F1.

    REGLA CRÍTICA: Este procedimiento debe ejecutarse SOLO sobre validación,
    NUNCA sobre test. El umbral óptimo se elige antes de ver el test.

    Args:
        y_true: Etiquetas reales (0/1) del conjunto de validación.
        y_proba: Probabilidades predichas por el modelo sobre validación.
        thresholds: Rango de umbrales a evaluar. Default: [0.01, 0.99] paso 0.01.

    Returns:
        Dict con:
        - threshold_f1: umbral que maximiza F1 (criterio principal).
        - threshold_youden: umbral óptimo por índice de Youden J (referencia).
        - f1_at_threshold: F1 alcanzado en threshold_f1.
        - precision_at_threshold: precisión en threshold_f1.
        - recall_at_threshold: recall en threshold_f1.
        - sweep_thresholds: array de umbrales evaluados.
        - sweep_f1: array de F1 por umbral.
        - sweep_precision: array de precision por umbral.
        - sweep_recall: array de recall por umbral.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    f1_scores = []
    prec_scores = []
    rec_scores = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        prec_scores.append(precision_score(y_true, y_pred, zero_division=0))
        rec_scores.append(recall_score(y_true, y_pred, zero_division=0))

    f1_arr = np.array(f1_scores)
    best_idx = int(np.argmax(f1_arr))
    t_star = float(thresholds[best_idx])

    # Youden J = Sensitivity + Specificity - 1 = TPR - FPR
    fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr_arr - fpr_arr
    youden_idx = int(np.argmax(youden_j))
    t_youden = float(roc_thresholds[youden_idx]) if youden_idx < len(roc_thresholds) else 0.5

    LOGGER.info(
        "Threshold óptimo (F1-max sobre val): t*=%.2f → F1=%.4f, Prec=%.4f, Rec=%.4f",
        t_star,
        f1_arr[best_idx],
        prec_scores[best_idx],
        rec_scores[best_idx],
    )
    LOGGER.info("Threshold Youden J (referencia): t_youden=%.2f", t_youden)

    return {
        "threshold_f1": t_star,
        "threshold_youden": t_youden,
        "f1_at_threshold": float(f1_arr[best_idx]),
        "precision_at_threshold": float(prec_scores[best_idx]),
        "recall_at_threshold": float(rec_scores[best_idx]),
        "sweep_thresholds": thresholds.tolist(),
        "sweep_f1": f1_arr.tolist(),
        "sweep_precision": prec_scores,
        "sweep_recall": rec_scores,
    }


# ─── Cálculo de métricas ──────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calcula F1, AUC-ROC, precision, recall, accuracy y average_precision."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "auc_roc": round(float(roc_auc_score(y_true, y_proba)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "accuracy": round(float((y_pred == y_true).mean()), 6),
        "average_precision": round(float(average_precision_score(y_true, y_proba)), 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ─── Figuras ─────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, figures_dir: Path) -> None:
    """Loss y AUC por época (train vs val)."""
    epochs = range(1, len(history["loss"]) + 1)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"], label="Val loss")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Loss por época — MLP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # AUC curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["auc"], label="Train AUC")
    ax.plot(epochs, history["val_auc"], label="Val AUC")
    ax.set_xlabel("Época")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC por época — MLP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "auc_curve.png", dpi=150)
    plt.close(fig)

    LOGGER.info("Curvas de entrenamiento guardadas en %s", figures_dir)


def plot_threshold_sweep(sweep_result: dict, figures_dir: Path) -> None:
    """F1, Precision, Recall vs threshold sobre validación."""
    thresholds = sweep_result["sweep_thresholds"]
    t_star = sweep_result["threshold_f1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, sweep_result["sweep_f1"], label="F1", color="blue")
    ax.plot(thresholds, sweep_result["sweep_precision"], label="Precision", color="green", alpha=0.7)
    ax.plot(thresholds, sweep_result["sweep_recall"], label="Recall", color="orange", alpha=0.7)
    ax.axvline(t_star, color="red", linestyle="--", label=f"t* = {t_star:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Métrica")
    ax.set_title("Barrido de umbral — Validación (MLP)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "threshold_sweep.png", dpi=150)
    plt.close(fig)

    LOGGER.info("Threshold sweep guardado.")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, figures_dir: Path) -> None:
    """Curva ROC sobre test."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"MLP (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC — Test")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc.png", dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, figures_dir: Path) -> None:
    """Curva Precision-Recall sobre test."""
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="blue", lw=2, label=f"MLP (AP = {ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall — Test")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "pr.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Matriz de confusión normalizada + absoluta."""
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, normalize in zip(axes, [None, "true"]):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm if normalize is None else confusion_matrix(y_true, y_pred, normalize="true"),
            display_labels=["No cáncer", "Cáncer"],
        )
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"{title} — {'Absoluta' if normalize is None else 'Normalizada'}")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_models_comparison(
    mlp_f1: float,
    mlp_auc: float,
    xgb_f1: float,
    xgb_auc: float,
    figures_dir: Path,
) -> None:
    """Barplot comparativo MLP vs XGBoost (F1 y AUC-ROC en test)."""
    models = ["XGBoost", "MLP"]
    f1_vals = [xgb_f1, mlp_f1]
    auc_vals = [xgb_auc, mlp_auc]

    x = np.arange(len(models))
    width = 0.30

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, f1_vals, width, label="F1 (clase positiva)", color=["steelblue", "darkorange"])
    bars2 = ax.bar(x + width / 2, auc_vals, width, label="AUC-ROC", color=["steelblue", "darkorange"], alpha=0.6)

    # Etiquetas
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparación MLP vs XGBoost — Test (threshold óptimo vs 0.5)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "mlp_vs_xgboost.png", dpi=150)
    plt.close(fig)

    LOGGER.info("Comparación de modelos guardada.")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> None:
    LOGGER.info("=" * 60)
    LOGGER.info("Inicio entrenamiento MLP — Predicción de cáncer")
    LOGGER.info("=" * 60)

    # 1. Carga de datos
    LOGGER.info("Cargando datos...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    LOGGER.info("Train shape: %s | Test shape: %s", train_df.shape, test_df.shape)
    LOGGER.info(
        "Prevalencia cancer=1 — Train: %.4f | Test: %.4f",
        train_df[TARGET_COL].mean(),
        test_df[TARGET_COL].mean(),
    )

    feature_cols = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS

    X_train_full = train_df[feature_cols]
    y_train_full = train_df[TARGET_COL].values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values

    # 2. Split sub-train / validación (80/20 estratificado)
    LOGGER.info("Subdividiendo train en sub-train/val (80/20, stratify=y, seed=42)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        stratify=y_train_full,
        random_state=RANDOM_SEED,
    )
    LOGGER.info(
        "Sub-train: %d muestras | Val: %d muestras",
        len(X_train),
        len(X_val),
    )

    # 3. Preprocesado — fit SOLO en sub-train
    LOGGER.info("Ajustando preprocesador sobre sub-train...")
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_t = preprocessor.transform(X_val).astype(np.float32)
    X_test_t = preprocessor.transform(X_test).astype(np.float32)

    input_dim = X_train_t.shape[1]
    LOGGER.info("Dimensión de entrada tras preprocesado: %d", input_dim)

    # Guardar preprocesador
    preprocessor_path = MODELS_DIR / "mlp_preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    LOGGER.info("Preprocesador guardado en %s", preprocessor_path)

    # 4. Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes.tolist(), weights.tolist()))
    LOGGER.info("Class weights: %s", class_weight)

    # 5. Construcción del modelo
    LOGGER.info("Construyendo modelo MLP...")
    model = build_model(input_dim=input_dim)
    model.summary(print_fn=LOGGER.info)

    # Verificación de parámetros
    n_params = model.count_params()
    LOGGER.info("Parámetros totales: %d (objetivo ~46.913)", n_params)

    # 6. Callbacks
    checkpoint_path = str(MODELS_DIR / "mlp_best.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # 7. Entrenamiento
    LOGGER.info("Iniciando entrenamiento (max_epochs=%d, batch_size=%d)...", MAX_EPOCHS, BATCH_SIZE)
    t0 = time.time()

    history = model.fit(
        X_train_t,
        y_train.astype(np.float32),
        validation_data=(X_val_t, y_val.astype(np.float32)),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    training_time = round(time.time() - t0, 2)
    n_epochs = len(history.history["loss"])
    LOGGER.info(
        "Entrenamiento completado en %.1f s — %d épocas (early stopping)",
        training_time,
        n_epochs,
    )

    # Aviso si convergencia sospechosamente rápida o lenta
    if n_epochs < 10:
        LOGGER.warning(
            "Convergencia muy rapida (%d epocas). Verifica la arquitectura y el learning rate.",
            n_epochs,
        )
    if n_epochs >= MAX_EPOCHS:
        LOGGER.warning(
            "El modelo no convergio en %d epocas. Considera aumentar MAX_EPOCHS o ajustar LR.",
            MAX_EPOCHS,
        )

    # 8. Predicciones sobre validación
    LOGGER.info("Prediciendo sobre validación para threshold tuning...")
    y_val_proba = model.predict(X_val_t, batch_size=BATCH_SIZE, verbose=0).ravel()

    # 9. Threshold tuning (SOLO sobre validación)
    LOGGER.info("Ejecutando barrido de umbral sobre validación (skill: threshold-tuning)...")
    sweep_result = tune_threshold(y_val, y_val_proba)
    t_star = sweep_result["threshold_f1"]
    t_youden = sweep_result["threshold_youden"]

    # Métricas en val (threshold=0.5 y threshold óptimo)
    val_metrics_default = compute_metrics(y_val, y_val_proba, threshold=0.5)
    val_metrics_optimal = compute_metrics(y_val, y_val_proba, threshold=t_star)

    LOGGER.info(
        "Val metrics — default (t=0.5): F1=%.4f, AUC=%.4f",
        val_metrics_default["f1"],
        val_metrics_default["auc_roc"],
    )
    LOGGER.info(
        "Val metrics — optimal (t=%.2f): F1=%.4f, AUC=%.4f",
        t_star,
        val_metrics_optimal["f1"],
        val_metrics_optimal["auc_roc"],
    )

    # 10. Evaluación final sobre TEST (una sola vez)
    LOGGER.info("=" * 60)
    LOGGER.info("EVALUACIÓN FINAL SOBRE TEST (única vez, threshold fijado desde val)")
    LOGGER.info("=" * 60)
    y_test_proba = model.predict(X_test_t, batch_size=BATCH_SIZE, verbose=0).ravel()

    test_metrics_default = compute_metrics(y_test, y_test_proba, threshold=0.5)
    test_metrics_optimal = compute_metrics(y_test, y_test_proba, threshold=t_star)

    LOGGER.info(
        "Test metrics — default (t=0.5): F1=%.4f, AUC=%.4f, Prec=%.4f, Rec=%.4f",
        test_metrics_default["f1"],
        test_metrics_default["auc_roc"],
        test_metrics_default["precision"],
        test_metrics_default["recall"],
    )
    LOGGER.info(
        "Test metrics — optimal (t=%.2f): F1=%.4f, AUC=%.4f, Prec=%.4f, Rec=%.4f",
        t_star,
        test_metrics_optimal["f1"],
        test_metrics_optimal["auc_roc"],
        test_metrics_optimal["precision"],
        test_metrics_optimal["recall"],
    )

    # Aviso si val y test divergen más de 5 puntos en F1
    f1_gap = abs(val_metrics_optimal["f1"] - test_metrics_optimal["f1"])
    if f1_gap > 0.05:
        LOGGER.warning(
            "DIVERGENCIA entre val y test en F1: %.4f puntos. "
            "Posible sobreajuste o distribución distinta.",
            f1_gap,
        )

    # 11. Figuras
    LOGGER.info("Generando figuras...")
    plot_training_curves(history.history, FIGURES_DIR)
    plot_threshold_sweep(sweep_result, FIGURES_DIR)
    plot_roc_curve(y_test, y_test_proba, FIGURES_DIR)
    plot_pr_curve(y_test, y_test_proba, FIGURES_DIR)

    y_test_pred_default = (y_test_proba >= 0.5).astype(int)
    y_test_pred_optimal = (y_test_proba >= t_star).astype(int)
    plot_confusion_matrix(y_test, y_test_pred_default, "MLP (t=0.5)", FIGURES_DIR / "cm_default.png")
    plot_confusion_matrix(y_test, y_test_pred_optimal, f"MLP (t={t_star:.2f})", FIGURES_DIR / "cm_optimal.png")

    plot_models_comparison(
        mlp_f1=test_metrics_optimal["f1"],
        mlp_auc=test_metrics_optimal["auc_roc"],
        xgb_f1=XGBOOST_TEST_F1,
        xgb_auc=XGBOOST_TEST_AUC,
        figures_dir=COMPARISON_FIGURES_DIR,
    )

    # 12. Guardar modelo
    model_path = MODELS_DIR / "mlp.keras"
    model.save(str(model_path))
    LOGGER.info("Modelo guardado en %s", model_path)

    # 13. JSON de resultados (ml-evaluation-protocol)
    # Serializar history para JSON (convertir float32 a float)
    history_serializable = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }

    results = {
        "model_name": "MLP (TensorFlow/Keras)",
        "model_key": "mlp",
        "hyperparameters": {
            "architecture": "Dense(300) → BN → ReLU → Dropout(0.25) → "
                            "Dense(100) → BN → ReLU → Dropout(0.25) → "
                            "Dense(30) → BN → ReLU → Dropout(0.20) → Dense(1, sigmoid)",
            "hidden_units": list(HIDDEN_UNITS),
            "dropout_rates": list(DROPOUT_RATES),
            "optimizer": "Adam",
            "learning_rate_initial": LEARNING_RATE,
            "loss": "binary_crossentropy",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "epochs_actual": n_epochs,
            "early_stopping_monitor": "val_auc",
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
            "class_weight": {str(k): round(v, 4) for k, v in class_weight.items()},
            "kernel_initializer": "he_normal",
            "total_params": n_params,
            "input_dim": input_dim,
            "note": (
                f"Arquitectura ajustada a (300,100,30) para input_dim={input_dim} "
                f"(tras OHE drop=first sobre 6 categorías). "
                f"Parámetros resultantes: {n_params} (objetivo enunciado: ~46.913). "
                "La arquitectura de referencia del enunciado (128-64-32) asume input_dim>100; "
                "con input_dim=39 se re-escalan las unidades manteniendo la proporción y el orden de magnitud."
            ),
        },
        "training_time_seconds": training_time,
        "threshold_selection": "validation_f1_max",
        "threshold_default": 0.5,
        "threshold_optimal_f1": t_star,
        "threshold_youden": round(t_youden, 4),
        "val_metrics_default": val_metrics_default,
        "val_metrics_optimal": val_metrics_optimal,
        "test_metrics_default": test_metrics_default,
        "test_metrics_optimal": test_metrics_optimal,
        "baseline_xgboost_test": {
            "f1": XGBOOST_TEST_F1,
            "auc_roc": XGBOOST_TEST_AUC,
            "threshold": 0.5,
        },
        "training_history": history_serializable,
    }

    results_path = MLP_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info("Resultados guardados en %s", results_path)

    # 14. Informe final
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("RESUMEN FINAL")
    LOGGER.info("=" * 60)
    LOGGER.info("%-20s %10s %10s", "Métrica", "Val (opt)", "Test (opt)")
    LOGGER.info("%-20s %10.4f %10.4f", "F1", val_metrics_optimal["f1"], test_metrics_optimal["f1"])
    LOGGER.info("%-20s %10.4f %10.4f", "AUC-ROC", val_metrics_optimal["auc_roc"], test_metrics_optimal["auc_roc"])
    LOGGER.info("%-20s %10.4f %10.4f", "Precision", val_metrics_optimal["precision"], test_metrics_optimal["precision"])
    LOGGER.info("%-20s %10.4f %10.4f", "Recall", val_metrics_optimal["recall"], test_metrics_optimal["recall"])
    LOGGER.info("Threshold óptimo (val, F1-max): t* = %.2f", t_star)
    LOGGER.info("Threshold Youden J (ref):       t  = %.2f", t_youden)
    LOGGER.info("Parámetros del modelo: %d", n_params)
    LOGGER.info("Épocas hasta convergencia: %d (early stopping)", n_epochs)
    LOGGER.info("")
    LOGGER.info("XGBoost test F1 (t=0.5): %.4f → MLP test F1 (t=%.2f): %.4f", XGBOOST_TEST_F1, t_star, test_metrics_optimal["f1"])
    delta_f1 = test_metrics_optimal["f1"] - XGBOOST_TEST_F1
    delta_auc = test_metrics_optimal["auc_roc"] - XGBOOST_TEST_AUC
    LOGGER.info("Delta F1: %+.4f | Delta AUC: %+.4f", delta_f1, delta_auc)

    print("\n" + "=" * 60)
    print("MLP entrenada — Resultados:")
    print(f"{'Métrica':<15} {'Validación':>12} {'Test':>12}")
    print("-" * 42)
    print(f"{'F1':<15} {val_metrics_optimal['f1']:>12.4f} {test_metrics_optimal['f1']:>12.4f}")
    print(f"{'AUC-ROC':<15} {val_metrics_optimal['auc_roc']:>12.4f} {test_metrics_optimal['auc_roc']:>12.4f}")
    print(f"{'Precision':<15} {val_metrics_optimal['precision']:>12.4f} {test_metrics_optimal['precision']:>12.4f}")
    print(f"{'Recall':<15} {val_metrics_optimal['recall']:>12.4f} {test_metrics_optimal['recall']:>12.4f}")
    print(f"\nUmbral seleccionado (validación, F1-óptimo): t* = {t_star:.2f}")
    print(f"Umbral Youden J (referencia): {t_youden:.2f}")
    print(f"Parámetros del modelo: {n_params:,}")
    print(f"Épocas hasta convergencia: {n_epochs} (early stopping)")
    print(f"Tiempo de entrenamiento: {training_time:.1f} s")
    print(f"\nComparación vs XGBoost (test):")
    print(f"  F1:     XGB={XGBOOST_TEST_F1:.4f}  MLP={test_metrics_optimal['f1']:.4f}  ({delta_f1:+.4f})")
    print(f"  AUC:    XGB={XGBOOST_TEST_AUC:.4f}  MLP={test_metrics_optimal['auc_roc']:.4f}  ({delta_auc:+.4f})")
    print(f"\nArtefactos:")
    print(f"  reports/mlp/results.json")
    print(f"  reports/mlp/train.log")
    print(f"  reports/mlp/figures/loss_curve.png")
    print(f"  reports/mlp/figures/auc_curve.png")
    print(f"  reports/mlp/figures/threshold_sweep.png")
    print(f"  reports/mlp/figures/roc.png")
    print(f"  reports/mlp/figures/pr.png")
    print(f"  reports/mlp/figures/cm_default.png")
    print(f"  reports/mlp/figures/cm_optimal.png")
    print(f"  reports/comparison/figures/mlp_vs_xgboost.png")
    print(f"  models/mlp.keras")
    print(f"  models/mlp_preprocessor.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
