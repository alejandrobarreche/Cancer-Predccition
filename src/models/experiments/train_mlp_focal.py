"""Experimento: MLP con focal loss para intentar mover F1 sobre el baseline BCE.

Motivación: el paso 2 de CONTINUAR.md sugiere probar focal loss o recalibración
isotónica. La calibración isotónica es monótona, así que F1 al umbral óptimo
(que depende solo del orden de las probas) es invariante bajo ella. Solo focal
loss puede cambiar el orden y, por tanto, el F1 alcanzable.

Diseño:
- Arquitectura, preprocesado, split y callbacks IDÉNTICOS al baseline
  (src/models/train_mlp.py) para aislar el efecto de la loss.
- Se reemplaza binary_crossentropy por tf.keras.losses.BinaryFocalCrossentropy
  con alpha calibrada a la prevalencia (sustituye al class_weight) y γ variable.
- Se barrren γ ∈ {1.0, 2.0, 3.0}.
- Threshold tuning SOLO sobre validación; evaluación en test UNA VEZ por modelo.
- Resultados a reports/mlp/experiments/focal/ — el baseline en reports/mlp/
  NO se sobrescribe.

Ejecutar:
    conda activate uax-tf
    python -m src.models.experiments.train_mlp_focal
"""

from __future__ import annotations

# TF debe importarse antes que sklearn/pandas/joblib (deadlock en macOS ARM)
import os
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

import numpy as np  # noqa: E402

np.random.seed(RANDOM_SEED)

import tensorflow as tf  # noqa: E402

tf.keras.utils.set_random_seed(RANDOM_SEED)

import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from src.models.mlp import build_model  # noqa: E402
from src.models.pipelines import (  # noqa: E402
    build_preprocessor_from_manifest,
    load_feature_groups,
)
from src.models.threshold import tune_threshold  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "reports" / "mlp" / "experiments" / "focal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUT_DIR / "train.log", mode="w"),
    ],
)
LOGGER = logging.getLogger(__name__)

TARGET_COL = "cancer"
BATCH_SIZE = 256
MAX_EPOCHS = 100
HIDDEN_UNITS = (300, 100, 30)
DROPOUT_RATES = (0.25, 0.25, 0.20)
LEARNING_RATE = 1e-3
GAMMAS = (1.0, 2.0, 3.0)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
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


def train_with_focal(
    gamma: float,
    alpha: float,
    X_train_t: np.ndarray,
    y_train: np.ndarray,
    X_val_t: np.ndarray,
    y_val: np.ndarray,
    X_test_t: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
) -> dict:
    """Entrena una MLP con BinaryFocalCrossentropy(gamma, alpha) y evalúa."""
    LOGGER.info("=" * 60)
    LOGGER.info("Focal loss — γ=%.1f, α=%.4f", gamma, alpha)
    LOGGER.info("=" * 60)

    # Reset semilla por run para reproducibilidad apples-to-apples entre γ
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    model = build_model(
        input_dim=input_dim,
        hidden_units=HIDDEN_UNITS,
        dropout_rates=DROPOUT_RATES,
        learning_rate=LEARNING_RATE,
    )
    # Recompilar con focal loss (sustituye BCE y al class_weight: α dentro de la loss)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=alpha,
            gamma=gamma,
            from_logits=False,
        ),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=10,
            restore_best_weights=True, verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_train_t,
        y_train.astype(np.float32),
        validation_data=(X_val_t, y_val.astype(np.float32)),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    training_time = round(time.time() - t0, 2)
    n_epochs = len(history.history["loss"])

    y_val_proba = model.predict(X_val_t, batch_size=BATCH_SIZE, verbose=0).ravel()
    sweep = tune_threshold(y_val, y_val_proba)
    t_star = sweep["threshold_f1"]
    t_youden = sweep["threshold_youden"]

    val_default = compute_metrics(y_val, y_val_proba, 0.5)
    val_optimal = compute_metrics(y_val, y_val_proba, t_star)

    y_test_proba = model.predict(X_test_t, batch_size=BATCH_SIZE, verbose=0).ravel()
    test_default = compute_metrics(y_test, y_test_proba, 0.5)
    test_optimal = compute_metrics(y_test, y_test_proba, t_star)

    LOGGER.info(
        "γ=%.1f → épocas=%d (%.1fs) | val F1(t*)=%.4f | test F1(t*=%.2f)=%.4f, AUC=%.4f, Prec=%.4f, Rec=%.4f",
        gamma, n_epochs, training_time,
        val_optimal["f1"],
        t_star, test_optimal["f1"], test_optimal["auc_roc"],
        test_optimal["precision"], test_optimal["recall"],
    )

    return {
        "gamma": gamma,
        "alpha": alpha,
        "epochs_actual": n_epochs,
        "training_time_seconds": training_time,
        "threshold_optimal_f1": t_star,
        "threshold_youden": round(t_youden, 4),
        "val_metrics_default": val_default,
        "val_metrics_optimal": val_optimal,
        "test_metrics_default": test_default,
        "test_metrics_optimal": test_optimal,
    }


def main() -> None:
    LOGGER.info("Experimento focal loss — MLP")
    LOGGER.info("Salida: %s", OUT_DIR)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    groups = load_feature_groups()
    feature_cols = groups["numeric"] + groups["binary"] + groups["categorical"]

    X_train_full = train_df[feature_cols]
    y_train_full = train_df[TARGET_COL].values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.20, stratify=y_train_full, random_state=RANDOM_SEED,
    )
    LOGGER.info(
        "Train: %d | Val: %d | Test: %d | Prevalencia train: %.4f",
        len(X_train), len(X_val), len(X_test), y_train.mean(),
    )

    preprocessor = build_preprocessor_from_manifest()
    X_train_t = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_t = preprocessor.transform(X_val).astype(np.float32)
    X_test_t = preprocessor.transform(X_test).astype(np.float32)
    input_dim = X_train_t.shape[1]
    LOGGER.info("input_dim post-preprocesado: %d", input_dim)

    # α de focal loss = peso de la clase positiva. Para reproducir
    # class_weight="balanced", α = 1 - prevalencia_positiva.
    prevalence = float(y_train.mean())
    alpha = 1.0 - prevalence
    LOGGER.info("α (peso clase positiva) = 1 − prev = %.4f", alpha)

    runs = []
    for gamma in GAMMAS:
        result = train_with_focal(
            gamma=gamma, alpha=alpha,
            X_train_t=X_train_t, y_train=y_train,
            X_val_t=X_val_t, y_val=y_val,
            X_test_t=X_test_t, y_test=y_test,
            input_dim=input_dim,
        )
        runs.append(result)

    # Resumen
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("RESUMEN — Focal loss vs baseline BCE")
    LOGGER.info("=" * 70)

    baseline_path = PROJECT_ROOT / "reports" / "mlp" / "results.json"
    baseline = json.loads(baseline_path.read_text())
    base_opt = baseline["test_metrics_optimal"]
    LOGGER.info(
        "Baseline BCE: t*=%.2f | F1=%.4f | AUC=%.4f | Prec=%.4f | Rec=%.4f",
        baseline["threshold_optimal_f1"], base_opt["f1"], base_opt["auc_roc"],
        base_opt["precision"], base_opt["recall"],
    )
    LOGGER.info("-" * 70)
    for r in runs:
        opt = r["test_metrics_optimal"]
        delta_f1 = opt["f1"] - base_opt["f1"]
        delta_auc = opt["auc_roc"] - base_opt["auc_roc"]
        LOGGER.info(
            "γ=%.1f       : t*=%.2f | F1=%.4f (%+.4f) | AUC=%.4f (%+.4f) | Prec=%.4f | Rec=%.4f",
            r["gamma"], r["threshold_optimal_f1"], opt["f1"], delta_f1,
            opt["auc_roc"], delta_auc, opt["precision"], opt["recall"],
        )

    summary = {
        "experiment": "focal_loss_gamma_sweep",
        "alpha": alpha,
        "gammas": list(GAMMAS),
        "baseline_bce": {
            "threshold_optimal_f1": baseline["threshold_optimal_f1"],
            "test_metrics_optimal": base_opt,
        },
        "runs": runs,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    LOGGER.info("Resultados guardados en %s", OUT_DIR / "results.json")


if __name__ == "__main__":
    main()
