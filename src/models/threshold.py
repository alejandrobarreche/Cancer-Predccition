"""Ajuste anti-leakage del umbral de decisión para clasificación binaria.

Implementa el skill `threshold-tuning` del proyecto:
- El barrido se ejecuta SIEMPRE sobre el conjunto de validación, nunca sobre test.
- El umbral óptimo se elige antes de evaluar test (regla no negociable del enunciado).
- Aplicable tanto a la MLP como a los modelos clásicos.

Sin dependencias de TensorFlow — solo numpy + scikit-learn — para que pueda
importarse desde scripts ligeros y tests sin pagar el coste de TF.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve

LOGGER = logging.getLogger(__name__)


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
