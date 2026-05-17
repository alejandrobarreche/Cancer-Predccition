"""Entrenamiento de modelos clásicos de ML para predicción de cáncer.

Entrena 4 modelos con búsqueda de hiperparámetros:
1. Logistic Regression (L2)
2. Random Forest
3. XGBoost
4. Gradient Boosting (HistGradientBoosting)

Protocolo:
- Preprocesador ajustado SOLO sobre X_train.
- GridSearchCV con StratifiedKFold(n_splits=5), scoring='f1'.
- Reentrenamiento sobre X_train+X_val con los mejores hiperparámetros.
- Evaluación sobre test UNA SOLA VEZ por modelo.
- Resultados globales en reports/classical/results.json y por-modelo en
  reports/classical/{model}.json.
- Figuras en reports/classical/figures/.
- Modelos serializados en models/.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")  # No display necesario
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
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
    accuracy_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.models.pipelines import (
    build_preprocessor_from_manifest,
    get_feature_names_out,
    load_feature_groups,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "f1"
VAL_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "classical"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "cancer"


# ---------------------------------------------------------------------------
# Utilidades de carga
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Carga train.csv y test.csv, devuelve (X_train_full, y_train_full, X_test, y_test)."""
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    LOGGER.info("Cargando %s", train_path)
    train_df = pd.read_csv(train_path)
    LOGGER.info("Cargando %s", test_path)
    test_df = pd.read_csv(test_path)

    X_train_full = train_df.drop(columns=[TARGET])
    y_train_full = train_df[TARGET].astype(int)
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int)

    LOGGER.info(
        "Train: %d filas | Test: %d filas | Prevalencia train: %.2f%% | Prevalencia test: %.2f%%",
        len(X_train_full),
        len(X_test),
        y_train_full.mean() * 100,
        y_test.mean() * 100,
    )

    return X_train_full, y_train_full, X_test, y_test


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split_name: str = "test",
) -> dict[str, float]:
    """Calcula el conjunto completo de métricas según ml-evaluation-protocol."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "split": split_name,
        "f1": round(float(f1_score(y_true, y_pred)), 6),
        "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "precision": round(float(precision_score(y_true, y_pred)), 6),
        "recall": round(float(recall_score(y_true, y_pred)), 6),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "average_precision": round(float(average_precision_score(y_true, y_prob)), 6),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    LOGGER.info(
        "[%s] F1=%.4f | AUC-ROC=%.4f | Prec=%.4f | Recall=%.4f | Acc=%.4f",
        split_name.upper(),
        metrics["f1"],
        metrics["auc_roc"],
        metrics["precision"],
        metrics["recall"],
        metrics["accuracy"],
    )

    return metrics


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Genera y guarda la matriz de confusión."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No cáncer", "Cáncer"])
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Matriz de Confusión — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Guardada matriz de confusión: %s", output_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Genera y guarda la curva ROC individual."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Aleatoria")
    ax.set_xlabel("Tasa de Falsos Positivos", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=11)
    ax.set_title(f"Curva ROC — {model_name}", fontsize=13, pad=12)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Guardada curva ROC: %s", output_path)


def plot_roc_comparison(
    models_data: dict[str, dict],
    output_path: Path,
) -> None:
    """Genera curva ROC comparativa de todos los modelos."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "tomato", "seagreen", "darkorange", "purple"]

    for i, (model_name, data) in enumerate(models_data.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{model_name} (AUC={roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Aleatoria")
    ax.set_xlabel("Tasa de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=12)
    ax.set_title("Curvas ROC — Comparación Modelos Clásicos", fontsize=13, pad=12)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Guardada curva ROC comparativa: %s", output_path)


def plot_pr_comparison(
    models_data: dict[str, dict],
    output_path: Path,
) -> None:
    """Genera curva Precision-Recall comparativa."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "tomato", "seagreen", "darkorange", "purple"]

    for i, (model_name, data) in enumerate(models_data.items()):
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_prob"])
        ap = average_precision_score(data["y_true"], data["y_prob"])
        color = colors[i % len(colors)]
        ax.plot(recall, precision, color=color, lw=2, label=f"{model_name} (AP={ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precisión", fontsize=12)
    ax.set_title("Curvas Precision-Recall — Comparación Modelos Clásicos", fontsize=13, pad=12)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Guardada curva PR comparativa: %s", output_path)


def plot_metrics_barplot(
    results: list[dict],
    output_path: Path,
) -> None:
    """Barplot comparativo de F1 y AUC-ROC."""
    model_names = [r["model_name"] for r in results]
    f1_scores = [r["test_metrics"]["f1"] for r in results]
    auc_scores = [r["test_metrics"]["auc_roc"] for r in results]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, f1_scores, width, label="F1 (clase positiva)", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, auc_scores, width, label="AUC-ROC", color="tomato", alpha=0.85)

    ax.set_xlabel("Modelo", fontsize=12)
    ax.set_ylabel("Métrica", fontsize=12)
    ax.set_title("Comparación F1 y AUC-ROC — Modelos Clásicos", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    # Etiquetas de valor
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=8,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Guardado barplot de métricas: %s", output_path)


# ---------------------------------------------------------------------------
# Entrenamiento individual
# ---------------------------------------------------------------------------


def train_model(
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
) -> dict[str, Any]:
    """Entrena un modelo con GridSearchCV, reentrena en train+val, evalúa en test.

    Args:
        model_name: Nombre descriptivo del modelo.
        pipeline: Pipeline sklearn con preprocesador + clasificador.
        param_grid: Grid de hiperparámetros para GridSearchCV.
        X_train: Features de entrenamiento (sin validación).
        y_train: Target de entrenamiento.
        X_val: Features de validación.
        y_val: Target de validación.
        X_test: Features de test.
        y_test: Target de test.
        X_train_val: X_train + X_val combinados.
        y_train_val: y_train + y_val combinados.

    Returns:
        Diccionario con resultados en formato ml-evaluation-protocol.
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Entrenando: %s", model_name)
    LOGGER.info("Grid: %s", param_grid)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    t0 = time.time()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        verbose=1,
        refit=True,  # Reentrena sobre X_train con los mejores params
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)
    cv_time = time.time() - t0

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    LOGGER.info("Mejores parámetros: %s", best_params)
    LOGGER.info("Mejor F1 CV: %.4f", best_cv_score)

    # Reentrenar sobre train+val con los mejores hiperparámetros
    LOGGER.info("Reentrenando sobre train+val con los mejores hiperparámetros...")
    best_pipeline = grid_search.best_estimator_
    # Clonar con los mejores params y reentrenar en train+val
    from sklearn.base import clone
    final_pipeline = clone(best_pipeline)
    t1 = time.time()
    final_pipeline.fit(X_train_val, y_train_val)
    retrain_time = time.time() - t1

    total_time = cv_time + retrain_time

    # Métricas en validación (para detectar overfitting)
    y_val_prob = final_pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = final_pipeline.predict(X_val)
    val_metrics = compute_metrics(
        np.array(y_val), np.array(y_val_pred), y_val_prob, split_name="validation"
    )

    # Métricas en test — UNA SOLA VEZ
    LOGGER.info("Evaluando en TEST (una sola vez)...")
    y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = final_pipeline.predict(X_test)
    test_metrics = compute_metrics(
        np.array(y_test), np.array(y_test_pred), y_test_prob, split_name="test"
    )

    # Estructura JSON según ml-evaluation-protocol
    result = {
        "model_name": model_name,
        "best_hyperparameters": {
            k.replace("classifier__", ""): v for k, v in best_params.items()
        },
        "cv_best_f1": round(float(best_cv_score), 6),
        "cv_folds": CV_FOLDS,
        "cv_scoring": SCORING,
        "training_time_seconds": round(total_time, 2),
        "threshold": 0.5,
        "threshold_selection": "default",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        # Para plots
        "_y_test": y_test,
        "_y_test_pred": y_test_pred,
        "_y_test_prob": y_test_prob,
        "_pipeline": final_pipeline,
    }

    return result


# ---------------------------------------------------------------------------
# Importancias de features
# ---------------------------------------------------------------------------


def extract_feature_importances(
    pipeline: Pipeline,
    feature_names: list[str],
    model_name: str,
    top_n: int = 20,
) -> list[dict]:
    """Extrae y ordena las importancias de features del clasificador del pipeline.

    Args:
        pipeline: Pipeline ajustado.
        feature_names: Nombres de las features transformadas.
        model_name: Nombre del modelo (para logging).
        top_n: Número de features a reportar.

    Returns:
        Lista de dicts {feature, importance} ordenada desc.
    """
    clf = pipeline.named_steps["classifier"]
    importances: np.ndarray | None = None

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])

    if importances is None:
        LOGGER.warning("Modelo %s no tiene importancias de features.", model_name)
        return []

    if len(importances) != len(feature_names):
        LOGGER.warning(
            "Desajuste: %d importancias vs %d nombres. Usando índices.",
            len(importances),
            len(feature_names),
        )
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    sorted_idx = np.argsort(importances)[::-1][:top_n]
    result = [
        {"feature": feature_names[i], "importance": round(float(importances[i]), 6)}
        for i in sorted_idx
    ]

    LOGGER.info("Top 5 features (%s): %s", model_name, result[:5])
    return result


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def main() -> None:
    """Función principal: carga datos, entrena modelos, guarda resultados."""

    # Crear directorios si no existen
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    X_train_full, y_train_full, X_test, y_test = load_data()

    # Split train/val (80/20 estratificado)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )

    LOGGER.info(
        "Split: X_train=%d | X_val=%d | X_test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # Cargar grupos de features
    groups = load_feature_groups()
    numeric_cols = groups["numeric"]
    binary_cols = groups["binary"]
    categorical_cols = groups["categorical"]

    # Calcular scale_pos_weight para XGBoost
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    LOGGER.info("scale_pos_weight para XGBoost: %.2f (neg=%d, pos=%d)", scale_pos_weight, n_neg, n_pos)

    # -----------------------------------------------------------------------
    # Definición de modelos y grids
    # -----------------------------------------------------------------------

    def make_pipeline(classifier) -> Pipeline:
        preprocessor = build_preprocessor_from_manifest()
        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

    model_configs = [
        # 1. Logistic Regression con regularización L2
        {
            "name": "Logistic Regression",
            "pipeline": make_pipeline(
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    solver="saga",
                    n_jobs=-1,
                )
            ),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__penalty": ["l1", "l2"],
            },
        },
        # 2. Random Forest
        {
            "name": "Random Forest",
            "pipeline": make_pipeline(
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
            "param_grid": {
                "classifier__n_estimators": [300, 500],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_leaf": [1, 5, 20],
            },
        },
        # 3. XGBoost
        {
            "name": "XGBoost",
            "pipeline": make_pipeline(
                xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1,
                    verbosity=0,
                )
            ),
            "param_grid": {
                "classifier__n_estimators": [300, 500],
                "classifier__max_depth": [4, 6, 8],
                "classifier__learning_rate": [0.05, 0.1],
            },
        },
        # 4. HistGradientBoosting (equivalente a LightGBM en sklearn)
        {
            "name": "HistGradientBoosting",
            "pipeline": make_pipeline(
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    early_stopping=False,
                )
            ),
            "param_grid": {
                "classifier__max_iter": [200, 400],
                "classifier__max_depth": [4, 6, None],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__min_samples_leaf": [20, 50],
            },
        },
    ]

    # -----------------------------------------------------------------------
    # Entrenamiento
    # -----------------------------------------------------------------------

    all_results = []
    comparison_data: dict[str, dict] = {}

    for config in model_configs:
        model_name = config["name"]
        result = train_model(
            model_name=model_name,
            pipeline=config["pipeline"],
            param_grid=config["param_grid"],
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_train_val=X_train_full,  # train completo (train + val originales)
            y_train_val=y_train_full,
        )

        # Generar figuras individuales
        safe_name = model_name.lower().replace(" ", "_")

        plot_confusion_matrix(
            y_true=np.array(result["_y_test"]),
            y_pred=result["_y_test_pred"],
            model_name=model_name,
            output_path=FIGURES_DIR / f"{safe_name}_cm.png",
        )

        plot_roc_curve(
            y_true=np.array(result["_y_test"]),
            y_prob=result["_y_test_prob"],
            model_name=model_name,
            output_path=FIGURES_DIR / f"{safe_name}_roc.png",
        )

        # Serializar modelo
        pipeline_path = MODELS_DIR / f"{safe_name}.pkl"
        joblib.dump(result["_pipeline"], pipeline_path)
        LOGGER.info("Modelo guardado: %s", pipeline_path)

        # Guardar también JSON individual
        model_key = safe_name
        individual_result_path = REPORTS_DIR / f"{safe_name}.json"

        # Obtener importancias de features
        final_pipeline = result["_pipeline"]
        preprocessor = final_pipeline.named_steps["preprocessor"]
        feature_names = get_feature_names_out(
            preprocessor, numeric_cols, binary_cols, categorical_cols
        )
        feature_importances = extract_feature_importances(
            final_pipeline, feature_names, model_name
        )

        # Construir resultado limpio (sin objetos numpy/sklearn)
        clean_result = {
            "model_name": model_name,
            "best_hyperparameters": result["best_hyperparameters"],
            "cv_best_f1": result["cv_best_f1"],
            "cv_folds": result["cv_folds"],
            "cv_scoring": result["cv_scoring"],
            "training_time_seconds": result["training_time_seconds"],
            "threshold": result["threshold"],
            "threshold_selection": result["threshold_selection"],
            "val_metrics": {
                k: v for k, v in result["val_metrics"].items() if k != "split"
            },
            "test_metrics": {
                k: v for k, v in result["test_metrics"].items() if k != "split"
            },
            "feature_importances_top20": feature_importances,
        }

        with open(individual_result_path, "w", encoding="utf-8") as f:
            json.dump(clean_result, f, indent=2, ensure_ascii=False)

        # Guardar datos para comparaciones
        comparison_data[model_name] = {
            "y_true": np.array(result["_y_test"]),
            "y_prob": result["_y_test_prob"],
        }

        # Añadir a lista con resultado limpio
        all_results.append({
            **clean_result,
            "model_key": safe_name,
        })

    # -----------------------------------------------------------------------
    # Figuras comparativas
    # -----------------------------------------------------------------------

    plot_roc_comparison(
        models_data=comparison_data,
        output_path=FIGURES_DIR / "comparison_roc.png",
    )

    plot_pr_comparison(
        models_data=comparison_data,
        output_path=FIGURES_DIR / "comparison_pr.png",
    )

    plot_metrics_barplot(
        results=all_results,
        output_path=FIGURES_DIR / "comparison_metrics.png",
    )

    # -----------------------------------------------------------------------
    # JSON global
    # -----------------------------------------------------------------------

    global_results_path = REPORTS_DIR / "results.json"
    with open(global_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    LOGGER.info("Resultados globales guardados: %s", global_results_path)

    # -----------------------------------------------------------------------
    # Resumen final
    # -----------------------------------------------------------------------

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("RESUMEN FINAL — MODELOS CLÁSICOS")
    LOGGER.info("=" * 70)

    sorted_results = sorted(all_results, key=lambda r: r["test_metrics"]["f1"], reverse=True)

    header = f"{'Modelo':<25} {'F1 test':>9} {'AUC-ROC':>9} {'Precision':>10} {'Recall':>8} {'Accuracy':>9}"
    LOGGER.info(header)
    LOGGER.info("-" * 75)

    for r in sorted_results:
        tm = r["test_metrics"]
        LOGGER.info(
            "%-25s %9.4f %9.4f %10.4f %8.4f %9.4f",
            r["model_name"],
            tm["f1"],
            tm["auc_roc"],
            tm["precision"],
            tm["recall"],
            tm["accuracy"],
        )

    best = sorted_results[0]
    LOGGER.info("\nMejor modelo por F1: %s (F1=%.4f)", best["model_name"], best["test_metrics"]["f1"])


if __name__ == "__main__":
    main()
