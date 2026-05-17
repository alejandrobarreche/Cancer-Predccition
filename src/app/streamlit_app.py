"""Aplicación Streamlit — Estudio de predicción de diagnóstico de cáncer (UAX).

Presenta los resultados del estudio completo: cohorte, análisis exploratorio,
modelos clásicos de referencia, red neuronal multicapa y comparativa final.
Incluye un estimador interactivo de riesgo individual con propósito académico.

Ejecutar desde la raíz del proyecto:
    conda activate uax-tf
    streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"

NUMERIC_RANGES: dict[str, dict[str, float | str]] = {
    "glucosa": {"min": 55.0, "max": 180.0, "median": 101.66, "unidad": "mg/dL"},
    "colesterol": {"min": 120.0, "max": 320.0, "median": 193.40, "unidad": "mg/dL"},
    "trigliceridos": {"min": 50.0, "max": 322.0, "median": 155.53, "unidad": "mg/dL"},
    "hemoglobina": {"min": 8.0, "max": 18.0, "median": 13.93, "unidad": "g/dL"},
    "leucocitos": {"min": 2.0, "max": 16.0, "median": 7.14, "unidad": "x10^3/uL"},
    "plaquetas": {"min": 100.0, "max": 480.0, "median": 255.0, "unidad": "x10^3/uL"},
    "creatinina": {"min": 0.35, "max": 2.10, "median": 1.00, "unidad": "mg/dL"},
    "edad": {"min": 20.0, "max": 90.0, "median": 54.0, "unidad": "años"},
    "num_hijos": {"min": 0.0, "max": 8.0, "median": 1.0, "unidad": ""},
    "distancia_hospital_km": {"min": 0.5, "max": 250.0, "median": 17.10, "unidad": "km"},
}

BINARY_LABELS: dict[str, str] = {
    "diabetes": "Diabetes mellitus",
    "hipertension": "Hipertensión arterial",
    "obesidad": "Obesidad",
    "enfermedad_cardiaca": "Enfermedad cardíaca",
    "asma": "Asma",
    "epoc": "EPOC",
    "mut_BRCA1": "Mutación BRCA1",
    "mut_TP53": "Mutación TP53",
    "mut_EGFR": "Mutación EGFR",
    "mut_KRAS": "Mutación KRAS",
    "mut_PIK3CA": "Mutación PIK3CA",
    "mut_ALK": "Mutación ALK",
    "mut_BRAF": "Mutación BRAF",
    "fumador": "Hábito tabáquico activo",
}

CATEGORICAL_LEVELS: dict[str, list[str]] = {
    "tipo_seguro": ["Mixto", "Privado", "Publico"],
    "actividad_fisica": ["Alta", "Baja", "Moderada"],
    "nivel_educativo": ["Primaria", "Secundaria", "Sin estudios", "Universitario"],
    "nivel_ingresos": ["Alto", "Bajo", "Medio", "Muy bajo"],
    "zona": ["Rural", "Semiurbana", "Urbana"],
    "estado_civil": ["Casado", "Divorciado", "Soltero", "Viudo"],
}


# ----------------------------------------------------------------------------
# Carga de datos cacheada
# ----------------------------------------------------------------------------


@st.cache_data
def load_eda() -> dict[str, Any]:
    return json.loads((REPORTS / "eda_report.json").read_text())


@st.cache_data
def load_classical() -> list[dict[str, Any]]:
    return json.loads((REPORTS / "classical_results.json").read_text())


@st.cache_data
def load_mlp() -> dict[str, Any]:
    return json.loads((REPORTS / "mlp_results.json").read_text())


@st.cache_data
def load_eda_memo() -> str:
    path = REPORTS / "eda_memo.md"
    return path.read_text() if path.exists() else ""


@st.cache_data
def load_feature_groups() -> dict[str, Any]:
    return json.loads((PROCESSED / "feature_groups.json").read_text())


@st.cache_resource
def load_xgboost():
    return joblib.load(MODELS / "xgboost.pkl")


@st.cache_resource
def load_mlp_model():
    import tensorflow as tf

    model = tf.keras.models.load_model(MODELS / "mlp.keras")
    preprocessor = joblib.load(MODELS / "mlp_preprocessor.pkl")
    return model, preprocessor


# ----------------------------------------------------------------------------
# Componentes reutilizables
# ----------------------------------------------------------------------------


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def metrics_table(metrics: dict[str, Any], threshold_label: str) -> pd.DataFrame:
    cm = metrics["confusion_matrix"]
    return pd.DataFrame(
        {
            "Indicador": [
                "F1 (clase positiva)",
                "AUC-ROC",
                "Precisión (VPP)",
                "Sensibilidad (recall)",
                "Exactitud",
                "Average Precision",
                "Verdaderos negativos",
                "Falsos positivos",
                "Falsos negativos",
                "Verdaderos positivos",
            ],
            f"Valor ({threshold_label})": [
                f"{metrics['f1']:.4f}",
                f"{metrics['auc_roc']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['accuracy']:.4f}",
                f"{metrics['average_precision']:.4f}",
                f"{cm['tn']:,}",
                f"{cm['fp']:,}",
                f"{cm['fn']:,}",
                f"{cm['tp']:,}",
            ],
        }
    )


def fig_path(name: str) -> Path:
    return FIGURES / name


def show_figure(name: str, caption: str | None = None) -> None:
    p = fig_path(name)
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"Figura no disponible: {name}")


# ----------------------------------------------------------------------------
# Páginas
# ----------------------------------------------------------------------------


def page_resumen() -> None:
    st.title("Estudio de Predicción de Diagnóstico de Cáncer")
    st.caption("Universidad Alfonso X el Sabio · Grado en Ingeniería Matemática · "
               "Inteligencia Artificial · Curso 2025–2026")

    st.markdown(
        """
        ### Resumen del estudio

        Este informe documenta el desarrollo, evaluación y comparación de modelos de aprendizaje
        automático para la **estimación del riesgo de diagnóstico oncológico** sobre una cohorte
        retrospectiva extraída de la base de datos clínica de la Universidad Alfonso X el Sabio
        alojada en Microsoft Azure. El propósito del trabajo es estrictamente académico y los
        resultados aquí mostrados **no constituyen un dispositivo médico** ni una recomendación
        clínica.
        """
    )

    eda = load_eda()
    mlp = load_mlp()
    xgb_test = mlp["baseline_xgboost_test"]

    st.markdown("---")
    st.subheader("Indicadores principales del estudio")
    cols = st.columns(4)
    cols[0].metric("Cohorte total", f"{eda['n_total']:,}", help="Número de pacientes únicos")
    cols[1].metric("Casos positivos", f"{eda['n_positivos']:,}",
                   help=f"Prevalencia: {eda['prevalencia_cancer']:.2%}")
    cols[2].metric("Variables analizadas", "30", help="Tras exclusión de fugas y constantes")
    cols[3].metric("Modelos comparados", "5", help="4 modelos clásicos + 1 red neuronal MLP")

    st.markdown("### Rendimiento diagnóstico final (conjunto de prueba)")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Modelo de referencia — XGBoost**")
        st.metric("F1 (clase positiva)", f"{xgb_test['f1']:.4f}")
        st.metric("AUC-ROC", f"{xgb_test['auc_roc']:.4f}")
        st.caption("Punto de corte 0,50")
    with cols[1]:
        st.markdown("**Modelo final — Red neuronal MLP**")
        m = mlp["test_metrics_optimal"]
        st.metric("F1 (clase positiva)", f"{m['f1']:.4f}", delta=f"{m['f1']-xgb_test['f1']:+.4f}")
        st.metric("AUC-ROC", f"{m['auc_roc']:.4f}", delta=f"{m['auc_roc']-xgb_test['auc_roc']:+.4f}")
        st.caption(f"Punto de corte ajustado en validación: {mlp['threshold_optimal_f1']:.2f}")

    st.markdown("---")
    st.subheader("Hallazgos clínicos relevantes")
    st.markdown(
        """
        - La **prevalencia oncológica** observada en la cohorte (19,29 %) genera un desbalance
          moderado de clases (1 : 4,18) que se gestiona mediante ponderación inversa de clases
          (`class_weight`) durante el entrenamiento.
        - Los **predictores con mayor capacidad discriminante** son: hábito tabáquico activo,
          mutación germinal *BRCA1*, mutación *TP53*, obesidad y mutación *KRAS*. Este perfil
          es **biológicamente plausible** y consistente con la literatura oncogenética.
        - Cinco variables económicas y de seguimiento (`coste_total`, `dias_hospital`,
          `coste_farmaco`, `num_ingresos`, `vive`) presentan correlaciones espurias muy elevadas
          con el desenlace (r > 0,64). Su exclusión es **obligatoria** por riesgo de fuga de
          información post-diagnóstico (*data leakage*).
        - El modelo final supera ligeramente al mejor clasificador clásico en F1 (+0,0146)
          tras un ajuste reproducible y anti-fuga del punto de corte sobre el conjunto de
          validación.
        """
    )

    with st.expander("Acerca de este informe"):
        st.markdown(
            """
            La aplicación se construye sobre los artefactos generados por el pipeline
            reproducible del proyecto. Todas las cifras mostradas se leen directamente de los
            ficheros JSON serializados en `reports/` y de las figuras en `reports/figures/`.
            No se reentrena ningún modelo dentro de esta interfaz.

            **Documentación adicional:**
            - `reports/eda_memo.md` — informe técnico del análisis exploratorio.
            - `reports/classical_results.json` — métricas detalladas de los 4 modelos clásicos.
            - `reports/mlp_results.json` — historial de entrenamiento, threshold tuning y métricas
              en validación y prueba de la red neuronal MLP.
            - `reports/slides/cancer_uax.pptx` — presentación ejecutiva en 5 diapositivas.
            """
        )


def page_cohorte() -> None:
    st.title("Cohorte y caracterización descriptiva")
    eda = load_eda()

    st.markdown(
        """
        La cohorte se construye mediante la unión por `paciente_id` de seis tablas relacionales
        provenientes del sistema de información clínica: variables clínicas, bioquímicas,
        genéticas, sociodemográficas, de hábitos generales y económicas. Cada paciente aparece
        de forma única en cada una de las seis fuentes, garantizando una **cobertura del 100 %**
        sin pérdida muestral en la fase de unión.
        """
    )

    st.subheader("Características poblacionales")
    cols = st.columns(3)
    cols[0].metric("Pacientes únicos", f"{eda['n_total']:,}")
    cols[1].metric("Diagnósticos confirmados de cáncer", f"{eda['n_positivos']:,}")
    cols[2].metric("Sin diagnóstico", f"{eda['n_negativos']:,}")
    cols = st.columns(3)
    cols[0].metric("Prevalencia", f"{eda['prevalencia_cancer']:.2%}")
    cols[1].metric("Razón de desbalance", f"1 : {eda['ratio_desbalance_1_a_N']:.2f}")
    cols[2].metric("Valores ausentes", f"{eda['nulos_joined']}",
                   help="Tras la unión de las seis tablas")

    st.subheader("Cobertura por fuente de información")
    fuentes_df = pd.DataFrame(eda["fuentes"]).T.reset_index()
    fuentes_df.columns = ["Fuente", "Filas", "Columnas", "IDs únicos", "Nulos", "Cobertura (%)"]
    st.dataframe(fuentes_df, use_container_width=True, hide_index=True)

    st.subheader("Distribución del desenlace")
    show_figure("fig01_target_distribution.png",
                "Distribución de la variable objetivo en la cohorte completa.")

    st.subheader("Distribuciones bioquímicas condicionadas")
    show_figure("fig02_bioquimicos_dist.png",
                "Distribuciones de las variables bioquímicas estratificadas por desenlace.")

    st.subheader("Variables sociodemográficas continuas")
    show_figure("fig08_sociodem_num.png",
                "Distribuciones de edad, número de hijos y distancia al hospital.")

    with st.expander("Notas metodológicas sobre la unión de fuentes"):
        st.markdown(
            """
            - Tabla principal: `clinicos`, por contener la variable objetivo.
            - Operación: `pd.merge(..., how='left', on='paciente_id')` aplicada de forma iterada
              a las cinco tablas restantes.
            - Resultado: matriz de `(50 001, 38)` con `0` valores ausentes.
            - El conjunto resultante se serializa en `data/interim/joined.csv` para la fase de
              modelado.
            """
        )


def page_seleccion() -> None:
    st.title("Análisis exploratorio y selección de variables")
    eda = load_eda()

    st.markdown(
        """
        El análisis exploratorio persigue tres objetivos: (i) caracterizar la fuerza de
        asociación univariante entre cada variable y el desenlace, (ii) detectar variables
        cuyo valor pueda haberse generado **con posterioridad al diagnóstico** —y por tanto
        contaminen el aprendizaje supervisado por fuga de información—, y (iii) construir una
        propuesta razonada de variables candidatas para los modelos predictivos.
        """
    )

    st.subheader("Variables excluidas por fuga post-diagnóstico")
    st.warning(
        "Estas variables presentan correlaciones inverosímilmente altas con el desenlace "
        "(r > 0,64). Su inclusión generaría métricas artificialmente óptimas que no se "
        "trasladarían a un escenario clínico real."
    )

    leakage = pd.DataFrame(
        [
            {"Variable": k, "Correlación con desenlace": eda["correlaciones_con_cancer"][k]}
            for k in ["coste_total", "dias_hospital", "coste_farmaco", "num_ingresos", "vive"]
        ]
    )
    leakage["Correlación con desenlace"] = leakage["Correlación con desenlace"].apply(lambda x: f"{x:+.4f}")
    st.dataframe(leakage, use_container_width=True, hide_index=True)
    show_figure("fig09_economicos_leakage.png",
                "Correlación de las variables económicas con el desenlace, indicativo de fuga.")

    st.subheader("Variables sin varianza")
    st.markdown(
        "- **`alcohol`**: presenta el valor 1 en el 100 % de los pacientes; correlación de "
        "Pearson indefinida y aporte informativo nulo. Se excluye del modelado."
    )

    st.subheader("Predictores con mayor señal univariante")
    st.markdown("Excluyendo las variables con fuga, las correlaciones lineales más altas son:")

    safe_corr = {
        k: v for k, v in eda["correlaciones_con_cancer"].items()
        if k not in {"coste_total", "dias_hospital", "coste_farmaco", "num_ingresos", "vive", "alcohol"}
        and pd.notna(v)
    }
    safe_df = pd.DataFrame(
        sorted(safe_corr.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12],
        columns=["Variable", "Correlación con desenlace"],
    )
    safe_df["Correlación con desenlace"] = safe_df["Correlación con desenlace"].apply(lambda x: f"{x:+.4f}")
    st.dataframe(safe_df, use_container_width=True, hide_index=True)

    show_figure("fig10_feature_signal.png",
                "Magnitud de señal por variable (correlación absoluta con el desenlace).")

    st.subheader("Asociaciones genéticas")
    show_figure("fig03_geneticos_lift.png",
                "Razón de prevalencia condicionada a la presencia de cada mutación germinal.")

    st.subheader("Asociaciones de comorbilidades clínicas")
    show_figure("fig04_clinicos_lift.png",
                "Razón de prevalencia condicionada a comorbilidades clínicas.")

    st.subheader("Variables categóricas — gradientes observados")
    show_figure("fig05_categorical_lift.png",
                "Prevalencia del desenlace por categoría dentro de cada variable.")

    st.subheader("Estructura de correlaciones del conjunto unido")
    show_figure("fig06_corr_heatmap.png",
                "Mapa de calor de correlaciones de Pearson entre variables candidatas.")

    st.subheader("Conjunto final de variables")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Bioquímicas y antropométricas continuas (10):**")
        for c in ["glucosa", "colesterol", "trigliceridos", "hemoglobina",
                  "leucocitos", "plaquetas", "creatinina", "edad",
                  "num_hijos", "distancia_hospital_km"]:
            st.markdown(f"- `{c}`")
        st.markdown("**Categóricas (6):**")
        for c in CATEGORICAL_LEVELS:
            st.markdown(f"- `{c}` — {len(CATEGORICAL_LEVELS[c])} niveles")
    with cols[1]:
        st.markdown("**Comorbilidades clínicas (6):**")
        for c in ["diabetes", "hipertension", "obesidad", "enfermedad_cardiaca", "asma", "epoc"]:
            st.markdown(f"- `{c}`")
        st.markdown("**Mutaciones germinales (7):**")
        for c in ["mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS",
                  "mut_PIK3CA", "mut_ALK", "mut_BRAF"]:
            st.markdown(f"- `{c}`")
        st.markdown("**Hábitos (1):**")
        st.markdown("- `fumador`")

    st.info("Total: **30 variables** tras el proceso de selección y exclusión.")


def page_clasicos() -> None:
    st.title("Modelos clásicos de referencia")
    classical = load_classical()

    st.markdown(
        """
        Se entrenan cuatro modelos clásicos como línea base de comparación. Todos comparten
        el mismo procedimiento: ajuste del preprocesado **únicamente sobre el conjunto de
        entrenamiento**, validación cruzada estratificada de 5 particiones para selección de
        hiperparámetros optimizando F1 de la clase positiva, y evaluación final sobre el
        conjunto de prueba con punto de corte 0,50. Todos los modelos incorporan ponderación
        inversa de clases (`class_weight='balanced'` o `scale_pos_weight` para XGBoost).
        """
    )

    st.subheader("Resumen comparativo (conjunto de prueba)")
    rows = []
    for m in classical:
        t = m["test_metrics"]
        rows.append({
            "Modelo": m["model_name"],
            "F1": t["f1"],
            "AUC-ROC": t["auc_roc"],
            "Precisión": t["precision"],
            "Sensibilidad": t["recall"],
            "Exactitud": t["accuracy"],
            "Tiempo (s)": m["training_time_seconds"],
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    st.dataframe(
        df.style.format({"F1": "{:.4f}", "AUC-ROC": "{:.4f}",
                         "Precisión": "{:.4f}", "Sensibilidad": "{:.4f}",
                         "Exactitud": "{:.4f}", "Tiempo (s)": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )
    st.success(f"Modelo de referencia seleccionado: **{df.iloc[0]['Modelo']}** "
               f"(F1 = {df.iloc[0]['F1']:.4f}, AUC-ROC = {df.iloc[0]['AUC-ROC']:.4f}).")

    st.subheader("Curvas características receptor-operador")
    show_figure("classical_roc_comparison.png",
                "Curvas ROC comparadas de los cuatro modelos clásicos sobre el conjunto de prueba.")

    st.subheader("Curvas precisión-sensibilidad")
    show_figure("classical_pr_comparison.png",
                "Curvas precisión-recall comparadas sobre el conjunto de prueba.")

    st.subheader("Métricas comparadas en formato gráfico")
    show_figure("classical_metrics_barplot.png",
                "F1 y AUC-ROC por modelo (umbral 0,50).")

    st.markdown("---")
    st.subheader("Detalle por modelo")
    tabs = st.tabs([m["model_name"] for m in classical])
    for tab, m in zip(tabs, classical):
        with tab:
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Hiperparámetros seleccionados (CV)**")
                hp_df = pd.DataFrame(
                    [{"Hiperparámetro": k, "Valor": str(v)}
                     for k, v in m["best_hyperparameters"].items()]
                )
                st.dataframe(hp_df, use_container_width=True, hide_index=True)
                st.metric("F1 medio en CV (5-fold)", f"{m['cv_best_f1']:.4f}")
                st.metric("Tiempo de entrenamiento", f"{m['training_time_seconds']:.2f} s")
            with cols[1]:
                st.markdown("**Matriz de confusión (test)**")
                cm_path = fig_path(f"{m['model_key']}_cm.png")
                if cm_path.exists():
                    st.image(str(cm_path), use_container_width=True)
                else:
                    st.info("Matriz no disponible.")

            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Métricas en validación**")
                st.dataframe(metrics_table(m["val_metrics"], "umbral 0,50"),
                             use_container_width=True, hide_index=True)
            with cols[1]:
                st.markdown("**Métricas en prueba**")
                st.dataframe(metrics_table(m["test_metrics"], "umbral 0,50"),
                             use_container_width=True, hide_index=True)

            if m.get("feature_importances_top20"):
                st.markdown("**Variables más relevantes**")
                imp_df = pd.DataFrame(m["feature_importances_top20"][:10])
                st.dataframe(
                    imp_df.style.format({"importance": "{:.4f}"}),
                    use_container_width=True, hide_index=True,
                )


def page_mlp() -> None:
    st.title("Red neuronal multicapa (MLP)")
    mlp = load_mlp()
    hp = mlp["hyperparameters"]

    st.markdown(
        """
        La red neuronal multicapa constituye el **modelo principal del estudio**, justificado
        por la capacidad de capturar interacciones no lineales entre variables clínicas,
        genéticas y de hábitos. La arquitectura, el procedimiento de entrenamiento y el
        ajuste del punto de corte siguen un protocolo estricto **anti-fuga de información**
        (skill `threshold-tuning` del proyecto): el conjunto de prueba se reserva íntegramente
        y se evalúa **una única vez** al final del proceso.
        """
    )

    st.subheader("Arquitectura")
    cols = st.columns(2)
    with cols[0]:
        st.code(hp["architecture"], language="text")
        st.markdown(f"**Parámetros entrenables totales:** `{hp['total_params']:,}`")
        st.markdown(f"**Dimensión de entrada (post-preprocesado):** `{hp['input_dim']}`")
    with cols[1]:
        st.markdown(
            f"""
            - **Inicialización de pesos:** `{hp['kernel_initializer']}`
            - **Función de pérdida:** `{hp['loss']}`
            - **Optimizador:** `{hp['optimizer']}` (lr inicial = `{hp['learning_rate_initial']}`)
            - **Tamaño de lote:** `{hp['batch_size']}`
            - **Épocas máximas:** `{hp['max_epochs']}` · **Reales:** `{hp['epochs_actual']}`
            - **Detención temprana:** monitor `{hp['early_stopping_monitor']}`,
              paciencia `{hp['early_stopping_patience']}`
            - **Reducción de tasa:** paciencia `{hp['reduce_lr_patience']}`
            """
        )

    with st.expander("Justificación del dimensionado"):
        st.markdown(hp["note"])

    st.subheader("Ponderación de clases (gestión del desbalance)")
    cw_df = pd.DataFrame(
        [{"Clase": "Sin diagnóstico (0)", "Peso": hp["class_weight"]["0"]},
         {"Clase": "Con diagnóstico (1)", "Peso": hp["class_weight"]["1"]}]
    )
    st.dataframe(cw_df, use_container_width=True, hide_index=True)

    st.subheader("Curvas de aprendizaje")
    cols = st.columns(2)
    with cols[0]:
        show_figure("mlp_loss_curve.png",
                    "Pérdida en entrenamiento y validación por época.")
    with cols[1]:
        show_figure("mlp_auc_curve.png",
                    "AUC-ROC en validación por época.")

    st.markdown("---")
    st.subheader("Selección del punto de corte (anti-fuga)")
    st.markdown(
        f"""
        El punto de corte de decisión se ajusta **únicamente** sobre el conjunto de validación
        (8 000 pacientes, separado del entrenamiento mediante división estratificada). Se barre
        el rango \\[0,01; 0,99\\] en pasos de 0,01, seleccionando el valor que maximiza F1 sobre
        la clase positiva.

        - **Punto de corte por defecto:** `{mlp['threshold_default']}`
        - **Punto de corte óptimo (F1-max sobre validación):** `{mlp['threshold_optimal_f1']}`
        - **Punto de corte por índice de Youden (referencia):** `{mlp['threshold_youden']}`

        El conjunto de prueba **no se utiliza en este proceso**; se evalúa únicamente al final
        con ambos puntos de corte.
        """
    )
    show_figure("mlp_threshold_sweep.png",
                "Barrido de F1, precisión y sensibilidad frente al punto de corte sobre validación.")

    st.markdown("---")
    st.subheader("Métricas de la red")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Validación (umbral 0,50)**")
        st.dataframe(metrics_table(mlp["val_metrics_default"], "umbral 0,50"),
                     use_container_width=True, hide_index=True)
        st.markdown(f"**Validación (umbral óptimo {mlp['threshold_optimal_f1']:.2f})**")
        st.dataframe(metrics_table(mlp["val_metrics_optimal"],
                                   f"umbral {mlp['threshold_optimal_f1']:.2f}"),
                     use_container_width=True, hide_index=True)
    with cols[1]:
        st.markdown("**Prueba (umbral 0,50)**")
        st.dataframe(metrics_table(mlp["test_metrics_default"], "umbral 0,50"),
                     use_container_width=True, hide_index=True)
        st.markdown(f"**Prueba (umbral óptimo {mlp['threshold_optimal_f1']:.2f})**")
        st.dataframe(metrics_table(mlp["test_metrics_optimal"],
                                   f"umbral {mlp['threshold_optimal_f1']:.2f}"),
                     use_container_width=True, hide_index=True)

    st.subheader("Curva ROC y curva precisión-recall (test)")
    cols = st.columns(2)
    with cols[0]:
        show_figure("mlp_roc.png", "Curva ROC sobre el conjunto de prueba.")
    with cols[1]:
        show_figure("mlp_pr.png", "Curva precisión-recall sobre el conjunto de prueba.")

    st.subheader("Matrices de confusión (test)")
    cols = st.columns(2)
    with cols[0]:
        show_figure("mlp_cm_default.png", "Umbral por defecto (0,50).")
    with cols[1]:
        show_figure("mlp_cm_optimal.png",
                    f"Umbral óptimo seleccionado en validación ({mlp['threshold_optimal_f1']:.2f}).")


def page_comparativa() -> None:
    st.title("Comparativa final y discusión clínica")
    mlp = load_mlp()
    classical = load_classical()
    xgb = next(m for m in classical if m["model_key"] == "xgboost")

    st.markdown(
        """
        La selección entre el modelo clásico de referencia (XGBoost) y la red neuronal MLP se
        formula como un **trade-off entre el ranking probabilístico** (capturado por AUC-ROC)
        **y la calidad de la decisión binaria** al punto de corte operativo (capturada por F1
        sobre la clase positiva).
        """
    )

    cmp = pd.DataFrame([
        {
            "Modelo": "XGBoost (referencia clásica)",
            "Punto de corte": f"{xgb['threshold']:.2f}",
            "F1 (clase positiva)": xgb["test_metrics"]["f1"],
            "AUC-ROC": xgb["test_metrics"]["auc_roc"],
            "Precisión (VPP)": xgb["test_metrics"]["precision"],
            "Sensibilidad": xgb["test_metrics"]["recall"],
            "Exactitud": xgb["test_metrics"]["accuracy"],
        },
        {
            "Modelo": "MLP (modelo final)",
            "Punto de corte": f"{mlp['threshold_optimal_f1']:.2f}",
            "F1 (clase positiva)": mlp["test_metrics_optimal"]["f1"],
            "AUC-ROC": mlp["test_metrics_optimal"]["auc_roc"],
            "Precisión (VPP)": mlp["test_metrics_optimal"]["precision"],
            "Sensibilidad": mlp["test_metrics_optimal"]["recall"],
            "Exactitud": mlp["test_metrics_optimal"]["accuracy"],
        },
    ])
    st.dataframe(
        cmp.style.format({c: "{:.4f}" for c in cmp.columns
                          if c not in {"Modelo", "Punto de corte"}}),
        use_container_width=True, hide_index=True,
    )

    show_figure("final_models_comparison.png",
                "Comparativa de F1 y AUC-ROC entre el modelo clásico de referencia y la MLP.")

    st.markdown("---")
    st.subheader("Discusión clínica")

    delta_f1 = mlp["test_metrics_optimal"]["f1"] - xgb["test_metrics"]["f1"]
    delta_auc = mlp["test_metrics_optimal"]["auc_roc"] - xgb["test_metrics"]["auc_roc"]

    st.markdown(
        f"""
        - **Decisión clínica al punto de corte operativo.** La MLP, evaluada al umbral óptimo
          ajustado sobre validación ({mlp['threshold_optimal_f1']:.2f}), supera al modelo de
          referencia en F1 con un incremento absoluto de **{delta_f1:+.4f}**, lo que se traduce
          en un mejor equilibrio entre valor predictivo positivo y sensibilidad.
        - **Capacidad de ordenación probabilística.** El AUC-ROC de la MLP es marginalmente
          inferior al del modelo de referencia ({delta_auc:+.4f}); ambos modelos comparten una
          capacidad de discriminación clínicamente equivalente al rango de la cohorte.
        - **Trade-off precisión-sensibilidad.** El umbral seleccionado privilegia la precisión
          frente a la sensibilidad, lo cual es coherente con un escenario de **cribado de
          segunda línea** (priorizar verdaderos positivos detectados frente a la generación de
          falsas alarmas). Si el caso de uso clínico requiriera maximizar la sensibilidad —por
          ejemplo, un cribado poblacional de primera línea— el umbral debería revisarse hacia
          valores más bajos, asumiendo el incremento de falsos positivos asociado.
        - **Reproducibilidad.** Toda la cadena de procesamiento utiliza semilla fija (42),
          división estratificada 80/20 y ajustes de hiperparámetros y umbral sin contacto con
          el conjunto de prueba.
        """
    )

    st.markdown("---")
    st.subheader("Limitaciones del estudio")
    st.markdown(
        """
        - **Naturaleza simulada del dataset.** Las distribuciones provienen de un modelo
          generativo académico y no reflejan totalmente la heterogeneidad de una cohorte
          clínica real (sesgo de selección, ruido en variables auto-reportadas, errores de
          codificación, etc.).
        - **Ausencia de validación externa.** Los resultados se evalúan únicamente sobre el
          conjunto de prueba interno; antes de cualquier despliegue serían necesarias
          validaciones externas en cohortes independientes.
        - **Calibración no auditada.** Las probabilidades emitidas por la MLP no han sido
          recalibradas (Platt scaling, isotonic). En un uso clínico, la calibración es tan
          crítica como la discriminación.
        - **Equidad y sesgo.** No se ha realizado análisis estratificado por subgrupos
          (sexo, edad, nivel socioeconómico). Antes de cualquier uso, la equidad debe
          evaluarse formalmente.
        - **Carácter no clínico.** Esta aplicación es un ejercicio académico. **No debe
          utilizarse para tomar decisiones clínicas.**
        """
    )


def _get_predictor_inputs() -> pd.DataFrame:
    cols = st.columns(2)

    with cols[0]:
        st.markdown("**Datos sociodemográficos**")
        edad = st.number_input("Edad (años)", min_value=20.0, max_value=90.0, value=55.0, step=1.0)
        num_hijos = st.number_input("Número de hijos", min_value=0, max_value=8, value=1, step=1)
        distancia_hospital_km = st.number_input(
            "Distancia al hospital (km)", min_value=0.5, max_value=250.0, value=15.0, step=0.5
        )
        nivel_educativo = st.selectbox("Nivel educativo", CATEGORICAL_LEVELS["nivel_educativo"], index=1)
        nivel_ingresos = st.selectbox("Nivel de ingresos", CATEGORICAL_LEVELS["nivel_ingresos"], index=2)
        zona = st.selectbox("Zona de residencia", CATEGORICAL_LEVELS["zona"], index=2)
        estado_civil = st.selectbox("Estado civil", CATEGORICAL_LEVELS["estado_civil"], index=0)
        tipo_seguro = st.selectbox("Tipo de cobertura sanitaria", CATEGORICAL_LEVELS["tipo_seguro"], index=2)

        st.markdown("**Hábitos de vida**")
        actividad_fisica = st.selectbox("Actividad física habitual",
                                        CATEGORICAL_LEVELS["actividad_fisica"], index=2)
        fumador = st.checkbox("Hábito tabáquico activo")

    with cols[1]:
        st.markdown("**Parámetros bioquímicos**")
        glucosa = st.number_input("Glucosa (mg/dL)", min_value=55.0, max_value=180.0, value=101.0, step=0.5)
        colesterol = st.number_input("Colesterol total (mg/dL)", min_value=120.0, max_value=320.0,
                                     value=193.0, step=1.0)
        trigliceridos = st.number_input("Triglicéridos (mg/dL)", min_value=50.0, max_value=322.0,
                                        value=155.0, step=1.0)
        hemoglobina = st.number_input("Hemoglobina (g/dL)", min_value=8.0, max_value=18.0,
                                      value=14.0, step=0.1)
        leucocitos = st.number_input("Leucocitos (×10³/uL)", min_value=2.0, max_value=16.0,
                                     value=7.0, step=0.1)
        plaquetas = st.number_input("Plaquetas (×10³/uL)", min_value=100.0, max_value=480.0,
                                    value=255.0, step=1.0)
        creatinina = st.number_input("Creatinina (mg/dL)", min_value=0.35, max_value=2.10,
                                     value=1.00, step=0.01)

        st.markdown("**Comorbilidades clínicas**")
        diabetes = st.checkbox("Diabetes mellitus")
        hipertension = st.checkbox("Hipertensión arterial")
        obesidad = st.checkbox("Obesidad")
        enfermedad_cardiaca = st.checkbox("Enfermedad cardíaca")
        asma = st.checkbox("Asma")
        epoc = st.checkbox("EPOC")

        st.markdown("**Mutaciones germinales detectadas**")
        mut_BRCA1 = st.checkbox("BRCA1")
        mut_TP53 = st.checkbox("TP53")
        mut_EGFR = st.checkbox("EGFR")
        mut_KRAS = st.checkbox("KRAS")
        mut_PIK3CA = st.checkbox("PIK3CA")
        mut_ALK = st.checkbox("ALK")
        mut_BRAF = st.checkbox("BRAF")

    row = {
        "glucosa": glucosa, "colesterol": colesterol, "trigliceridos": trigliceridos,
        "hemoglobina": hemoglobina, "leucocitos": leucocitos, "plaquetas": plaquetas,
        "creatinina": creatinina, "edad": edad, "num_hijos": int(num_hijos),
        "distancia_hospital_km": distancia_hospital_km,
        "diabetes": int(diabetes), "hipertension": int(hipertension),
        "obesidad": int(obesidad), "enfermedad_cardiaca": int(enfermedad_cardiaca),
        "asma": int(asma), "epoc": int(epoc),
        "mut_BRCA1": int(mut_BRCA1), "mut_TP53": int(mut_TP53),
        "mut_EGFR": int(mut_EGFR), "mut_KRAS": int(mut_KRAS),
        "mut_PIK3CA": int(mut_PIK3CA), "mut_ALK": int(mut_ALK),
        "mut_BRAF": int(mut_BRAF), "fumador": int(fumador),
        "tipo_seguro": tipo_seguro, "actividad_fisica": actividad_fisica,
        "nivel_educativo": nivel_educativo, "nivel_ingresos": nivel_ingresos,
        "zona": zona, "estado_civil": estado_civil,
    }
    return pd.DataFrame([row])


def page_estimador() -> None:
    st.title("Estimador interactivo de riesgo")
    st.warning(
        "**Aviso académico.** Esta herramienta forma parte de un trabajo universitario y "
        "no constituye un dispositivo médico. Los resultados son orientativos y no deben "
        "utilizarse para la toma de decisiones clínicas. Consulte siempre a un profesional "
        "sanitario cualificado."
    )

    st.markdown(
        """
        Introduzca los parámetros del paciente para obtener la estimación de riesgo de los dos
        modelos seleccionados al final del estudio. La interpretación se acompaña del punto
        de corte óptimo determinado durante la fase de validación de la MLP.
        """
    )

    with st.form("predictor_form"):
        df_input = _get_predictor_inputs()
        submitted = st.form_submit_button("Estimar riesgo")

    if not submitted:
        st.info("Complete el formulario y pulse el botón para ejecutar la estimación.")
        return

    mlp_meta = load_mlp()
    threshold_opt = mlp_meta["threshold_optimal_f1"]

    try:
        xgb = load_xgboost()
        proba_xgb = float(xgb.predict_proba(df_input)[0, 1])
    except Exception as exc:
        st.error(f"No se pudo cargar el modelo XGBoost: {exc}")
        return

    try:
        mlp_model, preprocessor = load_mlp_model()
        X = preprocessor.transform(df_input)
        proba_mlp = float(mlp_model.predict(X, verbose=0).flatten()[0])
    except Exception as exc:
        st.error(f"No se pudo cargar la MLP: {exc}")
        return

    st.markdown("### Resultados")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**XGBoost (referencia clásica) — umbral 0,50**")
        st.metric("Probabilidad estimada", f"{proba_xgb:.4f}")
        if proba_xgb >= 0.5:
            st.error("Predicción: **POSITIVA** (riesgo elevado).")
        else:
            st.success("Predicción: **NEGATIVA** (riesgo bajo).")
        st.progress(min(max(proba_xgb, 0.0), 1.0))

    with cols[1]:
        st.markdown(f"**MLP (modelo final) — umbral óptimo {threshold_opt:.2f}**")
        st.metric("Probabilidad estimada", f"{proba_mlp:.4f}")
        if proba_mlp >= threshold_opt:
            st.error("Predicción: **POSITIVA** (riesgo elevado).")
        else:
            st.success("Predicción: **NEGATIVA** (riesgo bajo).")
        st.progress(min(max(proba_mlp, 0.0), 1.0))

    delta = proba_mlp - proba_xgb
    if abs(delta) < 0.05:
        st.info("Ambos modelos coinciden en la magnitud del riesgo estimado.")
    elif delta > 0:
        st.info(
            f"La MLP estima un riesgo **superior** al de XGBoost en {abs(delta):.4f} unidades "
            "de probabilidad."
        )
    else:
        st.info(
            f"La MLP estima un riesgo **inferior** al de XGBoost en {abs(delta):.4f} unidades "
            "de probabilidad."
        )

    with st.expander("Detalle de los datos introducidos"):
        st.dataframe(df_input.T.rename(columns={0: "Valor"}), use_container_width=True)


def page_metodologia() -> None:
    st.title("Metodología y diseño del estudio")

    st.markdown(
        """
        ### Flujo de trabajo del estudio

        1. **Extracción** desde la base de datos clínica en Microsoft Azure (servidor
           `uaxmathfis.database.windows.net`, base `usecases`) con autenticación Azure AD
           interactiva. Salida: seis ficheros CSV bajo `data/raw/`.
        2. **Unión** por `paciente_id` mediante `pd.merge(..., how='left')` iterado sobre las
           seis fuentes. Cobertura del 100 % en todas ellas.
        3. **Análisis exploratorio** descriptivo y bivariante, con detección de:
            - Variables con varianza nula (exclusión sistemática).
            - Variables con fuga post-diagnóstico (correlaciones inverosímiles con el
              desenlace).
            - Variables con señal univariante moderada y plausibilidad biológica.
        4. **División estratificada 80/20** con semilla fija (42) por la variable objetivo.
           Sobre el conjunto de entrenamiento se realiza una segunda partición estratificada
           80/20 para construir el subconjunto de validación interna utilizado en el ajuste
           del punto de corte de la MLP.
        5. **Modelado clásico** con cuatro estimadores (Regresión logística L1,
           Random Forest, XGBoost, HistGradientBoosting) ajustados mediante validación cruzada
           estratificada de 5 particiones, optimizando F1 sobre la clase positiva.
        6. **Modelado neuronal** con red MLP densa, ajustada mediante detención temprana
           sobre AUC-ROC en validación, ponderación inversa de clases y reducción adaptativa
           del *learning rate*.
        7. **Ajuste anti-fuga del punto de corte** sobre el conjunto de validación interna
           (barrido entre 0,01 y 0,99 con paso 0,01; selección por F1-máximo).
        8. **Evaluación final** sobre el conjunto de prueba con punto de corte por defecto y
           con punto de corte óptimo. El conjunto de prueba se utiliza **una única vez**.

        ### Reglas no negociables del estudio

        - Ausencia total de fuga de información: scaler, codificadores y umbral se ajustan
          siempre sobre datos no vistos del conjunto de prueba.
        - División 80/20 estratificada por la variable objetivo.
        - Métricas primarias: F1 sobre la clase positiva y AUC-ROC. La exactitud se reporta
          únicamente como referencia.
        - Ponderación inversa de clases obligatoria para gestionar el desbalance.
        - Semilla fija (42) en todos los componentes estocásticos: `numpy`, `tensorflow`,
          `random` y `sklearn`.
        - Escalado y codificación ajustados exclusivamente sobre el conjunto de
          entrenamiento.

        ### Reproducibilidad

        - Entorno conda: `uax-tf` (Python 3.12, TensorFlow 2.21).
        - Pipeline reproducible mediante los scripts en `src/features/build_dataset.py`,
          `src/models/train_classical.py` y `src/models/train_mlp.py`.
        - Modelos serializados en `models/`.
        - Resultados serializados en `reports/*.json`.
        - Figuras serializadas en `reports/figures/`.

        ### Estructura del repositorio

        ```
        data/
          raw/         Salida del ETL (6 CSV)
          interim/     Tras la unión por paciente_id
          processed/   train.csv, test.csv y feature_groups.json
        src/
          etl/         Conexión a Azure y extracción
          features/    Preprocesado y construcción del dataset
          models/      Entrenamiento de ML clásico y MLP
          app/         Aplicación Streamlit (este informe)
        notebooks/     Exploración interactiva
        reports/
          figures/     Gráficos serializados
          slides/      Entregable final (cancer_uax.pptx)
        models/        Modelos serializados (.keras, .pkl)
        ```
        """
    )


# ----------------------------------------------------------------------------
# Punto de entrada
# ----------------------------------------------------------------------------


PAGES: dict[str, callable] = {  # type: ignore[type-arg]
    "Resumen del estudio": page_resumen,
    "Cohorte y caracterización": page_cohorte,
    "Selección de variables": page_seleccion,
    "Modelos clásicos de referencia": page_clasicos,
    "Red neuronal MLP": page_mlp,
    "Comparativa final y discusión": page_comparativa,
    "Estimador interactivo de riesgo": page_estimador,
    "Metodología": page_metodologia,
}


def main() -> None:
    st.set_page_config(
        page_title="Predicción de cáncer · Estudio UAX",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.sidebar.markdown("## Estudio UAX")
    st.sidebar.markdown(
        "Predicción de diagnóstico de cáncer\n\n"
        "Universidad Alfonso X el Sabio · Ingeniería Matemática · "
        "Inteligencia Artificial · 2025–2026"
    )
    st.sidebar.markdown("---")
    selection = st.sidebar.radio("Secciones del informe", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Trabajo académico. No constituye un dispositivo médico ni una recomendación clínica."
    )
    PAGES[selection]()


if __name__ == "__main__":
    main()
