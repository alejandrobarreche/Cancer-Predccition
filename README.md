# Predicción de Diagnóstico de Cáncer

Caso académico de la asignatura **Inteligencia Artificial** del Grado en Ingeniería Matemática de la **Universidad Alfonso X el Sabio** (curso 2025–2026).

El proyecto construye un pipeline reproducible que:

1. Extrae 6 tablas clínicas desde una base de datos en **Azure SQL**.
2. Las une por `paciente_id` y realiza el análisis exploratorio detectando variables con *data leakage*.
3. Entrena varios modelos clásicos de ML como línea base.
4. Diseña, entrena y ajusta el umbral de una **red neuronal MLP** siguiendo un protocolo anti-leakage estricto.
5. Presenta los resultados en una aplicación interactiva y en un entregable de 5 diapositivas.

> ⚠️ **Carácter académico.** Este trabajo no constituye un dispositivo médico ni una recomendación clínica. Los datos provienen de un modelo generativo simulado.

---

## Datos

- **Fuente:** base de datos `usecases` en `uaxmathfis.database.windows.net` (Azure SQL, auth Azure AD interactiva).
- **6 tablas crudas** unidas por `paciente_id`: bioquímicos, clínicos, genéticos, económicos, generales (hábitos) y sociodemográficos.
- **Volumen:** ~50 000 pacientes únicos, ninguna fuente con missingness.
- **Target:** `cancer` (binaria, desbalanceada — la clase positiva ronda el 19 %).
- **Diccionario de variables:** `.claude/skills/enunciado-context/references/metadata_dataset_cancer.md`.

Los CSV crudos y los datasets derivados **no se versionan** (ver `.gitignore`). Se regeneran ejecutando el pipeline.

---

## Stack técnico

- **Python 3.12** en el conda env `uax-tf`.
- **TensorFlow / Keras** para la MLP.
- **scikit-learn + XGBoost** para los modelos clásicos.
- **pandas / numpy / scipy** para manipulación.
- **matplotlib / seaborn** para visualización.
- **Streamlit** para la app interactiva.
- **python-pptx** para el entregable.

Dependencias completas en `requirements.txt`.

---

## Estructura del repositorio

```
data/
  raw/             # 6 CSV extraídos desde Azure (gitignored)
  interim/         # Dataset unido por paciente_id (gitignored)
  processed/       # train.csv, test.csv, feature_groups.json (gitignored)
notebooks/
  01_etl_azure.ipynb    # Extracción vía celdas SQL nativas del IDE
  02_eda.ipynb          # EDA y detección de leakage
src/
  features/
    build_dataset.py    # Split 80/20 estratificado + manifiesto de features
  models/
    pipelines.py        # Preprocesador (StandardScaler + passthrough + OHE)
    train_classical.py  # LR, Random Forest, XGBoost, HistGradientBoosting
    mlp.py              # Arquitectura MLP parametrizable
    train_mlp.py        # Entrenamiento, threshold tuning y evaluación de la MLP
  app/
    streamlit_app.py    # Informe interactivo + estimador de riesgo
reports/
  eda/             # Informe (memo.md), report.json, generate_figures.py + figures/
  classical/       # results.json global, {model}.json por modelo + figures/
  mlp/             # results.json, train.log + figures/
  comparison/      # figures/ con comparativas entre modelos (p. ej. MLP vs XGBoost)
  slides/          # Entregable .pptx
  build_slides.py  # Generador de las 5 diapositivas
  build_notebook.py# Andamiaje para regenerar notebooks/02_eda.ipynb
models/            # Modelos serializados .keras / .pkl (gitignored)
tests/             # pytest
```

---

## Cómo ponerlo en marcha

### 1. Entorno

```bash
conda activate uax-tf
# Si necesitas instalar dependencias desde cero:
pip install -r requirements.txt
```

> **No** uses `venv`, `virtualenv` ni `.venv` paralelos. El entorno es `uax-tf` y está gestionado con conda.

### 2. Configuración de Azure

La extracción se hace desde **celdas SQL nativas en PyCharm** apuntando al DataSource `usecases@uaxmathfis.database.windows.net` (auth Azure AD interactiva, sin credenciales en el repo). Detalle en `CLAUDE.md` y plantilla en `.env.example`.

### 3. Ejecución del pipeline

```bash
# 1. Extracción (en PyCharm, abrir el notebook y ejecutar las celdas SQL)
notebooks/01_etl_azure.ipynb

# 2. EDA (opcional — el informe ya está en reports/eda/memo.md)
notebooks/02_eda.ipynb

# 3. Construcción del dataset de train/test
python -m src.features.build_dataset

# 4. Entrenamiento de modelos clásicos
python -m src.models.train_classical

# 5. Entrenamiento de la MLP
python -m src.models.train_mlp

# 6. App interactiva con los resultados
streamlit run src/app/streamlit_app.py

# 7. Generación del entregable de 5 diapositivas
python reports/build_slides.py
```

Cada script escribe en `reports/` (métricas + figuras) y en `models/` (pesos serializados).

---

## Reglas críticas (anti-leakage)

Son **no negociables** y están aplicadas en el código:

1. **Sin data leakage.** El umbral de la MLP se ajusta exclusivamente sobre validación, **nunca sobre test**.
2. **División 80/20 estratificada** por la variable objetivo (`stratify=y`).
3. **Métricas principales:** F1 sobre la clase positiva y AUC-ROC. Precisión y recall se reportan también. La accuracy es referencia, no criterio de selección.
4. **`class_weight` balanceado** en la MLP y en los clásicos que lo soportan; `scale_pos_weight` en XGBoost.
5. **Semilla fija `42`** en `numpy`, `tensorflow`, `random` y `sklearn`.
6. **Escalado y codificación se ajustan solo sobre train** y se aplican a val/test.
7. **Variables con leakage post-diagnóstico excluidas sistemáticamente** (`coste_total`, `coste_farmaco`, `num_ingresos`, `dias_hospital`, `vive`).

---

## Tests

```bash
pytest tests/
```

Cubren las constantes anti-leakage del split, el preprocesador y el ajuste de umbral.

---

## Subagentes y skills

El proyecto define agentes y skills especializados en `.claude/`:

- **Subagentes:** `etl-azure`, `data-explorer`, `ml-classical`, `mlp-designer`, `deliverable-builder`.
- **Skills:** `project-conventions`, `ml-evaluation-protocol`, `threshold-tuning`, `enunciado-context`.

Detalle en `CLAUDE.md`.

---

## Estado y resultados

El pipeline está completo y se itera sobre arquitectura, hiperparámetros y selección de features. Los resultados actuales viven en:

- `reports/classical/results.json` y `reports/classical/{model}.json` — métricas de cada clásico.
- `reports/mlp/results.json` — métricas e historial de la MLP.
- `reports/{eda,classical,mlp,comparison}/figures/` — curvas, matrices de confusión y comparativas por fase.
- `reports/slides/cancer_uax.pptx` — entregable final.

### Dashboard interactivo (web app)

El dashboard se visualiza levantando la **web app de Streamlit**. Desde la raíz del proyecto, con el entorno `uax-tf` activado:

```bash
streamlit run src/app/streamlit_app.py
```

Esto abre el dashboard en el navegador (`http://localhost:8501`) con el informe interactivo y el estimador de riesgo.
