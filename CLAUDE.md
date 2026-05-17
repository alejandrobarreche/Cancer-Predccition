# Proyecto: Predicción de Diagnóstico de Cáncer (UAX)

## Contexto académico
- **Asignatura:** Inteligencia Artificial — Ingeniería Matemática 2025–2026
- **Tipo:** Caso optativo
- **Entregable final:** exactamente 5 diapositivas (PDF o PowerPoint) + código fuente completo
- **Estado actual:** pipeline end-to-end implementado (ETL en notebook, EDA, dataset split, 4 modelos clásicos, MLP, app Streamlit, slides). Iteración en curso.

## Datos
- **Origen:** base de datos en **Azure** (requiere ETL para extraer la información)
- **Formato intermedio:** 6 ficheros CSV bajo `data/raw/`
- **Clave de unión entre los 6 CSV:** `paciente_id`
- **Volumen:** ~50.001 registros únicos por `paciente_id`
- **Variable objetivo:** `cancer` (binaria, **clase positiva fuertemente desbalanceada**)
- **Diccionario de variables:** `metadata_dataset_cancer.md`, disponible en `.claude/skills/enunciado-context/references/`. Consultarlo antes de decidir selección o transformación de features.

## Stack técnico
- **Entorno:** **conda env `uax-tf`** (gestionado por el usuario). NO usar `venv`, `virtualenv` ni `.venv` paralelos. Activar con `conda activate uax-tf` antes de ejecutar nada.
- **Lenguaje:** Python 3.11+
- **ML clásico:** scikit-learn, xgboost, lightgbm
- **Red neuronal:** TensorFlow / Keras
- **Manipulación:** pandas, numpy
- **Visualización:** matplotlib, seaborn
- **Conexión Azure:** vía DataSource del IDE (`usecases@uaxmathfis.database.windows.net`) con Azure AD interactiva. Detalles abajo. **Nunca credenciales hardcodeadas.**
- **Instalación de dependencias:** `conda install <pkg>` dentro de `uax-tf`; usar `pip install <pkg>` como fallback solo si el paquete no está disponible en conda-forge.

## Conexión a Azure SQL

**Servidor:** `uaxmathfis.database.windows.net:1433` — base de datos `usecases` — auth Azure AD interactiva (usuario `abarrrui`).

### Vía primaria: celdas SQL nativas en notebooks (PyCharm)
- Toda extracción y consulta exploratoria contra Azure SQL se hace desde **celdas SQL en notebooks** ubicados en `notebooks/`.
- **Cada celda SQL** debe declarar explícitamente:
  - **DataSource:** `usecases@uaxmathfis.database.windows.net` (seleccionado en la barra de la celda en PyCharm).
  - **Output variable:** nombre del DataFrame de salida en el namespace del notebook (ej. `df_pacientes`).
- Convención adicional: como primera línea de cada celda SQL dejar dos comentarios para que la convención sea legible aunque la celda se ejecute como texto:
  ```sql
  -- @datasource: usecases@uaxmathfis.database.windows.net
  -- @output: df_<nombre>
  SELECT ...
  ```
- **Sin credenciales en el repo.** El IDE gestiona el token AAD; el contenedor del DataSource ya está versionado en `.idea/dataSources.xml` y el secret nunca sale de la máquina.

### Vía secundaria: ETL programático (pendiente)
- No implementada todavía. Si en el futuro se necesita una extracción scriptable (CI), añadir un módulo bajo `src/` con `pyodbc` + `azure-identity` (`Authentication=ActiveDirectoryInteractive`). Plantilla en `.env.example`.
- **No** usar usuario/contraseña SQL — solo AAD.

## Reglas críticas (no negociables)
1. **Sin data leakage.** El umbral de la MLP se ajusta SOLO sobre el conjunto de validación, nunca sobre test. Si alguien lo pide, hay que negarse y explicar por qué.
2. **División 80/20 estratificada por la variable objetivo** (`stratify=y`).
3. **Métricas principales:** F1 sobre la clase positiva y AUC-ROC. Reportar también precisión y recall.
4. **Accuracy es solo referencia**, nunca criterio único de selección.
5. **La MLP usa `class_weight`** para gestionar el desbalance.
6. **Random seed fijo (`42`)** en todo experimento que deba ser reproducible (`numpy`, `tensorflow`, `random`, `sklearn`).
7. **El escalado/normalización se ajusta sobre `train` y se aplica a `test`** — nunca al revés ni sobre el conjunto completo.

## Convenciones de código
- Type hints donde aporten claridad.
- Funciones puras y testeables en `src/`; los notebooks son solo para exploración.
- Imports ordenados (stdlib, terceros, locales) y nombres de variables descriptivos en español o inglés, pero **consistentes** dentro de un módulo.
- Logging estructurado a fichero cuando un script genere artefactos persistentes.

## Estructura del repositorio
```
data/
  raw/         # Salida del ETL — 6 CSV
  interim/     # Tras unir por paciente_id
  processed/   # Listo para entrenar (train.csv, test.csv)
src/
  features/    # Preprocesado, scaling, feature engineering
  models/      # Entrenamiento de ML clásico y MLP
  app/         # Aplicación Streamlit con informe interactivo
notebooks/     # Exploración y ETL (vía celdas SQL nativas)
reports/       # Resultados serializados organizados por fase
  eda/         # memo.md, report.json, generate_figures.py + figures/
  classical/   # results.json (global) + {model}.json + figures/
  mlp/         # results.json, train.log + figures/
  comparison/  # figures/ con comparativas entre modelos
  slides/      # Entregable final (.pptx o .pdf, 5 slides)
tests/         # pytest
models/        # Modelos serializados (.keras, .pkl)
```

## Comandos útiles
- Activar entorno: `conda activate uax-tf`
- Tests: `pytest tests/`
- Generar dataset procesado: `python -m src.features.build_dataset`
- Entrenar modelos clásicos: `python -m src.models.train_classical`
- Entrenar MLP: `python -m src.models.train_mlp`
- App interactiva: `streamlit run src/app/streamlit_app.py`
- Generar slides: `python reports/build_slides.py`

## Subagentes y skills disponibles
Este proyecto define cinco subagentes especializados en `.claude/agents/`:
- `etl-azure` — extracción desde Azure a CSV
- `data-explorer` — EDA y unión de las 6 fuentes
- `ml-classical` — modelos clásicos (mín. 3 modelos complejos)
- `mlp-designer` — red neuronal MLP (núcleo técnico del trabajo)
- `deliverable-builder` — construcción de las 5 diapositivas finales

Y cuatro skills transversales en `.claude/skills/`:
- `project-conventions` — reglas de código y estructura
- `ml-evaluation-protocol` — formato oficial de evaluación y reporte
- `threshold-tuning` — procedimiento anti-leakage para el umbral de la MLP
- `enunciado-context` — acceso al enunciado y al diccionario de variables
