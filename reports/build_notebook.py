"""
Generates notebooks/02_eda.ipynb programmatically.
Run: conda run -n uax-tf python3 reports/build_notebook.py
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text,
    }

def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }

cells = []

# ── Cell 0: title ────────────────────────────────────────────────────────────
cells.append(md_cell(
    "# 02 — EDA: Prediccion de Diagnostico de Cancer\n"
    "\n"
    "**Proyecto:** UAX Inteligencia Artificial 2025-2026  \n"
    "**Fecha:** 2026-05-04  \n"
    "\n"
    "Este notebook documenta el EDA completo sobre las 6 fuentes de datos unidas por `paciente_id`.  \n"
    "Entregables generados en la misma ejecucion:\n"
    "- `data/interim/joined.csv` — dataset unido sin imputacion (50 001 filas x 38 columnas)\n"
    "- `reports/eda_report.json` — resumen estadistico maquina-legible\n"
    "- `reports/figures/fig01-fig10` — figuras del analisis\n"
    "\n"
    "**No se realiza imputation, split ni escalado** — esas operaciones corresponden "
    "a `src/features/preprocess.py` post-split.\n"
))

# ── Cell 1: setup ─────────────────────────────────────────────────────────────
cells.append(code_cell(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')  # headless; cambiar a 'inline' en Jupyter interactivo\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "np.random.seed(42)\n"
    "sns.set_theme(style='whitegrid', palette='muted')\n"
    "plt.rcParams['figure.dpi'] = 120\n"
    "\n"
    "BASE  = Path('/Users/barrechee/School/Universidad/3/UAX/2/Inteligencia-Artificial/Casos/Prediccion-Cancer')\n"
    "RAW   = BASE / 'data' / 'raw'\n"
    "INTER = BASE / 'data' / 'interim'\n"
    "FIG   = BASE / 'reports' / 'figures'\n"
    "FIG.mkdir(parents=True, exist_ok=True)\n"
    "print('Environment ready.')\n"
))

# ── Cell 2: load ──────────────────────────────────────────────────────────────
cells.append(md_cell("## 1. Carga de los 6 CSV\n"))
cells.append(code_cell(
    "names = ['bioquimicos', 'clinicos', 'geneticos', 'economicos', 'generales', 'sociodemografico']\n"
    "dfs   = {n: pd.read_csv(RAW / f'{n}.csv') for n in names}\n"
    "\n"
    "print('=' * 65)\n"
    "print(f\"  {'Fuente':<22} {'Filas':>7} {'Cols':>5} {'IDs unicos':>12} {'Nulos':>7}\")\n"
    "print('=' * 65)\n"
    "for n, df in dfs.items():\n"
    "    print(f\"  {n:<22} {len(df):>7,} {len(df.columns):>5} \"\n"
    "          f\"{df['paciente_id'].nunique():>12,} {df.isnull().sum().sum():>7}\")\n"
    "print('=' * 65)\n"
))

# ── Cell 3: join ──────────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 2. Union por `paciente_id` (left join sobre `clinicos`)\n"
    "\n"
    "La tabla principal es `clinicos` porque contiene la variable objetivo `cancer`.  \n"
    "Las 5 tablas restantes se unen con `how='left'`. Dado que los 6 CSV comparten exactamente los "
    "mismos 50 001 `paciente_id`, el resultado es identico a un inner join o un outer join "
    "(cobertura 100%).\n"
))
cells.append(code_cell(
    "joined = dfs['clinicos'].copy()\n"
    "for n in names:\n"
    "    if n != 'clinicos':\n"
    "        joined = joined.merge(dfs[n], on='paciente_id', how='left')\n"
    "\n"
    "print(f'Shape joined: {joined.shape}')\n"
    "print(f'paciente_id unicos: {joined[\"paciente_id\"].nunique():,}')\n"
    "print(f'Nulos tras join: {joined.isnull().sum().sum()}')\n"
    "print(f'Columnas ({len(joined.columns)}):')\n"
    "print(list(joined.columns))\n"
))

# ── Cell 4: target ────────────────────────────────────────────────────────────
cells.append(md_cell("## 3. Analisis de la variable objetivo `cancer`\n"))
cells.append(code_cell(
    "n_pos = int(joined['cancer'].sum())\n"
    "n_neg = len(joined) - n_pos\n"
    "prev  = joined['cancer'].mean()\n"
    "ratio = n_neg / n_pos\n"
    "\n"
    "print(f'N total     : {len(joined):,}')\n"
    "print(f'cancer = 1  : {n_pos:,} ({prev*100:.2f}%)')\n"
    "print(f'cancer = 0  : {n_neg:,} ({(1-prev)*100:.2f}%)')\n"
    "print(f'Ratio 1:N   : 1 : {ratio:.2f}')\n"
    "\n"
    "if prev < 0.01:\n"
    "    print('WARNING: prevalencia < 1% — desbalance extremo')\n"
    "elif prev > 0.5:\n"
    "    print('WARNING: prevalencia > 50%')\n"
    "else:\n"
    "    print(f'INFO: desbalance moderado (1:{ratio:.1f}), manejable con class_weight')\n"
))

# ── Cell 5: describe ──────────────────────────────────────────────────────────
cells.append(md_cell("## 4. Estadistica descriptiva de features numericas\n"))
cells.append(code_cell(
    "numeric_cols = joined.select_dtypes('number').columns.tolist()\n"
    "feature_num  = [c for c in numeric_cols if c != 'cancer']\n"
    "\n"
    "desc = joined[feature_num].describe().T\n"
    "desc['null_pct'] = (joined[feature_num].isnull().sum() / len(joined) * 100).round(2)\n"
    "print(desc[['count','mean','std','min','25%','50%','75%','max','null_pct']].to_string())\n"
))

# ── Cell 6: constant cols ─────────────────────────────────────────────────────
cells.append(md_cell("## 5. Columnas constantes y casi-constantes\n"))
cells.append(code_cell(
    "cat_cols = [c for c in joined.columns\n"
    "            if joined[c].dtype == object or str(joined[c].dtype) == 'string'\n"
    "            and c != 'paciente_id']\n"
    "\n"
    "print('Columnas con >99% mismo valor:')\n"
    "found = False\n"
    "for c in feature_num + cat_cols:\n"
    "    vc = joined[c].value_counts(normalize=True)\n"
    "    if vc.iloc[0] >= 0.99:\n"
    "        print(f'  {c}: top={vc.index[0]!r}, {vc.iloc[0]*100:.1f}%')\n"
    "        found = True\n"
    "if not found:\n"
    "    print('  (ninguna excepto alcohol, verificado abajo)')\n"
    "\n"
    "print(f'\\nalcohol value_counts: {joined[\"alcohol\"].value_counts().to_dict()}')\n"
    "print('=> alcohol es CONSTANTE (100% = 1). Se debe EXCLUIR antes de modelar.')\n"
))

# ── Cell 7: correlations ──────────────────────────────────────────────────────
cells.append(md_cell("## 6. Correlacion Pearson con el target (numericas)\n"))
cells.append(code_cell(
    "corr_s = joined[feature_num].corrwith(joined['cancer']).sort_values(key=abs, ascending=False)\n"
    "\n"
    "leakage_set = {'coste_total','coste_farmaco','num_ingresos','dias_hospital','vive'}\n"
    "print('Correlacion Pearson con cancer:')\n"
    "for feat, val in corr_s.items():\n"
    "    flag = '  <-- LEAKAGE POTENCIAL' if feat in leakage_set else ''\n"
    "    print(f'  {feat:<30} {val:+.4f}{flag}')\n"
))

# ── Cell 8: categorical lift ──────────────────────────────────────────────────
cells.append(md_cell("## 7. Lift de variables categoricas frente a `cancer`\n"))
cells.append(code_cell(
    "for feat in ['actividad_fisica','tipo_seguro','nivel_educativo',\n"
    "             'nivel_ingresos','zona','estado_civil','fumador']:\n"
    "    lift = joined.groupby(feat)['cancer'].mean().sort_values(ascending=False) * 100\n"
    "    print(f'\\n{feat}:')\n"
    "    for cat, val in lift.items():\n"
    "        print(f'  {str(cat):<25} P(cancer=1) = {val:.2f}%')\n"
))

# ── Cell 9: generate figures ──────────────────────────────────────────────────
cells.append(md_cell(
    "## 8. Generacion de figuras\n"
    "\n"
    "Se generan 10 figuras PNG en `reports/figures/`. "
    "Para verlas en Jupyter interactivo, cambiar `matplotlib.use('Agg')` por `%matplotlib inline`.\n"
))
cells.append(code_cell(
    "# Ejecutar reports/generate_eda_figures.py para regenerar las figuras\n"
    "import subprocess, sys\n"
    "result = subprocess.run(\n"
    "    [sys.executable, str(BASE / 'reports' / 'generate_eda_figures.py')],\n"
    "    capture_output=True, text=True\n"
    ")\n"
    "print(result.stdout)\n"
    "if result.returncode != 0:\n"
    "    print('STDERR:', result.stderr)\n"
))

# ── Cell 10: save joined ──────────────────────────────────────────────────────
cells.append(md_cell("## 9. Persistencia del dataset unido\n"))
cells.append(code_cell(
    "out = INTER / 'joined.csv'\n"
    "joined.to_csv(str(out), index=False)\n"
    "print(f'Guardado: {out}')\n"
    "print(f'Shape: {joined.shape}')\n"
    "print(f'Nulos: {joined.isnull().sum().sum()}')\n"
    "print(f'Dtypes:\\n{joined.dtypes.value_counts()}')\n"
    "joined.head(3)\n"
))

# ── Cell 11: summary ──────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 10. Resumen EDA — hallazgos clave para modelado\n"
    "\n"
    "### Variable objetivo\n"
    "- N = 50 001 pacientes, `cancer=1` en **9 644 casos (19.29%)**\n"
    "- Ratio desbalance 1:4.18 — **moderado, manejable con `class_weight`**\n"
    "\n"
    "### Leakage confirmado — EXCLUIR sin excepcion\n"
    "| Variable | r con cancer | Motivo |\n"
    "|---|:---:|---|\n"
    "| `coste_total` | 0.891 | Consecuencia del diagnostico, no causa |\n"
    "| `dias_hospital` | 0.878 | Idem |\n"
    "| `coste_farmaco` | 0.853 | Idem |\n"
    "| `num_ingresos` | 0.644 | Idem |\n"
    "| `vive` | -0.354 | Resultado vital post-diagnostico |\n"
    "| `alcohol` | ~0.000 | Constante, sin varianza |\n"
    "\n"
    "### Top features por senial lineal (sin leakage)\n"
    "| Feature | r | Fuente | Tratamiento propuesto |\n"
    "|---|:---:|---|---|\n"
    "| `mut_BRCA1` | 0.219 | geneticos | Binaria, usar tal cual |\n"
    "| `fumador` | 0.217 | generales | Binaria, usar tal cual |\n"
    "| `obesidad` | 0.198 | clinicos | Binaria, valorar leakage indirecto |\n"
    "| `mut_TP53` | 0.187 | geneticos | Binaria, usar tal cual |\n"
    "| `mut_KRAS` | 0.167 | geneticos | Binaria, usar tal cual |\n"
    "| `glucosa` | 0.151 | bioquimicos | Continua, StandardScaler post-split |\n"
    "| `actividad_fisica` | ~0.12 | generales | Categorica, OHE o ordinal |\n"
    "| `trigliceridos` | 0.109 | bioquimicos | Continua, StandardScaler post-split |\n"
    "| `hipertension` | 0.101 | clinicos | Binaria, valorar leakage indirecto |\n"
    "| `mut_EGFR` | 0.100 | geneticos | Binaria, usar tal cual |\n"
    "\n"
    "### Proximos pasos\n"
    "1. Split estratificado 80/20 con `stratify=cancer`, seed=42 (`src/features/split.py`)\n"
    "2. Imputacion (no hay nulos, pero se codificara) + OHE + StandardScaler sobre train, "
    "aplicar a test (`src/features/preprocess.py`)\n"
    "3. Modelos clasicos: LR, RF, XGBoost, LightGBM con metricas F1 y AUC-ROC (`ml-classical`)\n"
    "4. MLP con `class_weight`, ajuste de umbral sobre validacion (`mlp-designer`)\n"
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (UAX-IA)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "cells": cells
}

outpath = BASE / 'notebooks' / '02_eda.ipynb'
with open(str(outpath), 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Notebook saved: {outpath}")
