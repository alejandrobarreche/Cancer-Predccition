"""
EDA Figure generation script.
Run with: conda run -n uax-tf python3 reports/eda/generate_figures.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

np.random.seed(42)
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

BASE    = Path(__file__).resolve().parents[2]
EDA_DIR = BASE / 'reports' / 'eda'
FIG     = EDA_DIR / 'figures'
RAW_DIR = BASE / 'data' / 'raw'
FIG.mkdir(parents=True, exist_ok=True)

joined = pd.read_csv(BASE / 'data' / 'interim' / 'joined.csv')
prev   = joined['cancer'].mean()
n_pos  = int(joined['cancer'].sum())
n_neg  = len(joined) - n_pos
ratio  = n_neg / n_pos
N      = len(joined)

print(f"Dataset: {joined.shape}, cancer=1: {n_pos} ({prev*100:.2f}%), ratio 1:{ratio:.2f}")

# ── FIG 1: Target distribution ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['cancer=0', 'cancer=1'], [n_neg, n_pos], color=['steelblue', 'tomato'])
axes[0].set_title('Distribucion de la variable objetivo')
axes[0].set_ylabel('Pacientes')
for i, v in enumerate([n_neg, n_pos]):
    axes[0].text(i, v + 300, f'{v:,} ({[1-prev, prev][i]*100:.1f}%)', ha='center', fontsize=9)
axes[1].pie(
    [n_neg, n_pos],
    labels=[f'cancer=0\n({(1-prev)*100:.1f}%)', f'cancer=1\n({prev*100:.1f}%)'],
    colors=['steelblue', 'tomato'], autopct='%1.1f%%', startangle=90
)
axes[1].set_title(f'Ratio desbalance 1:{ratio:.2f}')
plt.tight_layout()
plt.savefig(str(FIG / '01_target_distribution.png'), bbox_inches='tight')
plt.close()
print("fig01 saved")

# ── FIG 2: Bioquimicos distributions ─────────────────────────────────────────
bioq_feats = ['glucosa', 'colesterol', 'trigliceridos', 'hemoglobina', 'leucocitos', 'plaquetas', 'creatinina']
fig, axes = plt.subplots(2, 7, figsize=(22, 7))
for i, c in enumerate(bioq_feats):
    for lbl, color in [(0, 'steelblue'), (1, 'tomato')]:
        axes[0, i].hist(joined[joined['cancer'] == lbl][c], bins=40,
                        alpha=0.6, color=color, label=f'c={lbl}', density=True)
    axes[0, i].set_title(c, fontsize=9)
    axes[0, i].legend(fontsize=5)
    axes[1, i].boxplot(
        [joined[joined['cancer'] == 0][c].values,
         joined[joined['cancer'] == 1][c].values],
        labels=['c=0', 'c=1'], patch_artist=True,
        boxprops=dict(facecolor='lightblue')
    )
    axes[1, i].set_title(f'{c}\nboxplot', fontsize=7)
plt.suptitle('Bioquimicos - distribucion por cancer', fontsize=11)
plt.tight_layout()
plt.savefig(str(FIG / '02_bioquimicos_dist.png'), bbox_inches='tight')
plt.close()
print("fig02 saved")

# ── FIG 3: Genetics lift ──────────────────────────────────────────────────────
gen_feats = ['mut_BRCA1', 'mut_TP53', 'mut_EGFR', 'mut_KRAS', 'mut_PIK3CA', 'mut_ALK', 'mut_BRAF']
rate_mut   = [joined[joined[g] == 1]['cancer'].mean() for g in gen_feats]
rate_nomut = [joined[joined[g] == 0]['cancer'].mean() for g in gen_feats]
x = np.arange(len(gen_feats))
w = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w/2, [r * 100 for r in rate_nomut], w, label='Sin mutacion', color='steelblue')
ax.bar(x + w/2, [r * 100 for r in rate_mut],   w, label='Con mutacion', color='tomato')
ax.axhline(prev * 100, color='grey', linestyle='--', label=f'Prevalencia global {prev*100:.1f}%')
ax.set_xticks(x)
ax.set_xticklabels(gen_feats, rotation=30, ha='right')
ax.set_ylabel('P(cancer=1) %')
ax.set_title('Tasa de cancer por mutacion genetica')
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG / '03_geneticos_lift.png'), bbox_inches='tight')
plt.close()
print("fig03 saved")

# ── FIG 4: Clinical comorbidities lift ───────────────────────────────────────
clin_feats = ['diabetes', 'hipertension', 'obesidad', 'enfermedad_cardiaca', 'asma', 'epoc']
rates = [(joined[joined[c] == 1]['cancer'].mean(), joined[joined[c] == 0]['cancer'].mean())
         for c in clin_feats]
x = np.arange(len(clin_feats))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, [r[1] * 100 for r in rates], w, label='Sin comorbilidad', color='steelblue')
ax.bar(x + w/2, [r[0] * 100 for r in rates], w, label='Con comorbilidad', color='salmon')
ax.axhline(prev * 100, color='grey', linestyle='--', label=f'Prevalencia global {prev*100:.1f}%')
ax.set_xticks(x)
ax.set_xticklabels(clin_feats, rotation=20, ha='right')
ax.set_ylabel('P(cancer=1) %')
ax.set_title('Tasa de cancer por comorbilidad clinica')
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG / '04_clinicos_lift.png'), bbox_inches='tight')
plt.close()
print("fig04 saved")

# ── FIG 5: Categorical lift ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
cat_plot = [
    ('actividad_fisica', axes[0, 0]), ('fumador',         axes[0, 1]),
    ('tipo_seguro',      axes[0, 2]), ('nivel_educativo', axes[1, 0]),
    ('nivel_ingresos',   axes[1, 1]), ('zona',            axes[1, 2]),
]
for feat, ax in cat_plot:
    lift = joined.groupby(feat)['cancer'].mean().sort_values(ascending=False) * 100
    bar_colors = ['tomato' if v > prev * 100 * 1.05 else 'steelblue' for v in lift.values]
    bars = ax.bar(lift.index.astype(str), lift.values, color=bar_colors)
    ax.axhline(prev * 100, color='grey', linestyle='--', linewidth=1.2)
    ax.set_title(feat, fontsize=10)
    ax.set_ylabel('P(cancer=1) %')
    ax.tick_params(axis='x', rotation=25)
    for bar, val in zip(bars, lift.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', fontsize=8)
plt.suptitle('Lift de cancer por variables categoricas', fontsize=12)
plt.tight_layout()
plt.savefig(str(FIG / '05_categorical_lift.png'), bbox_inches='tight')
plt.close()
print("fig05 saved")

# ── FIG 6: Correlation heatmap (safe features) ───────────────────────────────
safe_num = [
    'glucosa', 'colesterol', 'trigliceridos', 'hemoglobina', 'leucocitos', 'plaquetas', 'creatinina',
    'mut_BRCA1', 'mut_TP53', 'mut_EGFR', 'mut_KRAS', 'mut_PIK3CA', 'mut_ALK', 'mut_BRAF',
    'fumador', 'edad', 'num_hijos', 'distancia_hospital_km', 'cancer',
]
corr_mat = joined[safe_num].corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, annot_kws={'size': 7}, linewidths=0.5)
ax.set_title('Correlaciones entre features seguras + target cancer', fontsize=12)
plt.tight_layout()
plt.savefig(str(FIG / '06_corr_heatmap.png'), bbox_inches='tight')
plt.close()
print("fig06 saved")

# ── FIG 7: Leakage warning ────────────────────────────────────────────────────
leakage_cols = ['coste_total', 'coste_farmaco', 'num_ingresos', 'dias_hospital', 'vive']
leakage_corr = [joined[c].corr(joined['cancer']) for c in leakage_cols]
fig, ax = plt.subplots(figsize=(8, 4))
bar_colors_lk = ['crimson' if abs(c) > 0.3 else 'orange' for c in leakage_corr]
bars = ax.barh(leakage_cols, leakage_corr, color=bar_colors_lk)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Pearson r con cancer')
ax.set_title('ADVERTENCIA: Variables sospechosas de leakage')
for bar, val in zip(bars, leakage_corr):
    ax.text(val + 0.008 * np.sign(val), bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(str(FIG / '07_leakage_warning.png'), bbox_inches='tight')
plt.close()
print("fig07 saved")

# ── FIG 8: Sociodemographic numerics ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ['edad', 'num_hijos', 'distancia_hospital_km']):
    for lbl, color in [(0, 'steelblue'), (1, 'tomato')]:
        ax.hist(joined[joined['cancer'] == lbl][col], bins=30, alpha=0.6, color=color,
                label=f'cancer={lbl}', density=True)
    ax.set_title(col)
    ax.legend(fontsize=7)
plt.suptitle('Variables sociodemograficas numericas por cancer', fontsize=11)
plt.tight_layout()
plt.savefig(str(FIG / '08_sociodem_num.png'), bbox_inches='tight')
plt.close()
print("fig08 saved")

# ── FIG 9: Economic leakage distribution ─────────────────────────────────────
eco_num = ['coste_total', 'coste_farmaco', 'num_ingresos', 'dias_hospital']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, c in enumerate(eco_num):
    for lbl, color in [(0, 'steelblue'), (1, 'tomato')]:
        axes[0, i].hist(joined[joined['cancer'] == lbl][c], bins=40, alpha=0.6, color=color,
                        label=f'c={lbl}', density=True)
    axes[0, i].set_title(f'{c}\n(LEAKAGE RISK)', fontsize=8, color='red')
    axes[0, i].legend(fontsize=6)
    axes[1, i].boxplot(
        [joined[joined['cancer'] == 0][c].values, joined[joined['cancer'] == 1][c].values],
        labels=['c=0', 'c=1'], patch_artist=True
    )
    axes[1, i].set_title(f'{c} boxplot', fontsize=8, color='red')
plt.suptitle('Variables economicas - ADVERTENCIA: leakage post-diagnostico', color='red', fontsize=11)
plt.tight_layout()
plt.savefig(str(FIG / '09_economicos_leakage.png'), bbox_inches='tight')
plt.close()
print("fig09 saved")

# ── FIG 10: Feature signal ranking (safe features only) ──────────────────────
numeric_cols = joined.select_dtypes('number').columns.tolist()
feature_num  = [c for c in numeric_cols if c != 'cancer']
corr_all     = pd.Series({c: joined[c].corr(joined['cancer']) for c in feature_num})
corr_all     = corr_all.sort_values(key=abs, ascending=False)
safe_only    = corr_all[[c for c in corr_all.index
                         if c not in ['coste_total', 'coste_farmaco', 'num_ingresos', 'dias_hospital', 'vive']]]
fig, ax = plt.subplots(figsize=(10, 8))
bar_colors2  = ['tomato' if v > 0 else 'steelblue' for v in safe_only.values[::-1]]
ax.barh(safe_only.index[::-1], safe_only.values[::-1], color=bar_colors2)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Pearson r con cancer')
ax.set_title('Senial lineal con cancer (sin variables leakage)')
plt.tight_layout()
plt.savefig(str(FIG / '10_feature_signal.png'), bbox_inches='tight')
plt.close()
print("fig10 saved")

# ── Compute stats for JSON report ─────────────────────────────────────────────
corr_dict  = corr_all.round(4).to_dict()
cat_lift   = {}
for feat in ['actividad_fisica', 'tipo_seguro', 'nivel_educativo', 'nivel_ingresos', 'zona', 'estado_civil']:
    cat_lift[feat] = joined.groupby(feat)['cancer'].mean().round(4).to_dict()

report = {
    "n_total": N,
    "n_positivos": int(n_pos),
    "n_negativos": int(n_neg),
    "prevalencia_cancer": round(prev, 4),
    "ratio_desbalance_1_a_N": round(ratio, 2),
    "fuentes": {
        name: {
            "filas": len(df),
            "columnas": len(df.columns),
            "paciente_id_unicos": int(df['paciente_id'].nunique()),
            "nulos": int(df.isnull().sum().sum()),
            "cobertura_pct": 100.0
        }
        for name, df in {
            n: pd.read_csv(RAW_DIR / f'{n}.csv')
            for n in ['bioquimicos', 'clinicos', 'geneticos', 'economicos', 'generales', 'sociodemografico']
        }.items()
    },
    "joined_shape": list(joined.shape),
    "nulos_joined": int(joined.isnull().sum().sum()),
    "correlaciones_con_cancer": corr_dict,
    "lift_categoricas": cat_lift,
    "columnas_constantes": ["alcohol (100% = 1)"],
    "columnas_leakage_confirmado": ["coste_total", "coste_farmaco", "num_ingresos", "dias_hospital", "vive"],
    "features_candidatas_seguras": [
        "glucosa", "colesterol", "trigliceridos", "hemoglobina", "leucocitos", "plaquetas", "creatinina",
        "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS", "mut_PIK3CA", "mut_ALK", "mut_BRAF",
        "fumador", "actividad_fisica", "edad"
    ],
    "features_valorar": [
        "diabetes", "hipertension", "obesidad", "enfermedad_cardiaca", "asma", "epoc",
        "nivel_educativo", "nivel_ingresos", "zona", "estado_civil", "num_hijos",
        "distancia_hospital_km", "tipo_seguro"
    ],
    "features_excluir": ["alcohol", "vive", "coste_total", "coste_farmaco", "num_ingresos", "dias_hospital"]
}

outpath = EDA_DIR / 'report.json'
with open(str(outpath), 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\nJSON report saved: {outpath}")

print(f"\nAll outputs complete.")
print(f"Figures: {[f.name for f in sorted(FIG.iterdir()) if f.suffix == '.png']}")
