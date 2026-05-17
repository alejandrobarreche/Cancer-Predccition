# EDA — Prediccion de cancer

## 1. Resumen ejecutivo

- **N pacientes unicos:** 50 001
- **Prevalencia cancer=1:** 9 644 casos (19.29%)
- **Ratio de desbalance:** 1 : 4.18 (moderado; manejable con `class_weight`)
- **Nulos en el dataset unido:** 0 (ninguna fuente tiene missingness)
- **Cobertura de cada una de las 6 fuentes:**

| Fuente | Filas | Cols | IDs unicos | Cobertura |
|---|---:|---:|---:|---:|
| bioquimicos | 50 001 | 8 | 50 001 | 100% |
| clinicos (principal) | 50 001 | 8 | 50 001 | 100% |
| geneticos | 50 001 | 8 | 50 001 | 100% |
| economicos | 50 001 | 6 | 50 001 | 100% |
| generales | 50 001 | 5 | 50 001 | 100% |
| sociodemografico | 50 001 | 8 | 50 001 | 100% |

- **Shape del dataset unido:** 50 001 filas x 38 columnas (37 features + `paciente_id`)
- **IDs huerfanos:** ninguno; la interseccion de los 6 conjuntos es exactamente los mismos 50 001 IDs.

---

## 2. Decisiones de union

| Tabla | Logica | Motivo |
|---|---|---|
| `clinicos` | Tabla principal (left base) | Contiene la variable objetivo `cancer` |
| `bioquimicos` | Merge directo 1:1 por `paciente_id` | Un registro por paciente, sin ambiguedad |
| `geneticos` | Merge directo 1:1 | Un registro por paciente |
| `economicos` | Merge directo 1:1 | Un registro por paciente (coste total del episodio) |
| `generales` | Merge directo 1:1 | Habitos de vida, un registro por paciente |
| `sociodemografico` | Merge directo 1:1 | Perfil demografico, un registro por paciente |

Todas las tablas tienen exactamente una fila por `paciente_id` (no hay relaciones multi-fila), por lo que no fue necesario aplicar ningun tipo de agregacion. La operacion es `pd.merge(..., how='left', on='paciente_id')` iterada 5 veces, produciendo el mismo resultado que un `inner` o `outer` join dado que la cobertura es 100%.

---

## 3. Features candidatas (top 20)

| feature | tipo | fuente | missing % | r con cancer | senial | tratamiento propuesto |
|---|---|---|---:|---:|---|---|
| `mut_BRCA1` | binaria (0/1) | geneticos | 0% | +0.219 | alta | Usar tal cual; predictor causal directo |
| `fumador` | binaria (0/1) | generales | 0% | +0.217 | alta | Usar tal cual; factor de riesgo causal |
| `obesidad` | binaria (0/1) | clinicos | 0% | +0.198 | alta | Valorar; comorbilidad correlacionada por diseno |
| `mut_TP53` | binaria (0/1) | geneticos | 0% | +0.187 | alta | Usar tal cual; predictor causal directo |
| `mut_KRAS` | binaria (0/1) | geneticos | 0% | +0.167 | alta | Usar tal cual; predictor causal directo |
| `glucosa` | continua (mg/dL) | bioquimicos | 0% | +0.151 | media-alta | StandardScaler post-split |
| `actividad_fisica` | categorica (3 niveles) | generales | 0% | ~0.12 (lift) | media | OHE o codificacion ordinal (Baja=0, Mod=1, Alta=2) |
| `trigliceridos` | continua (mg/dL) | bioquimicos | 0% | +0.109 | media | StandardScaler post-split |
| `hipertension` | binaria (0/1) | clinicos | 0% | +0.101 | media | Valorar; comorbilidad correlacionada por diseno |
| `mut_EGFR` | binaria (0/1) | geneticos | 0% | +0.100 | media | Usar tal cual; predictor causal directo |
| `leucocitos` | continua (x10^3/uL) | bioquimicos | 0% | +0.098 | media | StandardScaler post-split |
| `colesterol` | continua (mg/dL) | bioquimicos | 0% | +0.087 | baja-media | StandardScaler post-split |
| `diabetes` | binaria (0/1) | clinicos | 0% | +0.076 | baja | Valorar; comorbilidad correlacionada por diseno |
| `hemoglobina` | continua (g/dL) | bioquimicos | 0% | -0.073 | baja | StandardScaler post-split; valor bajo asociado a cancer |
| `mut_PIK3CA` | binaria (0/1) | geneticos | 0% | +0.068 | baja | Usar tal cual |
| `edad` | entera (anos) | sociodemografico | 0% | +0.054 | baja | StandardScaler post-split; proxy riesgo acumulado |
| `mut_BRAF` | binaria (0/1) | geneticos | 0% | +0.048 | baja | Usar tal cual |
| `tipo_seguro` | categorica (3 niveles) | economicos | 0% | lift Privado=31.9% | baja (posible proxy) | OHE; con cautela (puede ser proxy de nivel socioeconomico) |
| `epoc` | binaria (0/1) | clinicos | 0% | +0.031 | muy baja | Opcional; puede aportar en modelos no lineales |
| `nivel_educativo` | categorica (4 niveles) | sociodemografico | 0% | ~0% (lift plano) | sin senial | Opcional; incluir por completitud pero sin expectativas |

---

## 4. Variables a EXCLUIR (leakage o sin informacion)

### Leakage post-diagnostico confirmado (diccionario de variables)

| Variable | r con cancer | Motivo de exclusion |
|---|:---:|---|
| `coste_total` | **0.891** | Coste del episodio asistencial — consecuencia del diagnostico |
| `dias_hospital` | **0.878** | Dias de hospitalizacion — consecuencia del diagnostico |
| `coste_farmaco` | **0.853** | Coste de medicacion — consecuencia del diagnostico |
| `num_ingresos` | **0.644** | Numero de ingresos — consecuencia del diagnostico |
| `vive` | **-0.354** | Supervivencia al cierre del seguimiento — resultado vital post-diagnostico |

La correlacion de coste_total con cancer (r=0.891) es inverosimil para un predictor honesto: confirma que estas variables reflejan consumo sanitario POSTERIOR al diagnostico. Su inclusion provocaria data leakage severo (accuracy artificialmente cercano al 100%).

### Sin varianza

| Variable | Motivo |
|---|---|
| `alcohol` | Constante: 100% de los pacientes tienen valor 1. Pearson indefinido. Sin informacion. |

---

## 5. Variables a VALORAR con cautela

Las comorbilidades clinicas (`diabetes`, `hipertension`, `obesidad`, `enfermedad_cardiaca`, `asma`, `epoc`) correlacionan con `cancer` por diseno del modelo generativo (ver diccionario). Esto no es leakage causal directo, pero implica que el modelo podria estar aprendiendo una asociacion artificial. El diccionario las marca como "Valorar". Recomendacion: **incluirlas en una primera ronda de modelado** y medir su importancia; si dominan el modelo de forma sospechosa, entrenar una version sin ellas como ablacion.

`tipo_seguro` muestra un lift notable (Privado: 31.9% vs Publico: 13.8%), pero puede ser un proxy del nivel socioeconomico o del acceso a pruebas diagnosticas, no una causa del cancer. Incluir como feature opcional y vigilar su importancia.

Las variables sociodemograficas restantes (`nivel_educativo`, `nivel_ingresos`, `zona`, `estado_civil`, `num_hijos`, `distancia_hospital_km`) no muestran senial lineal significativa (r < 0.01 para la mayoria). Se incluyen en el dataset unido para que los modelos no lineales puedan explotarlas, pero es improbable que aporten.

---

## 6. Analisis de missingness

No hay ningun valor nulo en ninguno de los 6 CSV ni en el dataset unido. El analisis de missingness es trivialmente nulo. No se requiere ninguna estrategia de imputacion.

Sin embargo, para robustez del pipeline (en caso de nuevos datos en produccion), se recomienda:
- Numericas: imputar con mediana de entrenamiento.
- Categoricas: imputar con moda de entrenamiento o categoria "Desconocido".
- Toda imputacion debe ajustarse solo sobre el conjunto de entrenamiento y aplicarse al test.

---

## 7. Analisis bivariante — hallazgos destacados

### Geneticas
- `mut_BRCA1`: P(cancer|mut=1) = 45.5% vs P(cancer|mut=0) = 17.5% — lift de 2.6x
- `mut_TP53`: P(cancer|mut=1) = 40.5% vs 16.6% — lift de 2.4x
- `mut_KRAS`: similar orden de magnitud
- Todas las mutaciones muestran lift positivo consistente con el modelo generativo del diccionario

### Habitos de vida
- `fumador`: P(cancer|fumador=1) = 29.4% vs P(cancer|fumador=0) = 12.5% — lift de 2.4x
- `actividad_fisica`: Baja=24.0%, Moderada=17.2%, Alta=12.3% — gradiente claro, factor protector confirmado

### Bioquimica
- `glucosa` muestra separacion visible en las distribuciones condicionadas (cancer=1 tiene mediana ligeramente mayor)
- `hemoglobina` muestra correlacion negativa debil: pacientes con cancer tienden a tener hemoglobina algo menor
- `leucocitos`: pacientes con cancer tienen distribucion ligeramente desplazada hacia valores altos (inflamacion cronica)

### Sociodemografia
- `edad`: correlacion baja (r=0.054) — sorprendente dado el diseno del modelo generativo que le asigna peso +0.4 para edad>55. La distribucion de edad es similar entre cancer=0 y cancer=1 en los histogramas.
- El resto de variables sociodemograficas no muestra senial practica.

---

## 8. Riesgos y warnings

1. **LEAKAGE GRAVE:** `coste_total` (r=0.891), `dias_hospital` (r=0.878), `coste_farmaco` (r=0.853), `num_ingresos` (r=0.644) deben EXCLUIRSE sin excepcion. Cualquier modelo que las incluya reportara metricas infladas (AUC cercano a 1.0) que no se traduciran a datos reales.

2. **LEAKAGE MODERADO:** `vive` (r=-0.354) es consecuencia del cancer, no su predictor. Excluir.

3. **COLUMNA CONSTANTE:** `alcohol` tiene varianza cero. Incluirla causaria problemas en regularizacion L1/L2 y en el StandardScaler. Excluir.

4. **COMORBILIDADES — leakage indirecto:** `obesidad`, `hipertension`, `diabetes` correlacionan con cancer por diseno del modelo generativo. No son leakage directo pero el modelo puede sobreajustarse a asociaciones artificiales. Monitorizar importancias.

5. **BAJA SENIAL EN SOCIODEMOGRAFIA:** La mayoria de variables sociodemograficas tienen r < 0.01. Su inclusion no dara al modelo senial util pero tampoco daniara — mantener en el dataset unido y dejar que los modelos lo confirmen.

6. **TIPO_SEGURO:** El lift de `Privado` (31.9% vs 13.8% en Publico) es llamativo. Puede reflejar sesgo de diagnostico (mayor acceso a pruebas) mas que riesgo real. No excluir, pero vigilar.

---

## 9. Proximos pasos

- **Split estratificado 80/20** con `stratify=cancer`, `random_state=42` (`src/features/split.py`)
- **Imputacion + encoding + escalado** ajustados sobre train y aplicados a test (`src/features/preprocess.py`):
  - OHE para `actividad_fisica`, `tipo_seguro`, `nivel_educativo`, `nivel_ingresos`, `zona`, `estado_civil`
  - StandardScaler para `glucosa`, `colesterol`, `trigliceridos`, `hemoglobina`, `leucocitos`, `plaquetas`, `creatinina`, `edad`, `num_hijos`, `distancia_hospital_km`
  - Binarias: sin transformacion
  - Excluir antes del pipeline: `alcohol`, `vive`, `coste_total`, `coste_farmaco`, `num_ingresos`, `dias_hospital`
- **Entrenar modelos clasicos** (LR, RF, XGBoost, LightGBM) con F1 clase positiva y AUC-ROC como metricas principales (`ml-classical`)
- **Disenar MLP** con `class_weight={0:1, 1:ratio}`, ajuste de umbral sobre validacion (`mlp-designer`)
