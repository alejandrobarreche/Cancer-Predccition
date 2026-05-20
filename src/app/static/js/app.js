/* eslint-disable */
// Dashboard interactivity: Plotly chart configs + live predictor.

const PALETTE = {
  primary: "#0a2540",
  accent: "#0ea5e9",
  positive: "#10b981",
  warning: "#f59e0b",
  danger: "#ef4444",
  purple: "#8b5cf6",
  textSoft: "#475569",
  muted: "#94a3b8",
  border: "#e4e7ee",
};

const MODEL_COLORS = {
  mlp: PALETTE.primary,
  xgboost: PALETTE.accent,
  logistic_regression: PALETTE.danger,
  random_forest: PALETTE.warning,
  histgradientboosting: PALETTE.purple,
};

const PLOTLY_CONFIG = {
  displayModeBar: false,
  responsive: true,
};

const BASE_LAYOUT = {
  font: { family: "Inter, system-ui, sans-serif", color: "#0f172a", size: 12 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  margin: { l: 56, r: 24, t: 32, b: 48 },
  xaxis: { gridcolor: "#eef0f6", linecolor: PALETTE.border, zeroline: false, ticks: "outside", tickcolor: PALETTE.border, automargin: true },
  yaxis: { gridcolor: "#eef0f6", linecolor: PALETTE.border, zeroline: false, ticks: "outside", tickcolor: PALETTE.border, automargin: true },
  legend: { orientation: "h", y: -0.22, x: 0, font: { size: 11 } },
  hoverlabel: { bgcolor: "white", bordercolor: PALETTE.border, font: { family: "Inter", size: 12 } },
};

function layout(extra = {}) {
  const base = JSON.parse(JSON.stringify(BASE_LAYOUT));
  return Object.assign(base, extra, {
    xaxis: Object.assign({}, base.xaxis, extra.xaxis || {}),
    yaxis: Object.assign({}, base.yaxis, extra.yaxis || {}),
  });
}

// ─── 1. ROC overlay ───────────────────────────────────────────────────────────
function renderROCOverlay(curves) {
  const traces = Object.entries(curves).map(([key, c]) => ({
    x: c.fpr,
    y: c.tpr,
    type: "scatter",
    mode: "lines",
    name: `${c.label} (AUC=${c.auc_roc.toFixed(4)})`,
    line: { color: MODEL_COLORS[key] || PALETTE.muted, width: key === "mlp" ? 3 : 2 },
    hovertemplate: "FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra>%{fullData.name}</extra>",
  }));
  traces.push({
    x: [0, 1], y: [0, 1],
    type: "scatter", mode: "lines",
    name: "Aleatorio",
    line: { color: PALETTE.muted, width: 1, dash: "dot" },
    hoverinfo: "skip", showlegend: false,
  });

  Plotly.newPlot("plot-roc-overlay", traces, layout({
    xaxis: { title: "Tasa de falsos positivos (FPR)", range: [0, 1] },
    yaxis: { title: "Tasa de verdaderos positivos (TPR)", range: [0, 1.02] },
  }), PLOTLY_CONFIG);
}

// ─── 2. PR overlay ────────────────────────────────────────────────────────────
function renderPROverlay(curves) {
  const traces = Object.entries(curves).map(([key, c]) => ({
    x: c.recall,
    y: c.precision,
    type: "scatter",
    mode: "lines",
    name: `${c.label} (AP=${c.ap.toFixed(4)})`,
    line: { color: MODEL_COLORS[key] || PALETTE.muted, width: key === "mlp" ? 3 : 2 },
    hovertemplate: "Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra>%{fullData.name}</extra>",
  }));
  Plotly.newPlot("plot-pr-overlay", traces, layout({
    xaxis: { title: "Recall (sensibilidad)", range: [0, 1] },
    yaxis: { title: "Precisión", range: [0, 1] },
  }), PLOTLY_CONFIG);
}

// ─── 3. MLP ROC standalone ────────────────────────────────────────────────────
function renderMLPRoc(curves) {
  const c = curves.mlp;
  Plotly.newPlot("plot-mlp-roc", [
    {
      x: c.fpr, y: c.tpr, type: "scatter", mode: "lines",
      name: `MLP (AUC=${c.auc_roc.toFixed(4)})`,
      line: { color: PALETTE.primary, width: 3 },
      fill: "tozeroy", fillcolor: "rgba(10, 37, 64, 0.06)",
      hovertemplate: "FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>",
    },
    {
      x: [0, 1], y: [0, 1], type: "scatter", mode: "lines",
      line: { color: PALETTE.muted, width: 1, dash: "dot" },
      hoverinfo: "skip", showlegend: false,
    },
  ], layout({
    xaxis: { title: "FPR", range: [0, 1] },
    yaxis: { title: "TPR", range: [0, 1.02] },
    showlegend: false,
  }), PLOTLY_CONFIG);
}

// ─── 4. MLP PR standalone ─────────────────────────────────────────────────────
function renderMLPPr(curves) {
  const c = curves.mlp;
  Plotly.newPlot("plot-mlp-pr", [
    {
      x: c.recall, y: c.precision, type: "scatter", mode: "lines",
      line: { color: PALETTE.accent, width: 3 },
      fill: "tozeroy", fillcolor: "rgba(14, 165, 233, 0.08)",
      hovertemplate: "Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>",
    },
  ], layout({
    xaxis: { title: "Recall", range: [0, 1] },
    yaxis: { title: "Precisión", range: [0, 1] },
    showlegend: false,
  }), PLOTLY_CONFIG);
}

// ─── 5. Training curves (loss + AUC) ─────────────────────────────────────────
function renderTrainingCurves(history) {
  if (!history || !history.loss) return;
  const epochs = history.loss.map((_, i) => i + 1);
  const traces = [
    { x: epochs, y: history.loss, type: "scatter", mode: "lines", name: "Pérdida (train)", line: { color: PALETTE.primary, width: 2 } },
    { x: epochs, y: history.val_loss, type: "scatter", mode: "lines", name: "Pérdida (val)", line: { color: PALETTE.accent, width: 2 } },
  ];
  Plotly.newPlot("plot-training-loss", traces, layout({
    xaxis: { title: "Época" },
    yaxis: { title: "Loss" },
  }), PLOTLY_CONFIG);

  if (history.auc) {
    const auc_traces = [
      { x: epochs, y: history.auc, type: "scatter", mode: "lines", name: "AUC (train)", line: { color: PALETTE.primary, width: 2 } },
      { x: epochs, y: history.val_auc, type: "scatter", mode: "lines", name: "AUC (val)", line: { color: PALETTE.accent, width: 2 } },
    ];
    Plotly.newPlot("plot-training-auc", auc_traces, layout({
      xaxis: { title: "Época" },
      yaxis: { title: "AUC-ROC", range: [0.5, 1] },
    }), PLOTLY_CONFIG);
  }
}

// ─── 6. Comparison bar chart ─────────────────────────────────────────────────
function renderComparisonBars(rows) {
  const names = rows.map(r => r.name);
  const f1 = rows.map(r => r.f1);
  const auc = rows.map(r => r.auc);
  const colors = rows.map(r => r.key === "mlp" ? PALETTE.primary : PALETTE.accent);

  Plotly.newPlot("plot-comparison-bars", [
    {
      x: names, y: f1, type: "bar",
      name: "F1 (umbral óptimo)",
      marker: { color: colors, opacity: 0.92 },
      text: f1.map(v => v.toFixed(4)),
      textposition: "outside",
      textfont: { size: 11, color: PALETTE.textSoft },
      hovertemplate: "%{x}<br>F1 = %{y:.4f}<extra></extra>",
    },
    {
      x: names, y: auc, type: "bar",
      name: "AUC-ROC",
      marker: { color: colors, opacity: 0.4 },
      text: auc.map(v => v.toFixed(4)),
      textposition: "outside",
      textfont: { size: 11, color: PALETTE.textSoft },
      hovertemplate: "%{x}<br>AUC = %{y:.4f}<extra></extra>",
    },
  ], layout({
    xaxis: { tickangle: -15 },
    yaxis: { range: [0, 1], title: "Puntuación" },
    barmode: "group",
    bargap: 0.32,
  }), PLOTLY_CONFIG);
}

// ─── 7. Distribution explorer ────────────────────────────────────────────────
function renderDistribution(featureKey, allDistributions) {
  const d = allDistributions[featureKey];
  if (!d) return;
  Plotly.newPlot("plot-distribution", [
    {
      x: d.centers, y: d.hist_negative, type: "scatter", mode: "lines",
      name: "Sin diagnóstico",
      line: { color: PALETTE.accent, width: 2.5, shape: "spline" },
      fill: "tozeroy", fillcolor: "rgba(14, 165, 233, 0.18)",
      hovertemplate: "%{x:.2f}<br>densidad %{y:.4f}<extra>Sin cáncer</extra>",
    },
    {
      x: d.centers, y: d.hist_positive, type: "scatter", mode: "lines",
      name: "Con diagnóstico",
      line: { color: PALETTE.danger, width: 2.5, shape: "spline" },
      fill: "tozeroy", fillcolor: "rgba(239, 68, 68, 0.20)",
      hovertemplate: "%{x:.2f}<br>densidad %{y:.4f}<extra>Cáncer</extra>",
    },
  ], layout({
    xaxis: { title: d.label },
    yaxis: { title: "Densidad" },
    showlegend: false,
    shapes: [
      { type: "line", x0: d.median_negative, x1: d.median_negative, yref: "paper", y0: 0, y1: 1, line: { color: PALETTE.accent, dash: "dot", width: 1.5 } },
      { type: "line", x0: d.median_positive, x1: d.median_positive, yref: "paper", y0: 0, y1: 1, line: { color: PALETTE.danger, dash: "dot", width: 1.5 } },
    ],
    annotations: [
      { x: d.median_negative, xref: "x", yref: "paper", y: 1.02, text: `mediana negativos: ${d.median_negative.toFixed(2)}`, showarrow: false, font: { size: 10, color: PALETTE.accent }, xanchor: "left" },
      { x: d.median_positive, xref: "x", yref: "paper", y: 0.92, text: `mediana positivos: ${d.median_positive.toFixed(2)}`, showarrow: false, font: { size: 10, color: PALETTE.danger }, xanchor: "left" },
    ],
  }), PLOTLY_CONFIG);
}

// ─── 8. Predictor gauge ──────────────────────────────────────────────────────
function renderGauge(proba, threshold) {
  const pctValue = proba * 100;
  const data = [{
    type: "indicator",
    mode: "gauge+number",
    value: pctValue,
    number: { suffix: "%", font: { family: "Outfit, Inter, sans-serif", size: 48, color: PALETTE.primary } },
    gauge: {
      axis: { range: [0, 100], tickwidth: 1, tickcolor: PALETTE.border, tickfont: { size: 10 } },
      bar: { color: PALETTE.primary, thickness: 0.28 },
      bgcolor: "white",
      borderwidth: 0,
      steps: [
        { range: [0, 20], color: "#d1fae5" },
        { range: [20, 40], color: "#bbf7d0" },
        { range: [40, threshold * 100], color: "#fef3c7" },
        { range: [threshold * 100, 85], color: "#fed7aa" },
        { range: [85, 100], color: "#fecaca" },
      ],
      threshold: {
        line: { color: PALETTE.danger, width: 4 },
        thickness: 0.78,
        value: threshold * 100,
      },
    },
  }];
  Plotly.newPlot("gauge-mlp", data, {
    height: 240,
    margin: { l: 20, r: 20, t: 30, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, sans-serif" },
  }, PLOTLY_CONFIG);
}

// ─── 9. Predictor form submission ────────────────────────────────────────────
function gatherFormPayload(form) {
  const data = {};
  new FormData(form).forEach((value, key) => {
    data[key] = value;
  });
  // numeric fields
  const numericFields = ["glucosa", "colesterol", "trigliceridos", "hemoglobina",
    "leucocitos", "plaquetas", "creatinina", "edad", "num_hijos", "distancia_hospital_km"];
  numericFields.forEach(k => { data[k] = parseFloat(data[k]); });
  // checkboxes: convert presence to 1, absence to 0
  const checkboxFields = ["diabetes", "hipertension", "obesidad", "enfermedad_cardiaca",
    "asma", "epoc", "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS", "mut_PIK3CA",
    "mut_ALK", "mut_BRAF", "fumador"];
  checkboxFields.forEach(k => { data[k] = form.elements[k].checked ? 1 : 0; });
  data["num_hijos"] = parseInt(data["num_hijos"], 10);
  return data;
}

async function handlePredict(event) {
  event.preventDefault();
  const form = event.target;
  const button = form.querySelector("button[type=submit]");
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = "Calculando…";

  try {
    const payload = gatherFormPayload(form);
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Error en la predicción");
    }
    const result = await response.json();
    renderResult(result);
  } catch (err) {
    document.getElementById("result-panel").innerHTML =
      `<div style="color:var(--danger);padding:20px;text-align:center"><strong>Error</strong><br><span style="font-size:13px">${err.message}</span></div>`;
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

function renderResult(result) {
  const { predictions, band, derived_features } = result;
  const mlp = predictions.mlp;
  const xgb = predictions.xgboost;

  const panel = document.getElementById("result-panel");
  panel.classList.remove("empty");
  panel.innerHTML = `
    <div class="gauge-wrap"><div id="gauge-mlp" style="width:100%;max-width:340px"></div></div>
    <div class="risk-band ${band.color}">
      <div class="band-label">Banda de riesgo</div>
      <div class="band-value">${band.label}</div>
    </div>
    <div class="model-row">
      <div class="model-card">
        <div class="mc-label">MLP · umbral ${mlp.threshold.toFixed(2)}</div>
        <div class="mc-proba">${(mlp.proba * 100).toFixed(1)}<span style="font-size:18px;color:var(--muted)">%</span></div>
        <span class="mc-decision ${mlp.positive ? 'positive' : 'negative'}">${mlp.positive ? 'Positivo' : 'Negativo'}</span>
      </div>
      <div class="model-card">
        <div class="mc-label">XGBoost · umbral ${xgb.threshold.toFixed(2)}</div>
        <div class="mc-proba">${(xgb.proba * 100).toFixed(1)}<span style="font-size:18px;color:var(--muted)">%</span></div>
        <span class="mc-decision ${xgb.positive ? 'positive' : 'negative'}">${xgb.positive ? 'Positivo' : 'Negativo'}</span>
      </div>
    </div>
    <div class="derived-features">
      <h5>Marcadores clínicos detectados</h5>
      <div class="derived-flags">
        <span class="flag ${derived_features.glucosa_alta ? 'active' : ''}">Glucosa &gt; 130</span>
        <span class="flag ${derived_features.hemoglobina_baja ? 'active' : ''}">Hemoglobina &lt; 11</span>
        <span class="flag ${derived_features.leucocitos_altos ? 'active' : ''}">Leucocitos &gt; 10</span>
        <span class="flag ${derived_features.edad_riesgo ? 'active' : ''}">Edad &gt; 55</span>
      </div>
      <div class="derived-counters">
        <div class="dc"><span>Mutaciones totales:</span><span class="dc-val">${derived_features.n_mutaciones_total}/7</span></div>
        <div class="dc"><span>Alto riesgo:</span><span class="dc-val">${derived_features.n_mutaciones_alto_riesgo}/3</span></div>
      </div>
    </div>
    <div style="margin-top:22px;padding-top:18px;border-top:1px solid var(--border);font-size:12px;color:var(--muted);text-align:center;line-height:1.5">
      Resultado orientativo con fines académicos.<br>No constituye diagnóstico médico.
    </div>
  `;
  renderGauge(mlp.proba, mlp.threshold);
}

// ─── 10. Boot ────────────────────────────────────────────────────────────────
function boot() {
  const data = window.__DATA__;
  if (!data) return;

  renderROCOverlay(data.curves);
  renderPROverlay(data.curves);
  renderMLPRoc(data.curves);
  renderMLPPr(data.curves);
  renderTrainingCurves(data.history);
  renderComparisonBars(data.comparison);

  const featSelect = document.getElementById("feature-select");
  if (featSelect) {
    renderDistribution(featSelect.value, data.distributions);
    featSelect.addEventListener("change", e => {
      renderDistribution(e.target.value, data.distributions);
    });
  }

  const form = document.getElementById("predictor-form");
  if (form) form.addEventListener("submit", handlePredict);

  // Smooth scroll for nav links
  document.querySelectorAll('.nav-links a[href^="#"]').forEach(link => {
    link.addEventListener("click", e => {
      const target = document.querySelector(link.getAttribute("href"));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
