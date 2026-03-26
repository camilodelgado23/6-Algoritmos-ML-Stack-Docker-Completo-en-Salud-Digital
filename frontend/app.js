// ═══════════════════════════════════════════════════════════════
//  SALUD DIGITAL IA — app.js
//  Dashboard clínico: 6 modelos ML + FHIR + MLflow
// ═══════════════════════════════════════════════════════════════

const AI_URL   = window.AI_SERVICE_URL  || "http://localhost:8000";
const FHIR_URL = window.FHIR_BASE_URL   || "http://localhost:8080/fhir";

// ── Paleta de algoritmos ─────────────────────────────────────────────────────
const ALGO_COLORS = {
  dt:     { label:"Árbol",    color:"#00ff9d", icon:"🌳" },
  knn:    { label:"KNN",      color:"#ff2d55", icon:"📍" },
  gbm:    { label:"GBM",      color:"#ff9f0a", icon:"🚀" },
  lr:     { label:"Reg. Log.",color:"#5e5ce6", icon:"📈" },
  xgb:    { label:"XGBoost",  color:"#ffd60a", icon:"⚡" },
  tabpfn: { label:"TabPFN",   color:"#bf5af2", icon:"🧠" },
};

let activeTab   = "dt";
let latestPreds = null;
let patientData = null;
let radarChart  = null;
let barChart    = null;

// ── Render principal ─────────────────────────────────────────────────────────
function renderAlgoTabs() {
  const container = document.getElementById("algo-tabs");
  container.innerHTML = Object.entries(ALGO_COLORS).map(([k, v]) => `
    <button class="algo-tab ${k === activeTab ? "active" : ""}"
            data-key="${k}"
            style="--tab-color:${v.color}"
            onclick="switchTab('${k}')">
      <span class="tab-icon">${v.icon}</span>
      <span class="tab-label">${v.label}</span>
    </button>
  `).join("");
}

function switchTab(key) {
  activeTab = key;
  renderAlgoTabs();
  renderAlgoDetail(key);
}

// ── Teoría de cada algoritmo ─────────────────────────────────────────────────
const ALGO_THEORY = {
  dt: {
    use: "Ideal para problemas de clasificación cuando se necesita interpretabilidad. Supone que los datos pueden dividirse por umbrales en sus features.",
    formula: "Gini(t) = 1 − Σ p²ᵢ",
    pseudocode: `función árbol(X, y, profundidad):
  si profundidad == 0 o puro(y): retornar hoja(y)
  feature, umbral = mejor_split(X, y)  # minimiza Gini
  izq = árbol(X[X[f]<u], y[X[f]<u], prof−1)
  der = árbol(X[X[f]≥u], y[X[f]≥u], prof−1)
  retornar Nodo(feature, umbral, izq, der)`,
    pros: ["Alta interpretabilidad — reglas legibles", "No requiere normalización", "Rápido en inferencia"],
    cons: ["Propenso a overfitting sin poda", "Inestable ante pequeñas variaciones", "Baja precisión vs ensambles"],
    complexity: "O(n·m·log n) entrenamiento"
  },
  knn: {
    use: "Clasificación no paramétrica. Útil cuando los límites de decisión son complejos. Supone que ejemplos similares tienen la misma clase.",
    formula: "ŷ = mayoría({yᵢ : xᵢ ∈ Nₖ(x)})  donde d(x,xᵢ) = √Σ(xⱼ−xᵢⱼ)²",
    pseudocode: `función predecir(x, X_train, y_train, k):
  distancias = [euclidiana(x, xᵢ) para xᵢ en X_train]
  vecinos_k  = argsort(distancias)[:k]
  retornar moda(y_train[vecinos_k])`,
    pros: ["Simple de entender e implementar", "No asume forma del modelo", "Se adapta naturalmente a nuevos datos"],
    cons: ["Lento en inferencia O(n) — no escala", "Sensible a features irrelevantes", "Requiere normalización obligatoria"],
    complexity: "O(n·m) inferencia"
  },
  gbm: {
    use: "Ensamble que construye árboles secuenciales minimizando residuales. Excelente para datos tabulares con relaciones no lineales complejas.",
    formula: "F_m(x) = F_{m−1}(x) + η · h_m(x)  donde h_m = argmin L(y, F+h)",
    pseudocode: `F₀(x) = constante (log-odds)
para m = 1..M:
  rᵢ = −∂L(yᵢ, F(xᵢ))/∂F(xᵢ)  # pseudo-residuales
  hₘ = árbol_regresión(X, r)
  γ  = argmin Σ L(yᵢ, Fₘ₋₁(xᵢ) + γ·hₘ(xᵢ))
  Fₘ = Fₘ₋₁ + η·γ·hₘ`,
    pros: ["Alta precisión en datos tabulares", "Robusto ante outliers", "Feature importance nativa"],
    cons: ["Más lento de entrenar que RF", "Muchos hiperparámetros a tunear", "Riesgo de overfitting sin early stopping"],
    complexity: "O(M·n·log n) entrenamiento"
  },
  lr: {
    use: "Modelo lineal para clasificación binaria. Ideal como baseline interpretable. Supone independencia y linealidad entre features y log-odds.",
    formula: "P(y=1|x) = σ(wᵀx + b)  σ(z) = 1/(1+e⁻ᶻ)  L = −Σ[y·log(ŷ) + (1−y)·log(1−ŷ)]",
    pseudocode: `inicializar w=0, b=0
para cada epoch:
  ŷ = σ(Xw + b)
  grad_w = Xᵀ(ŷ−y)/n + λw  # L2 regularización
  grad_b = mean(ŷ−y)
  w -= α·grad_w
  b -= α·grad_b`,
    pros: ["Altamente interpretable — coeficientes = importancia", "Muy rápido en entrenamiento e inferencia", "Probabilidades bien calibradas"],
    cons: ["No captura relaciones no lineales", "Sensible a correlación entre features", "Requiere normalización y feature engineering"],
    complexity: "O(n·m·epochs) entrenamiento"
  },
  xgb: {
    use: "Implementación optimizada de Gradient Boosting con regularización L1/L2. Ganador frecuente en competencias de datos tabulares.",
    formula: "Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₖ)  Ω(f) = γT + ½λ‖w‖²",
    pseudocode: `para m = 1..M:
  gᵢ = ∂L(yᵢ,ŷᵢ)/∂ŷᵢ   # gradientes de 1er orden
  hᵢ = ∂²L(yᵢ,ŷᵢ)/∂ŷᵢ² # gradientes de 2do orden
  # Ganancia óptima del split:
  Gain = ½[GL²/(HL+λ) + GR²/(HR+λ) − (GL+GR)²/(HL+HR+λ)] − γ
  w*ⱼ = −Gⱼ/(Hⱼ+λ)  # pesos óptimos de hojas`,
    pros: ["Estado del arte en datos tabulares", "Regularización built-in evita overfitting", "Maneja NaN nativamente"],
    cons: ["Menos interpretable que DT/LR", "Más memoria que modelos simples", "Tiempo de entrenamiento moderado-alto"],
    complexity: "O(M·n·log n) con paralelismo por columna"
  },
  tabpfn: {
    use: "Red neuronal pre-entrenada sobre millones de datasets sintéticos. Predicción bayesiana exacta en una sola pasada. Ideal para datasets pequeños (<1000 muestras).",
    formula: "P(y|x, D_train) = ∫ P(y|x,θ)P(θ|D_train) dθ  ≈ Transformer(x, D_train)",
    pseudocode: `# Sin entrenamiento clásico — Prior-Fitted
modelo_base = Transformer preentrenado en P(D)
función predecir(x_test, D_train):
  # D_train es el contexto del Transformer
  entrada = [D_train_tokens, x_test_token]
  logits  = Transformer(entrada)
  retornar softmax(logits[-1])`,
    pros: ["Sin hiperparámetros que ajustar", "Excelente en datasets pequeños", "Inferencia bayesiana exacta"],
    cons: ["Solo funciona con <1000 muestras de entrenamiento", "Lento en inferencia (forward pass completo)", "Requiere GPU para velocidad óptima"],
    complexity: "O(n²·d) inferencia (atención cuadrática)"
  }
};

function renderAlgoDetail(key) {
  const t   = ALGO_THEORY[key];
  const col = ALGO_COLORS[key];
  const el  = document.getElementById("algo-detail");
  const pred = latestPreds && latestPreds[key];

  el.innerHTML = `
    <div class="algo-detail-header" style="border-color:${col.color}">
      <span class="algo-big-icon">${col.icon}</span>
      <div>
        <h2 style="color:${col.color}">${col.label}</h2>
        ${pred ? `
          <div class="pred-badge ${pred.prediction===1 ? 'risk-high' : 'risk-low'}">
            ${pred.risk_label}
          </div>
          <div class="pred-prob">Probabilidad: <strong>${(pred.probability*100).toFixed(1)}%</strong></div>
          <div class="pred-metrics">F1: ${pred.f1?.toFixed(4)||'—'}  |  AUC: ${pred.auc_roc?.toFixed(4)||'—'}</div>
        ` : ''}
      </div>
    </div>

    <div class="theory-grid">
      <div class="theory-card">
        <h4>🎯 Teoría de uso</h4>
        <p>${t.use}</p>
      </div>
      <div class="theory-card">
        <h4>📐 Fórmula clave</h4>
        <code class="formula">${t.formula}</code>
      </div>
    </div>

    <div class="theory-card full-width">
      <h4>🔧 Pseudocódigo</h4>
      <pre class="pseudocode">${t.pseudocode}</pre>
    </div>

    <div class="pros-cons-grid">
      <div class="theory-card pros">
        <h4>✅ Ventajas</h4>
        <ul>${t.pros.map(p => `<li>${p}</li>`).join("")}</ul>
      </div>
      <div class="theory-card cons">
        <h4>❌ Desventajas</h4>
        <ul>${t.cons.map(c => `<li>${c}</li>`).join("")}</ul>
      </div>
    </div>

    <div class="theory-card complexity-card">
      <h4>⏱️ Complejidad</h4>
      <code>${t.complexity}</code>
    </div>
  `;
}

// ── Predicción ───────────────────────────────────────────────────────────────
async function runPrediction() {
  const patientId = document.getElementById("patient-id").value.trim() || "P001";
  const fields    = ["age","sex","cp","trestbps","chol","fbs","restecg",
                     "thalach","exang","oldpeak","slope","ca","thal"];

  const payload = { patient_id: patientId };
  for (const f of fields) {
    const val = parseFloat(document.getElementById(f)?.value);
    if (isNaN(val)) { showToast(`Campo '${f}' inválido`, "error"); return; }
    payload[f] = val;
  }
  patientData = payload;

  const btn = document.getElementById("predict-btn");
  btn.disabled    = true;
  btn.textContent = "⏳ Analizando...";

  try {
    // Intentar con todos los modelos primero, si falla usar demo
    let data;
    try {
      const res = await fetch(`${AI_URL}/predict/all/${patientId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      data = await res.json();
    } catch (e1) {
      console.warn("Endpoint /predict/all falló, usando /predict/demo:", e1.message);
      const res2 = await fetch(`${AI_URL}/predict/demo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload)
      });
      if (!res2.ok) throw new Error("Servicio no disponible: " + e1.message);
      data = await res2.json();
      showToast("⚠️ Usando predicciones demo — entrena modelos para resultados reales", "warn");
    }

    latestPreds = data.predictions;
    renderResults(data);
    renderAlgoDetail(activeTab);
    updateCharts(data);

    document.getElementById("results-section").classList.remove("hidden");
    document.getElementById("charts-section").classList.remove("hidden");
    document.getElementById("fhir-badge").textContent =
      data.fhir_saved ? "✅ FHIR guardado" : "⚠️ FHIR offline";
    document.getElementById("fhir-badge").style.borderColor =
      data.fhir_saved ? "var(--green)" : "var(--orange)";

    document.getElementById("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
    showToast("✅ Análisis completado — " + Object.keys(data.predictions).length + " modelos", "ok");
  } catch (e) {
    showToast("Error: " + e.message, "error");
    console.error(e);
  } finally {
    btn.disabled    = false;
    btn.textContent = "🔬 Analizar Paciente";
  }
}

function renderResults(data) {
  const grid = document.getElementById("results-grid");
  grid.innerHTML = Object.entries(data.predictions).map(([k, p]) => {
    const col    = ALGO_COLORS[k] || { color:"#aaa", label: k, icon:"🤖" };
    const isBest = k === data.best_model;
    return `
      <div class="result-card ${p.prediction===1 ? 'high-risk' : 'low-risk'} ${isBest ? 'best-model' : ''}"
           style="--card-color:${col.color}"
           onclick="switchTab('${k}')">
        ${isBest ? '<span class="best-badge">🏆 MEJOR</span>' : ''}
        <div class="rc-header">
          <span class="rc-icon">${col.icon}</span>
          <span class="rc-name">${col.label}</span>
        </div>
        <div class="rc-risk ${p.prediction===1 ? 'risk-positive' : 'risk-negative'}">
          ${p.prediction===1 ? '⚠️ RIESGO' : '✅ SANO'}
        </div>
        <div class="rc-prob">
          <div class="prob-bar" style="width:${(p.probability*100).toFixed(0)}%; background:${col.color}"></div>
          <span>${(p.probability*100).toFixed(1)}%</span>
        </div>
        <div class="rc-metrics">
          <span>F1: ${p.f1?.toFixed(3)||'—'}</span>
          <span>AUC: ${p.auc_roc?.toFixed(3)||'—'}</span>
        </div>
      </div>
    `;
  }).join("");
}

// ── Gráficas comparativas ─────────────────────────────────────────────────────
function updateCharts(data) {
  const preds  = data.predictions;
  const keys   = Object.keys(preds);
  const colors = keys.map(k => ALGO_COLORS[k]?.color || "#aaa");
  const labels = keys.map(k => ALGO_COLORS[k]?.label || k);

  // Bar chart — métricas
  const ctx1 = document.getElementById("bar-chart").getContext("2d");
  if (barChart) barChart.destroy();
  barChart = new Chart(ctx1, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label:"F1",       data: keys.map(k => preds[k].f1 || 0),      backgroundColor: colors.map(c => c + "cc") },
        { label:"AUC-ROC",  data: keys.map(k => preds[k].auc_roc || 0), backgroundColor: colors.map(c => c + "66") },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color:"#e0e0e0" } } },
      scales: {
        x: { ticks: { color:"#aaa" }, grid: { color:"#333" } },
        y: { min: 0, max: 1, ticks: { color:"#aaa" }, grid: { color:"#333" } }
      }
    }
  });

  // Radar chart — probabilidades
  const ctx2 = document.getElementById("radar-chart").getContext("2d");
  if (radarChart) radarChart.destroy();
  radarChart = new Chart(ctx2, {
    type: "radar",
    data: {
      labels,
      datasets: [{
        label: "P(riesgo)",
        data:  keys.map(k => preds[k].probability),
        backgroundColor: "rgba(255,45,85,0.2)",
        borderColor:     "#ff2d55",
        pointBackgroundColor: colors,
        pointRadius: 6
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { r: { min:0, max:1, ticks:{ color:"#555", backdropColor:"transparent" }, grid:{ color:"#333" }, pointLabels:{ color:"#aaa" } } },
      plugins: { legend: { labels: { color:"#e0e0e0" } } }
    }
  });
}

// ── Cargar métricas desde API ─────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res  = await fetch(`${AI_URL}/metrics/comparison`);
    const data = await res.json();
    renderMetricsTable(data.models);
  } catch (e) {
    console.warn("Métricas no disponibles:", e.message);
    renderMetricsTable(null);
  }
}

function renderMetricsTable(models) {
  const tbody = document.getElementById("metrics-tbody");
  if (!models || models.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:#666">
      Ejecutar entrenamiento primero (run_all_training.py)</td></tr>`;
    return;
  }
  tbody.innerHTML = models.map((m, i) => {
    const col = Object.values(ALGO_COLORS).find(v => v.label === m.algorithm)?.color || "#aaa";
    return `<tr class="${m.best ? 'best-row' : ''}">
      <td>${i===0 ? "🏆" : ""}  <span style="color:${col}">${m.algorithm}</span></td>
      <td>${m.f1?.toFixed(4)||'—'}</td>
      <td>${m.auc_roc?.toFixed(4)||'—'}</td>
      <td>${m.accuracy?.toFixed(4)||'—'}</td>
      <td>${m.best ? '<span class="best-chip">✓ Seleccionado</span>' : ''}</td>
    </tr>`;
  }).join("");
}

// ── Toast notifications ───────────────────────────────────────────────────────
function showToast(msg, type = "ok") {
  const existing = document.getElementById("toast");
  if (existing) existing.remove();
  const colors = { ok: "#00ff9d", error: "#ff2d55", warn: "#ff9f0a" };
  const t = document.createElement("div");
  t.id = "toast";
  t.textContent = msg;
  Object.assign(t.style, {
    position: "fixed", bottom: "1.5rem", right: "1.5rem",
    background: "#0d1117", border: `1px solid ${colors[type]}`,
    color: colors[type], padding: ".6rem 1.2rem",
    borderRadius: "6px", fontFamily: "Space Mono, monospace",
    fontSize: ".75rem", zIndex: "9999",
    boxShadow: `0 0 16px ${colors[type]}44`,
    animation: "fadeInUp .25s ease",
  });
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// ── Cargar modelos disponibles ────────────────────────────────────────────────
async function checkServiceHealth() {
  try {
    const res  = await fetch(`${AI_URL}/health`);
    const data = await res.json();
    const el   = document.getElementById("service-status");
    const n    = data.models_loaded?.length || 0;
    el.textContent = `● ${n} modelo${n!==1?"s":""} listos | Mejor: ${data.best || "—"}`;
    el.style.color = n > 0 ? "#00ff9d" : "#ff9f0a";
  } catch {
    const el = document.getElementById("service-status");
    el.textContent = "● Servicio AI offline — modo demo activo";
    el.style.color = "#ff9f0a";
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  renderAlgoTabs();
  renderAlgoDetail(activeTab);
  loadMetrics();
  checkServiceHealth();

  // Datos de ejemplo (paciente con riesgo moderado — Heart Disease UCI)
  const demo = {
    age:63, sex:1, cp:3, trestbps:145, chol:233, fbs:1,
    restecg:0, thalach:150, exang:0, oldpeak:2.3, slope:0, ca:0, thal:1
  };
  for (const [k, v] of Object.entries(demo)) {
    const el = document.getElementById(k);
    if (el) el.value = v;
  }

  // Enter en cualquier campo lanza predicción
  document.querySelectorAll(".field-group input").forEach(inp => {
    inp.addEventListener("keydown", e => {
      if (e.key === "Enter") runPrediction();
    });
  });

  // Health check periódico cada 30s
  setInterval(checkServiceHealth, 30000);
  // Recargar métricas cada 60s (puede que el training termine mientras)
  setInterval(loadMetrics, 60000);
});

// CSS para animación del toast (inyectado dinámicamente)
const toastStyle = document.createElement("style");
toastStyle.textContent = `@keyframes fadeInUp {
  from { opacity:0; transform:translateY(10px); }
  to   { opacity:1; transform:translateY(0); }
}`;
document.head.appendChild(toastStyle);