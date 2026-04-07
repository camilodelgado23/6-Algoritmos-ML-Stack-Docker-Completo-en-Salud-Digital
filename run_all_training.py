"""
run_all_training.py — Entrena los 6 modelos, los compara y elige el mejor.
Ejecutar desde la raíz del proyecto:
  python run_all_training.py
"""

import subprocess, sys, json, os
from pathlib import Path

ROOT      = Path(__file__).parent
MODEL_DIR = os.getenv("MODEL_DIR", str(ROOT / "ai-service" / "models"))
TRAIN_DIR = ROOT / "training"

os.makedirs(MODEL_DIR, exist_ok=True)

scripts = [
    ("Decision Tree",        "train_dt.py"),
    ("KNN",                  "train_knn.py"),
    ("Gradient Boosting",    "train_gbm.py"),
    ("Logistic Regression",  "train_lr.py"),
    ("XGBoost",              "train_xgb.py"),
    ("TabPFN",               "train_tabpfn.py"),
]

print("=" * 60)
print("  SALUD DIGITAL IA — Entrenamiento completo 6 modelos")
print("  Dataset: Heart Disease UCI (id=45)")
print("=" * 60)

for name, script in scripts:
    print(f"\n{'─'*60}")
    print(f"  ▶  Entrenando: {name}")
    print(f"{'─'*60}")
    result = subprocess.run(
        [sys.executable, str(TRAIN_DIR / script)],
        env={**os.environ, "MODEL_DIR": MODEL_DIR}
    )
    if result.returncode != 0:
        print(f"  ⚠️  {name} falló con código {result.returncode}")

# ── Comparar resultados ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  COMPARACIÓN FINAL DE MODELOS")
print("=" * 60)

meta_files = {
    "Decision Tree":       "dt_meta.json",
    "KNN":                 "knn_meta.json",
    "Gradient Boosting":   "gbm_meta.json",
    "Logistic Regression": "lr_meta.json",
    "XGBoost":             "xgb_meta.json",
    "TabPFN":              "tabpfn_meta.json",
}

results = []
for name, fname in meta_files.items():
    path = Path(MODEL_DIR) / fname
    if path.exists():
        with open(path) as f:
            m = json.load(f)
        results.append({
            "algorithm": name,
            "f1":        m.get("f1", 0),
            "auc_roc":   m.get("auc_roc", 0),
            "accuracy":  m.get("accuracy", 0),
        })
    else:
        print(f"  ⚠️  No se encontró {fname}")

results.sort(key=lambda x: x["f1"], reverse=True)

print(f"\n{'Algoritmo':<22} {'F1':>8} {'AUC-ROC':>10} {'Accuracy':>10}")
print("-" * 54)
for i, r in enumerate(results):
    marker = " ◄ MEJOR" if i == 0 else ""
    print(f"  {r['algorithm']:<20} {r['f1']:>8.4f} {r['auc_roc']:>10.4f} {r['accuracy']:>10.4f}{marker}")

if results:
    winner = results[0]
    print(f"\n🏆  Modelo seleccionado: {winner['algorithm']}")
    print(f"   F1={winner['f1']:.4f}  AUC-ROC={winner['auc_roc']:.4f}  Acc={winner['accuracy']:.4f}")

    with open(f"{MODEL_DIR}/best_model.json", "w") as f:
        json.dump({"winner": winner["algorithm"], "metrics": winner}, f, indent=2)
    print(f"\n✅  Resultado guardado en {MODEL_DIR}/best_model.json")
    print(f"   Ahora puedes levantar el stack: docker-compose up --build")