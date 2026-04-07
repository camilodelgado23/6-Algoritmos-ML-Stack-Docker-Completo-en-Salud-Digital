"""
train_tabpfn.py — TabPFN (Prior-Fitted Networks)
Dataset: Heart Disease (UCI id=45)
"""

import json, time, pickle, os
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MODEL_DIR  = os.getenv("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
    print("✅  TabPFN disponible")
except ImportError:
    TABPFN_AVAILABLE = False
    print("⚠️   TabPFN no instalado — usando fallback RandomForest para demo")
    from sklearn.ensemble import RandomForestClassifier

print("📥  Descargando Heart Disease (UCI id=45)...")
repo = fetch_ucirepo(id=45)
X    = repo.data.features.copy()
y    = (repo.data.targets.squeeze() > 0).astype(int)

X = X.apply(lambda col: col.fillna(col.median()) if col.dtype != object
            else col.fillna(col.mode()[0]))
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X_np = X.values.astype(np.float32)
y_np = y.values

X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("heart-disease-comparison")

with mlflow.start_run(run_name="TabPFN"):
    if TABPFN_AVAILABLE:
        # TabPFN: N_ensemble_configurations controla la cantidad de forward passes
        clf = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        print("   (usando RandomForest como proxy TabPFN)")

    t0  = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t1    = time.time()
    preds = clf.predict(X_test)
    infer_time = (time.time() - t1) / len(X_test) * 1000

    proba = clf.predict_proba(X_test)[:, 1]
    f1    = f1_score(y_test, preds)
    auc   = roc_auc_score(y_test, proba)
    acc   = accuracy_score(y_test, preds)

    mlflow.log_params({"algorithm": "TabPFN", "tabpfn_available": TABPFN_AVAILABLE})
    mlflow.log_metrics({"f1": f1, "auc_roc": auc, "accuracy": acc,
                        "train_time_s": train_time, "infer_time_ms": infer_time})

    print(f"\n✅  TabPFN → F1={f1:.4f} AUC={auc:.4f} Acc={acc:.4f}")
    print(f"   Train: {train_time:.2f}s  Infer: {infer_time:.4f}ms/sample")

with open(f"{MODEL_DIR}/tabpfn_model.pkl", "wb") as f:
    pickle.dump(clf, f)

meta = {
    "algorithm": "TabPFN" if TABPFN_AVAILABLE else "TabPFN (RF fallback)",
    "dataset": "Heart Disease UCI (id=45)",
    "tabpfn_native": TABPFN_AVAILABLE,
    "f1":      float(f1),
    "auc_roc": float(auc),
    "accuracy":float(acc),
    "train_time_s":  float(train_time),
    "infer_time_ms": float(infer_time),
    "features": list(X.columns),
    "report": classification_report(y_test, preds),
    "confusion_matrix": confusion_matrix(y_test, preds).tolist()
}
with open(f"{MODEL_DIR}/tabpfn_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps({k: v for k, v in meta.items() if k != "report"}, indent=2))
print("\n📊  Classification report:\n", meta["report"])