"""
train_dt.py — Decision Tree
Dataset: Heart Disease (UCI id=45) — 303 pacientes, clasificación binaria
"""

import json
import time
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MODEL_DIR  = os.getenv("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Cargar dataset ──────────────────────────────────────────────────────────
print("📥  Descargando Heart Disease (UCI id=45)...")
repo = fetch_ucirepo(id=45)
X    = repo.data.features.copy()
y    = repo.data.targets.copy()

# Target: num > 0 → enfermedad (1), 0 → sano (0)
y = (y.squeeze() > 0).astype(int)

# Limpiar: rellenar NaN con mediana
X = X.apply(lambda col: col.fillna(col.median()) if col.dtype != object
            else col.fillna(col.mode()[0]))

# Codificar categóricas
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── MLflow ──────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("heart-disease-comparison")

best_params = {"max_depth": None, "min_samples_split": 2}
best_f1     = -1

print("🔍  Grid search Decision Tree...")
for depth in [3, 5, 7, None]:
    for split in [2, 5, 10]:
        with mlflow.start_run(run_name=f"DT_d{depth}_s{split}"):
            clf = DecisionTreeClassifier(
                max_depth=depth, min_samples_split=split, random_state=42
            )
            t0  = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - t0

            t1   = time.time()
            preds = clf.predict(X_test)
            infer_time = (time.time() - t1) / len(X_test) * 1000  # ms/sample

            proba = clf.predict_proba(X_test)[:, 1]
            f1    = f1_score(y_test, preds)
            auc   = roc_auc_score(y_test, proba)
            acc   = accuracy_score(y_test, preds)
            cv    = cross_val_score(clf, X, y, cv=5, scoring="f1").mean()

            mlflow.log_params({"max_depth": depth, "min_samples_split": split})
            mlflow.log_metrics({
                "f1": f1, "auc_roc": auc, "accuracy": acc,
                "cv_f1_mean": cv,
                "train_time_s": train_time,
                "infer_time_ms": infer_time
            })
            mlflow.sklearn.log_model(clf, "model")

            print(f"  depth={depth} split={split} → F1={f1:.4f} AUC={auc:.4f}")
            if f1 > best_f1:
                best_f1     = f1
                best_params = {"max_depth": depth, "min_samples_split": split}
                best_clf    = clf

# ── Guardar mejor modelo ────────────────────────────────────────────────────
print(f"\n✅  Mejor DT: {best_params}  F1={best_f1:.4f}")
with open(f"{MODEL_DIR}/dt_model.pkl", "wb") as f:
    pickle.dump(best_clf, f)

# Metadata
meta = {
    "algorithm": "Decision Tree",
    "dataset": "Heart Disease UCI (id=45)",
    "best_params": best_params,
    "f1":     float(f1_score(y_test, best_clf.predict(X_test))),
    "auc_roc": float(roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1])),
    "accuracy": float(accuracy_score(y_test, best_clf.predict(X_test))),
    "train_samples": int(len(X_train)),
    "test_samples":  int(len(X_test)),
    "features": list(X.columns),
    "report": classification_report(y_test, best_clf.predict(X_test)),
    "confusion_matrix": confusion_matrix(y_test, best_clf.predict(X_test)).tolist()
}
with open(f"{MODEL_DIR}/dt_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(json.dumps({k: v for k, v in meta.items() if k != "report"}, indent=2))
print("\n📊  Classification report:\n", meta["report"])