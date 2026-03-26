"""
train_lr.py — Logistic Regression
Dataset: Heart Disease (UCI id=45)
"""

import json, time, pickle, os
import mlflow, mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_DIR  = os.getenv("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("📥  Descargando Heart Disease (UCI id=45)...")
repo = fetch_ucirepo(id=45)
X    = repo.data.features.copy()
y    = (repo.data.targets.squeeze() > 0).astype(int)

X = X.apply(lambda col: col.fillna(col.median()) if col.dtype != object
            else col.fillna(col.mode()[0]))
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("heart-disease-comparison")

best_f1, best_clf, best_params = -1, None, {}

print("🔍  Grid search Logistic Regression...")
for C in [0.01, 0.1, 1.0, 10.0]:
    for solver in ["lbfgs", "saga"]:
        for penalty in ["l2"]:
            with mlflow.start_run(run_name=f"LR_C{C}_{solver}"):
                clf = LogisticRegression(
                    C=C, solver=solver, penalty=penalty,
                    max_iter=1000, random_state=42
                )
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
                cv    = cross_val_score(clf, X_sc, y, cv=5, scoring="f1").mean()

                mlflow.log_params({"C": C, "solver": solver, "penalty": penalty})
                mlflow.log_metrics({"f1": f1, "auc_roc": auc, "accuracy": acc,
                                    "cv_f1_mean": cv, "train_time_s": train_time,
                                    "infer_time_ms": infer_time})
                mlflow.sklearn.log_model(clf, "model")

                print(f"  C={C} solver={solver} → F1={f1:.4f} AUC={auc:.4f}")
                if f1 > best_f1:
                    best_f1, best_clf = f1, clf
                    best_params = {"C": C, "solver": solver}

print(f"\n✅  Mejor LR: {best_params}  F1={best_f1:.4f}")
bundle = {"model": best_clf, "scaler": scaler}
with open(f"{MODEL_DIR}/lr_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

meta = {
    "algorithm": "Logistic Regression",
    "dataset": "Heart Disease UCI (id=45)",
    "best_params": best_params,
    "f1":      float(f1_score(y_test, best_clf.predict(X_test))),
    "auc_roc": float(roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1])),
    "accuracy":float(accuracy_score(y_test, best_clf.predict(X_test))),
    "needs_scaling": True,
    "features": list(repo.data.features.columns),
    "report": classification_report(y_test, best_clf.predict(X_test)),
    "confusion_matrix": confusion_matrix(y_test, best_clf.predict(X_test)).tolist()
}
with open(f"{MODEL_DIR}/lr_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps({k: v for k, v in meta.items() if k != "report"}, indent=2))