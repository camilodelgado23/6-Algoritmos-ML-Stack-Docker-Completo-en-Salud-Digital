"""
train_gbm.py — Gradient Boosting Machine
Dataset: Heart Disease (UCI id=45)
"""

import json, time, pickle, os
import mlflow, mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("heart-disease-comparison")

best_f1, best_clf, best_params = -1, None, {}

print("🔍  Grid search GBM...")
for n_est in [100, 200]:
    for lr in [0.05, 0.1, 0.2]:
        for depth in [3, 5]:
            with mlflow.start_run(run_name=f"GBM_n{n_est}_lr{lr}_d{depth}"):
                clf = GradientBoostingClassifier(
                    n_estimators=n_est, learning_rate=lr,
                    max_depth=depth, random_state=42
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
                cv    = cross_val_score(clf, X, y, cv=5, scoring="f1").mean()

                mlflow.log_params({"n_estimators": n_est, "learning_rate": lr, "max_depth": depth})
                mlflow.log_metrics({"f1": f1, "auc_roc": auc, "accuracy": acc,
                                    "cv_f1_mean": cv, "train_time_s": train_time,
                                    "infer_time_ms": infer_time})
                mlflow.sklearn.log_model(clf, "model")

                print(f"  n={n_est} lr={lr} d={depth} → F1={f1:.4f} AUC={auc:.4f}")
                if f1 > best_f1:
                    best_f1, best_clf = f1, clf
                    best_params = {"n_estimators": n_est, "learning_rate": lr, "max_depth": depth}

print(f"\n✅  Mejor GBM: {best_params}  F1={best_f1:.4f}")
with open(f"{MODEL_DIR}/gbm_model.pkl", "wb") as f:
    pickle.dump(best_clf, f)

meta = {
    "algorithm": "Gradient Boosting",
    "dataset": "Heart Disease UCI (id=45)",
    "best_params": best_params,
    "f1":      float(f1_score(y_test, best_clf.predict(X_test))),
    "auc_roc": float(roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1])),
    "accuracy":float(accuracy_score(y_test, best_clf.predict(X_test))),
    "features": list(X.columns),
    "report": classification_report(y_test, best_clf.predict(X_test)),
    "confusion_matrix": confusion_matrix(y_test, best_clf.predict(X_test)).tolist()
}
with open(f"{MODEL_DIR}/gbm_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps({k: v for k, v in meta.items() if k != "report"}, indent=2))