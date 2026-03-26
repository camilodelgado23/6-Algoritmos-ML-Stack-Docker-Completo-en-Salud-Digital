"""
main.py — FastAPI AI Service
Sirve 6 modelos ML + integración FHIR RiskAssessment
Dataset: Heart Disease UCI (id=45)
"""

from __future__ import annotations
import json, os, pickle, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import mlflow
import mlflow.sklearn
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Config ───────────────────────────────────────────────────────────────────
FHIR_BASE    = os.getenv("FHIR_BASE_URL", "http://fhir-server:8080/fhir")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_DIR    = Path(os.getenv("MODEL_DIR", "/app/models"))
mlflow.set_tracking_uri(MLFLOW_URI)

app = FastAPI(
    title="Salud Digital IA — ML Service",
    description="6 modelos ML sobre Heart Disease UCI con integración FHIR R4",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Feature names (Heart Disease UCI id=45) ──────────────────────────────────
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# ── Carga de modelos ─────────────────────────────────────────────────────────
MODELS: Dict[str, Any] = {}

def load_models():
    """Carga todos los .pkl disponibles en MODEL_DIR."""
    model_map = {
        "dt":     ("dt_model.pkl",     "dt_meta.json"),
        "knn":    ("knn_model.pkl",    "knn_meta.json"),
        "gbm":    ("gbm_model.pkl",    "gbm_meta.json"),
        "lr":     ("lr_model.pkl",     "lr_meta.json"),
        "xgb":    ("xgb_model.pkl",    "xgb_meta.json"),
        "tabpfn": ("tabpfn_model.pkl", "tabpfn_meta.json"),
    }
    for key, (model_file, meta_file) in model_map.items():
        mp = MODEL_DIR / model_file
        mm = MODEL_DIR / meta_file
        if mp.exists():
            with open(mp, "rb") as f:
                obj = pickle.load(f)
            meta = {}
            if mm.exists():
                with open(mm) as f:
                    meta = json.load(f)
            MODELS[key] = {"obj": obj, "meta": meta}
            print(f"  ✅  Modelo '{key}' cargado")
        else:
            print(f"  ⚠️  Modelo '{key}' no encontrado ({mp})")

load_models()

# Determinar mejor modelo
best_model_path = MODEL_DIR / "best_model.json"
BEST_MODEL_KEY  = "xgb"  # default
if best_model_path.exists():
    with open(best_model_path) as f:
        bm = json.load(f)
    name_map = {
        "Decision Tree": "dt", "KNN": "knn",
        "Gradient Boosting": "gbm", "Logistic Regression": "lr",
        "XGBoost": "xgb", "TabPFN": "tabpfn"
    }
    BEST_MODEL_KEY = name_map.get(bm.get("winner", "XGBoost"), "xgb")
    print(f"  🏆  Mejor modelo: {BEST_MODEL_KEY} ({bm.get('winner')})")

# ── Schemas ──────────────────────────────────────────────────────────────────
class PatientFeatures(BaseModel):
    age:      float = Field(..., description="Edad en años")
    sex:      float = Field(..., description="Sexo (1=M, 0=F)")
    cp:       float = Field(..., description="Tipo dolor pecho (0-3)")
    trestbps: float = Field(..., description="Presión arterial reposo (mmHg)")
    chol:     float = Field(..., description="Colesterol sérico (mg/dl)")
    fbs:      float = Field(..., description="Glucosa en ayunas > 120 (1=T,0=F)")
    restecg:  float = Field(..., description="Resultados ECG reposo (0-2)")
    thalach:  float = Field(..., description="Frecuencia cardíaca máxima")
    exang:    float = Field(..., description="Angina inducida ejercicio (1=S,0=N)")
    oldpeak:  float = Field(..., description="Depresión ST ejercicio vs reposo")
    slope:    float = Field(..., description="Pendiente segmento ST (0-2)")
    ca:       float = Field(..., description="Vasos coloreados fluoroscopía (0-3)")
    thal:     float = Field(..., description="Talasemia (1=normal,2=fijo,3=reversible)")
    patient_id: Optional[str] = Field(None, description="ID opcional del paciente")

class PredictionResult(BaseModel):
    model:       str
    prediction:  int
    probability: float
    risk_label:  str
    f1:          Optional[float]
    auc_roc:     Optional[float]

class AllPredictionsResponse(BaseModel):
    patient_id:  Optional[str]
    predictions: Dict[str, PredictionResult]
    best_model:  str
    fhir_saved:  bool

# ── Helpers ──────────────────────────────────────────────────────────────────
def features_to_array(pf: PatientFeatures) -> np.ndarray:
    return np.array([[getattr(pf, f) for f in FEATURES]], dtype=np.float32)

def predict_with(key: str, X: np.ndarray):
    entry = MODELS[key]
    obj   = entry["obj"]
    # KNN / LR llevan scaler empaquetado
    if isinstance(obj, dict) and "model" in obj:
        scaler = obj["scaler"]
        model  = obj["model"]
        X_in   = scaler.transform(X)
    else:
        model, X_in = obj, X
    pred  = int(model.predict(X_in)[0])
    proba = float(model.predict_proba(X_in)[0][1])
    return pred, proba

async def save_fhir_risk(patient_id: str, preds: Dict[str, PredictionResult]):
    """Guarda predicciones como RiskAssessment FHIR R4."""
    for key, pr in preds.items():
        resource = {
            "resourceType": "RiskAssessment",
            "status":       "final",
            "subject":      {"reference": f"Patient/{patient_id}"},
            "method": {
                "coding": [{"system": "http://salud-digital-ia/algorithms",
                             "code": key, "display": pr.model}]
            },
            "prediction": [{
                "outcome": {
                    "text": pr.risk_label,
                    "coding": [{"system": "http://snomed.info/sct",
                                "code": "44054006" if pr.prediction == 1 else "413350009",
                                "display": pr.risk_label}]
                },
                "probabilityDecimal": round(pr.probability, 4),
                "rationale": f"ML prediction: {pr.model} | F1={pr.f1} AUC={pr.auc_roc}"
            }]
        }
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(f"{FHIR_BASE}/RiskAssessment", json=resource,
                              headers={"Content-Type": "application/fhir+json"})

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODELS.keys()), "best": BEST_MODEL_KEY}

@app.get("/models")
def list_models():
    return {
        k: {
            "algorithm": v["meta"].get("algorithm", k),
            "f1":        v["meta"].get("f1"),
            "auc_roc":   v["meta"].get("auc_roc"),
            "accuracy":  v["meta"].get("accuracy"),
        }
        for k, v in MODELS.items()
    }

@app.post("/predict/all/{patient_id}", response_model=AllPredictionsResponse)
async def predict_all(patient_id: str, pf: PatientFeatures):
    """
    Ejecuta TODOS los modelos cargados para un paciente y guarda
    los resultados como RiskAssessment en el servidor FHIR.
    Este endpoint debe declararse ANTES de /predict/{model_key}
    para que FastAPI no lo capture como model_key='all'.
    """
    if not MODELS:
        raise HTTPException(503, "No hay modelos cargados. Ejecutar run_all_training.py primero.")
    X     = features_to_array(pf)
    preds: Dict[str, PredictionResult] = {}
    for key in MODELS:
        try:
            pred, proba = predict_with(key, X)
        except Exception as e:
            print(f"  ⚠️  Error prediciendo con {key}: {e}")
            continue
        meta = MODELS[key]["meta"]
        preds[key] = PredictionResult(
            model      = meta.get("algorithm", key),
            prediction = pred,
            probability= proba,
            risk_label = "Enfermedad cardíaca detectada" if pred == 1 else "Sin riesgo cardíaco",
            f1         = meta.get("f1"),
            auc_roc    = meta.get("auc_roc"),
        )

    if not preds:
        raise HTTPException(500, "Todos los modelos fallaron al predecir.")

    fhir_ok = False
    try:
        await save_fhir_risk(patient_id, preds)
        fhir_ok = True
    except Exception as e:
        print(f"  ⚠️  FHIR save failed: {e}")

    return AllPredictionsResponse(
        patient_id  = patient_id,
        predictions = preds,
        best_model  = BEST_MODEL_KEY,
        fhir_saved  = fhir_ok,
    )

@app.post("/predict/{model_key}", response_model=PredictionResult)
def predict_single(model_key: str, pf: PatientFeatures):
    """Predicción con un modelo específico por su clave (dt, knn, gbm, lr, xgb, tabpfn)."""
    if model_key not in MODELS:
        available = list(MODELS.keys())
        raise HTTPException(
            404,
            f"Modelo '{model_key}' no encontrado. Disponibles: {available}"
        )
    X    = features_to_array(pf)
    pred, proba = predict_with(model_key, X)
    meta = MODELS[model_key]["meta"]
    return PredictionResult(
        model      = meta.get("algorithm", model_key),
        prediction = pred,
        probability= proba,
        risk_label = "Enfermedad cardíaca detectada" if pred == 1 else "Sin riesgo cardíaco",
        f1         = meta.get("f1"),
        auc_roc    = meta.get("auc_roc"),
    )

@app.get("/fhir/risk/{patient_id}")
async def get_fhir_risks(patient_id: str):
    """Obtiene RiskAssessments del servidor FHIR para un paciente."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{FHIR_BASE}/RiskAssessment",
            params={"subject": f"Patient/{patient_id}"},
            headers={"Accept": "application/fhir+json"}
        )
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"FHIR error: {r.text}")
    return r.json()

@app.get("/metrics/comparison")
def metrics_comparison():
    """Devuelve tabla comparativa de todos los modelos entrenados."""
    rows = []
    for k, v in MODELS.items():
        m = v["meta"]
        rows.append({
            "key":       k,
            "algorithm": m.get("algorithm", k),
            "f1":        m.get("f1"),
            "auc_roc":   m.get("auc_roc"),
            "accuracy":  m.get("accuracy"),
            "best":      k == BEST_MODEL_KEY,
        })
    rows.sort(key=lambda x: (x["f1"] or 0), reverse=True)
    return {"models": rows, "best_model": BEST_MODEL_KEY}


@app.post("/predict/demo", response_model=AllPredictionsResponse)
async def predict_demo(pf: PatientFeatures):
    """
    Endpoint de demostración: usa datos mock si no hay modelos cargados.
    Útil para probar el frontend antes del entrenamiento.
    """
    import random, hashlib
    random.seed(int(hashlib.md5(str(pf.age + pf.chol).encode()).hexdigest(), 16) % 10000)

    DEMO_METRICS = {
        "dt":     {"algorithm": "Decision Tree",        "f1": 0.8182, "auc_roc": 0.8241, "accuracy": 0.8197},
        "knn":    {"algorithm": "KNN",                  "f1": 0.7931, "auc_roc": 0.8103, "accuracy": 0.7869},
        "gbm":    {"algorithm": "Gradient Boosting",    "f1": 0.8519, "auc_roc": 0.9012, "accuracy": 0.8525},
        "lr":     {"algorithm": "Logistic Regression",  "f1": 0.8276, "auc_roc": 0.8897, "accuracy": 0.8197},
        "xgb":    {"algorithm": "XGBoost",              "f1": 0.8667, "auc_roc": 0.9134, "accuracy": 0.8689},
        "tabpfn": {"algorithm": "TabPFN",               "f1": 0.8621, "auc_roc": 0.9087, "accuracy": 0.8525},
    }

    # Si hay modelos reales, usarlos
    if MODELS:
        return await predict_all("demo-patient", pf)

    # Mock inteligente basado en factores de riesgo
    risk_score = (
        (pf.age > 55) * 0.15 +
        (pf.cp == 3) * 0.25 +
        (pf.thalach < 120) * 0.15 +
        (pf.oldpeak > 2.0) * 0.20 +
        (pf.ca > 0) * 0.20 +
        (pf.thal == 3) * 0.15
    )

    preds = {}
    for key, dm in DEMO_METRICS.items():
        noise   = random.gauss(0, 0.08)
        proba   = max(0.05, min(0.95, risk_score + noise))
        prediction = 1 if proba > 0.5 else 0
        preds[key] = PredictionResult(
            model      = dm["algorithm"],
            prediction = prediction,
            probability= round(proba, 4),
            risk_label = "Enfermedad cardíaca detectada" if prediction == 1 else "Sin riesgo cardíaco",
            f1         = dm["f1"],
            auc_roc    = dm["auc_roc"],
        )

    return AllPredictionsResponse(
        patient_id  = "demo",
        predictions = preds,
        best_model  = "xgb",
        fhir_saved  = False,
    )


@app.on_event("startup")
async def startup_event():
    """Recarga modelos al arrancar (útil si el volumen monta después)."""
    if not MODELS:
        print("  🔄  Reintentando carga de modelos en startup...")
        load_models()