# 🫀 Salud Digital IA

> **6 algoritmos ML · Heart Disease UCI · FastAPI · FHIR R4 · Docker**

Sistema clínico de inteligencia artificial que implementa, compara y despliega 6 algoritmos de Machine Learning para detección de enfermedad cardíaca, integrado con un servidor FHIR local dockerizado.

---

## 🎯 Dataset elegido: Heart Disease UCI (id=45)

**¿Por qué este dataset?**
- **303 pacientes** — pequeño y limpio, entrenamiento rápido
- **13 features** clínicas bien documentadas (edad, colesterol, ECG, etc.)
- **Clasificación binaria** clara: enfermedad cardíaca sí/no
- **Sin dependencias complejas** — funciona con `ucimlrepo` en una línea
- **Compatible con TabPFN** — que requiere ≤1000 muestras

```python
from ucimlrepo import fetch_ucirepo
repo = fetch_ucirepo(id=45)  # listo
```

---

## 🏗️ Arquitectura Docker

```
C1 fhir-db       postgres:15-alpine     (interno)
C2 fhir-server   hapiproject/hapi       :8080
C3 ai-service    ./ai-service           :8000  ← FastAPI + 6 modelos
C4 mlflow        mlflow:v2.12.2         :5000
C5 frontend      ./frontend             :3000
          └──── fhir-net (bridge) ────────────┘
```

---

## 🚀 Inicio rápido

### 1. Entrenamiento (fuera de Docker)
```bash
pip install scikit-learn xgboost mlflow ucimlrepo
python run_all_training.py
# Genera: ai-service/models/*.pkl + *_meta.json + best_model.json
```

### 2. Levantar stack completo
```bash
docker-compose up --build
```

### 3. URLs de acceso
| Servicio   | URL                        |
|-----------|---------------------------|
| Frontend   | http://localhost:3000      |
| API /docs  | http://localhost:8000/docs |
| MLflow     | http://localhost:5000      |
| FHIR UI    | http://localhost:8080      |

---

## 🤖 Los 6 algoritmos

| # | Algoritmo         | Archivo            | Ventaja clave              |
|---|------------------|--------------------|---------------------------|
| 1 | Decision Tree    | train_dt.py        | Interpretabilidad máxima  |
| 2 | KNN              | train_knn.py       | No paramétrico, simple    |
| 3 | Gradient Boosting| train_gbm.py       | Robusto, alta precisión   |
| 4 | Logistic Reg.    | train_lr.py        | Baseline interpretable    |
| 5 | XGBoost          | train_xgb.py       | Estado del arte tabular   |
| 6 | TabPFN           | train_tabpfn.py    | Bayesiano, sin tunning    |

Cada script:
- Hace grid search con MLflow tracking
- Guarda el mejor modelo como `.pkl`
- Genera metadatos `.json` (F1, AUC-ROC, accuracy, confusion matrix)

---

## 📡 API Endpoints

```
GET  /health                    Estado del servicio + modelos cargados
GET  /models                    Lista modelos disponibles con métricas
POST /predict/{model_key}       Predicción con un modelo específico
POST /predict/all/{patient_id}  Todos los modelos + guarda en FHIR
GET  /fhir/risk/{patient_id}    Lee RiskAssessments del servidor FHIR
GET  /metrics/comparison        Tabla comparativa de modelos
```

---

## 🏥 Integración FHIR

Las predicciones se guardan como recursos `RiskAssessment` FHIR R4:

```json
{
  "resourceType": "RiskAssessment",
  "status": "final",
  "subject": {"reference": "Patient/P001"},
  "method": {"coding": [{"code": "xgb", "display": "XGBoost"}]},
  "prediction": [{
    "outcome": {"text": "Enfermedad cardíaca detectada"},
    "probabilityDecimal": 0.8723
  }]
}
```

---

## 📊 Entregables por algoritmo (en Dashboard)

Para cada modelo el dashboard muestra:
1. **Teoría de uso** — cuándo y por qué usarlo
2. **Principio de funcionamiento** — fórmula matemática + pseudocódigo
3. **Ventajas y desventajas** — tabla comparativa honesta
4. **Implementación Docker** — microservicio FastAPI integrado
5. **Métricas y comparativa** — F1, AUC-ROC, accuracy