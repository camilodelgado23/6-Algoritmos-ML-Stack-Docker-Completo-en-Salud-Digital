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
C1 fhir-db       postgres:15-alpine     (interno, sin puerto)
C2 fhir-server   hapiproject/hapi       :8080
C3 ai-service    ./ai-service           :8000  ← FastAPI + 6 modelos
C4 mlflow        mlflow:v2.12.2         :5000
C5 frontend      ./frontend             :3000
          └──── fhir-net (bridge) ────────────┘
```

---

## 📁 Estructura de archivos

```
salud-digital-ia/
│
├── docker-compose.yml        ← orquesta los 5 contenedores
├── Makefile                  ← comandos rápidos
├── run_all_training.py       ← entrena los 6 modelos y elige el mejor
├── README.md
│
├── ai-service/
│   ├── Dockerfile            ← imagen Docker del servicio AI
│   ├── main.py               ← FastAPI: endpoints + integración FHIR
│   ├── requirements.txt
│   └── models/               ← aquí caen los .pkl tras el training
│
├── training/
│   ├── Dockerfile            ← opcional: entrenar dentro de Docker
│   ├── requirements.txt
│   ├── train_dt.py           ← Decision Tree
│   ├── train_knn.py          ← KNN
│   ├── train_gbm.py          ← Gradient Boosting
│   ├── train_lr.py           ← Logistic Regression
│   ├── train_xgb.py          ← XGBoost
│   └── train_tabpfn.py       ← TabPFN
│
├── frontend/
│   ├── Dockerfile            ← nginx:alpine
│   ├── nginx.conf            ← puerto 3000, proxy /api/
│   ├── index.html            ← dashboard clínico
│   └── app.js                ← lógica: predicción, Chart.js, tabs
│
├── fhir-server/
│   └── application.yaml      ← config HAPI FHIR: CORS, IDs externos
│
├── scripts/
│   ├── prepare_dataset.py    ← descarga Heart Disease UCI
│   └── init_fhir_patients.py ← crea pacientes de prueba en FHIR
│
└── data/
    └── heart_disease.csv     ← generado por prepare_dataset.py
```

---

## 🚀 Ejecución paso a paso

### Requisitos previos
- Python 3.10 o superior instalado
- Docker Desktop instalado y **corriendo**
- Conexión a internet (para descargar el dataset e imágenes Docker)

---

### Paso 1 — Entrar al directorio

```bash
cd 6-Algoritmos-ML-Stack-Docker-Completo-en-Salud-Digital
```

### Paso 2 — Instalar dependencias Python para el training

```bash
pip install -r training/requirements.txt
```

---

### Paso 3 — Descargar el dataset

```bash
python scripts/prepare_dataset.py
```

Salida esperada:
```
✅  Dataset guardado en data/heart_disease.csv
   Filas: 303  |  Columnas: 14
   Distribución target: {0: 164, 1: 139}
```

---

### Paso 4 — Entrenar los 6 modelos

```bash
python run_all_training.py
```

Tarda entre **3 y 8 minutos**. Al final imprime la tabla comparativa y elige automáticamente el mejor modelo, ejemplo visual:

```
══════════════════════════════════════════════════════════
  COMPARACIÓN FINAL DE MODELOS 
══════════════════════════════════════════════════════════

Algoritmo                  F1      AUC-ROC    Accuracy
──────────────────────────────────────────────────────────
  XGBoost              0.8667     0.9134      0.8689  ◄ MEJOR
  TabPFN               0.8621     0.9087      0.8525
  Gradient Boosting    0.8519     0.9012      0.8525
  Logistic Regression  0.8276     0.8897      0.8197
  Decision Tree        0.8182     0.8241      0.8197
  KNN                  0.7931     0.8103      0.7869

🏆  Modelo seleccionado: XGBoost
✅  Resultado guardado en ai-service/models/best_model.json
```

Verifica que se crearon los modelos:

```bash
ls ai-service/models/
# dt_model.pkl   knn_model.pkl   gbm_model.pkl
# lr_model.pkl   xgb_model.pkl   tabpfn_model.pkl
# best_model.json
```

---

### Paso 5 — Levantar el stack Docker

```bash
docker-compose up --build
```

---

### Paso 6 — Crear pacientes de prueba en FHIR

Abre **una terminal nueva** (deja Docker corriendo en la anterior):

```bash
python scripts/init_fhir_patients.py
```

---

### Paso 7 — Abrir en el navegador

| Servicio | URL | Descripción |
|---|---|---|
| **Dashboard** | http://localhost:3000 | Dashboard clínico — aquí trabajas |
| **API docs** | http://localhost:8000/docs | Endpoints FastAPI interactivos |
| **MLflow** | http://localhost:5000 | Experimentos y métricas de training |
| **FHIR UI** | http://localhost:8080 | Servidor HAPI — ver RiskAssessments |

---

## 🛠️ Comandos del día a día

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Ver logs solo del ai-service
docker-compose logs -f ai-service

# Reiniciar solo el ai-service (si cambias main.py)
docker-compose restart ai-service

# Detener el stack (conserva los datos)
docker-compose down

# Detener y borrar volúmenes (reset total)
docker-compose down -v

# Estado de todos los contenedores
docker-compose ps
```

Con el Makefile también se puedes usar:

```bash
make up        # levantar stack
make down      # detener stack
make logs      # ver logs
make train     # entrenar modelos
make test-api  # probar API con curl
make status    # estado de contenedores
make all       # flujo completo de una vez
```

---

## 🤖 Los 6 algoritmos

| # | Algoritmo | Archivo | Ventaja clave |
|---|---|---|---|
| 1 | Decision Tree | `train_dt.py` | Interpretabilidad máxima |
| 2 | KNN | `train_knn.py` | No paramétrico, simple |
| 3 | Gradient Boosting | `train_gbm.py` | Robusto, alta precisión |
| 4 | Logistic Regression | `train_lr.py` | Baseline interpretable |
| 5 | XGBoost | `train_xgb.py` | Estado del arte tabular |
| 6 | TabPFN | `train_tabpfn.py` | Bayesiano, sin tunning |

Cada script hace grid search con MLflow tracking, guarda el mejor modelo como `.pkl` y genera metadatos `.json` con F1, AUC-ROC, accuracy y confusion matrix.

---

## 📡 API Endpoints

```
GET  /health                    Estado del servicio + modelos cargados
GET  /models                    Lista modelos disponibles con métricas
POST /predict/{model_key}       Predicción con un modelo específico
POST /predict/all/{patient_id}  Todos los modelos + guarda en FHIR
POST /predict/demo              Demo sin modelos entrenados (datos mock)
GET  /fhir/risk/{patient_id}    Lee RiskAssessments del servidor FHIR
GET  /metrics/comparison        Tabla comparativa de modelos
```

Claves válidas para `{model_key}`: `dt`, `knn`, `gbm`, `lr`, `xgb`, `tabpfn`

---

## 🏥 Integración FHIR

Las predicciones se guardan automáticamente como recursos `RiskAssessment` FHIR R4:

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

## 📊 Entregables por algoritmo en el Dashboard

Para cada uno de los 6 modelos el dashboard muestra:

1. **Teoría de uso** — cuándo y por qué usarlo, supuestos del modelo
2. **Principio de funcionamiento** — fórmula matemática clave + pseudocódigo
3. **Ventajas y desventajas** — comparativa honesta de complejidad, interpretabilidad, escalabilidad
4. **Implementación Docker** — el modelo corre dentro del microservicio FastAPI dockerizado
5. **Métricas y comparativa final** — F1, AUC-ROC, accuracy en test set