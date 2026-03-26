"""
scripts/prepare_dataset.py
Descarga Heart Disease UCI (id=45) y lo guarda en data/heart_disease.csv
Ejecutar una sola vez antes del entrenamiento.
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUT_CSV  = os.path.join(DATA_DIR, "heart_disease.csv")

print("📥  Descargando Heart Disease UCI (id=45)...")
repo = fetch_ucirepo(id=45)

X = repo.data.features.copy()
y = repo.data.targets.copy()

# Target binario: num > 0 → enfermedad (1)
y_bin = (y.squeeze() > 0).astype(int).rename("target")

df = pd.concat([X, y_bin], axis=1)
df.to_csv(OUT_CSV, index=False)

print(f"✅  Dataset guardado en {OUT_CSV}")
print(f"   Filas: {len(df)}  |  Columnas: {len(df.columns)}")
print(f"   Distribución target: {df['target'].value_counts().to_dict()}")
print(f"\nFeatures ({len(X.columns)}):")
for col in X.columns:
    print(f"  {col:12}  dtype={X[col].dtype}  nulls={X[col].isnull().sum()}")