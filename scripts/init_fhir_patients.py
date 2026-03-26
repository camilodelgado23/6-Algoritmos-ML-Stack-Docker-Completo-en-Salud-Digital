"""
scripts/init_fhir_patients.py
Crea pacientes de prueba en el servidor HAPI FHIR local.
Ejecutar después de que el stack Docker esté levantado:
  python scripts/init_fhir_patients.py
"""

import json
import httpx

FHIR_BASE = "http://localhost:8080/fhir"
HEADERS   = {
    "Content-Type": "application/fhir+json",
    "Accept":       "application/fhir+json",
}

PATIENTS = [
    {
        "id":     "P001",
        "name":   "Carlos Mendoza",
        "gender": "male",
        "birth":  "1960-03-15",
        "note":   "Paciente con historial de hipertensión",
    },
    {
        "id":     "P002",
        "name":   "Ana González",
        "gender": "female",
        "birth":  "1972-07-22",
        "note":   "Control anual sin antecedentes cardíacos",
    },
    {
        "id":     "P003",
        "name":   "Roberto Díaz",
        "gender": "male",
        "birth":  "1955-11-08",
        "note":   "Diabético tipo 2, seguimiento cardíaco",
    },
]

def fhir_patient(p: dict) -> dict:
    given, family = p["name"].split(" ", 1)
    return {
        "resourceType": "Patient",
        "id":           p["id"],
        "active":       True,
        "name": [{"use": "official", "family": family, "given": [given]}],
        "gender":    p["gender"],
        "birthDate": p["birth"],
        "text": {
            "status": "generated",
            "div":    f'<div xmlns="http://www.w3.org/1999/xhtml">{p["name"]} — {p["note"]}</div>'
        }
    }

def main():
    print(f"🏥  Conectando a FHIR: {FHIR_BASE}")
    for p in PATIENTS:
        resource = fhir_patient(p)
        try:
            r = httpx.put(
                f"{FHIR_BASE}/Patient/{p['id']}",
                json=resource,
                headers=HEADERS,
                timeout=15
            )
            if r.status_code in (200, 201):
                print(f"  ✅  {p['id']} — {p['name']} creado/actualizado")
            else:
                print(f"  ⚠️   {p['id']} — HTTP {r.status_code}: {r.text[:120]}")
        except httpx.ConnectError:
            print(f"  ❌  No se pudo conectar a {FHIR_BASE}")
            print("      Asegúrate de que el stack Docker esté corriendo:")
            print("      docker-compose up -d")
            break

    print("\n🔍  Verificando pacientes creados...")
    try:
        r = httpx.get(f"{FHIR_BASE}/Patient", headers=HEADERS, timeout=10)
        bundle = r.json()
        total  = bundle.get("total", "?")
        print(f"  Total pacientes en FHIR: {total}")
    except Exception as e:
        print(f"  Error al verificar: {e}")

if __name__ == "__main__":
    main()