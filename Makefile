# ═══════════════════════════════════════════════════════════════
#  Salud Digital IA — Makefile
#  Uso: make <comando>
# ═══════════════════════════════════════════════════════════════

.PHONY: help install dataset train up down logs clean reset test-api fhir-init

# ── Colores ──────────────────────────────────────────────────────
CYAN  = \033[0;36m
GREEN = \033[0;32m
RESET = \033[0m

help:  ## Muestra esta ayuda
	@echo ""
	@echo "  $(CYAN)SALUD DIGITAL IA — Comandos disponibles$(RESET)"
	@echo "  ─────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2}'
	@echo ""

install:  ## Instala dependencias Python para training
	pip install -r training/requirements.txt

dataset:  ## Descarga Heart Disease UCI y lo guarda en data/
	python scripts/prepare_dataset.py

train:  ## Entrena los 6 modelos y elige el mejor
	python run_all_training.py

train-docker:  ## Entrena dentro de Docker (requiere stack levantado)
	docker-compose run --rm training

up:  ## Levanta el stack completo
	docker-compose up --build -d
	@echo ""
	@echo "  $(GREEN)Stack levantado:$(RESET)"
	@echo "   Frontend  → http://localhost:3000"
	@echo "   API docs  → http://localhost:8000/docs"
	@echo "   MLflow    → http://localhost:5000"
	@echo "   FHIR UI   → http://localhost:8080"

down:  ## Detiene el stack
	docker-compose down

logs:  ## Muestra logs de todos los servicios
	docker-compose logs -f --tail=50

logs-ai:  ## Muestra logs del ai-service
	docker-compose logs -f ai-service

clean:  ## Detiene el stack y elimina volúmenes
	docker-compose down -v
	@echo "  $(CYAN)Volúmenes eliminados$(RESET)"

reset:  ## Limpieza total: modelos + volúmenes + imágenes
	docker-compose down -v --rmi local
	rm -f ai-service/models/*.pkl ai-service/models/*.json
	@echo "  $(CYAN)Reset completo$(RESET)"

fhir-init:  ## Crea pacientes de prueba en FHIR
	pip install httpx -q
	python scripts/init_fhir_patients.py

test-api:  ## Test rápido del API con curl
	@echo "\n  $(CYAN)GET /health$(RESET)"
	@curl -s http://localhost:8000/health | python3 -m json.tool
	@echo "\n  $(CYAN)GET /models$(RESET)"
	@curl -s http://localhost:8000/models | python3 -m json.tool
	@echo "\n  $(CYAN)POST /predict/demo$(RESET)"
	@curl -s -X POST http://localhost:8000/predict/demo \
	  -H "Content-Type: application/json" \
	  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}' \
	  | python3 -m json.tool

status:  ## Estado de todos los contenedores
	docker-compose ps

# ── Flujo completo en un comando ─────────────────────────────────
all: install dataset train up fhir-init  ## Flujo completo: instalar → dataset → train → up → FHIR
	@echo ""
	@echo "  $(GREEN)✅ Proyecto completo levantado$(RESET)"
	@echo "   Abre http://localhost:3000 en tu navegador"