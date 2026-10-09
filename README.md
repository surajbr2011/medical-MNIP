# Medical Negligence Intelligence Platform (MNIP)

The **Medical Negligence Intelligence Platform (MNIP)** is a production-grade AI platform designed to proactively detect, score, and provide legal intelligence for medical negligence incidents from clinical EHR data. It targets Indian tertiary care hospitals and is ABDM-compliant, parsing FHIR R4 JSON structures.

---

## System Architecture

MNIP is built using 4 core modular engines:

1. **Module 1: Data Ingestion (`mnip/ingestion/`)**
   - Ingests HL7 FHIR R4 JSON bundles (Patient, Encounter, Observation, MedicationRequest, Procedure, DocumentReference).
   - De-identifies text using Microsoft Presidio and custom clinical pattern recognizers (ABHA Health ID, UHID, Aadhaar).
   - Persists structured clinical data to PostgreSQL while logging de-identification events to an audit trail table.

2. **Module 2: Negligence Detection Engine (`mnip/detection/`)**
   - Fine-tuned **Bio_ClinicalBERT** model with a multi-task classification head predicting 8 WHO ICPS categories + 1 binary negligence label.
   - Implements gradient-based token attribution (saliency maps) for explainability heatmaps.

3. **Module 3: Risk Scoring Engine (`mnip/risk/`)**
   - Extracts 31 clinical, temporal, medication, administrative, and NLP features (plus 31 missingness flags, 62 dimensions total).
   - Predicts calibrated negligence risk scores utilizing a LightGBM + Random Forest stacking ensemble with Isotonic regression.
   - Computes SHAP value impact scores and translates them into clinical risk narratives.

4. **Module 4: Legal Intelligence Module (`mnip/legal/`)**
   - Ingests Indian case law precedents and Supreme court judgments into a ChromaDB vector database using `all-MiniLM-L6-v2`.
   - Utilizes a 3-Stage RAG Pipeline: Dense Retrieval (top-15), Cross-Encoder Reranking (top-5 using `ms-marco-MiniLM-L-6-v2`), and LLM Advice Generation (Mistral-7B-Instruct via local Ollama).

---

## Technology Stack

- **Core**: Python 3.11, FastAPI, Uvicorn
- **AI/NLP**: PyTorch, Transformers (Bio_ClinicalBERT), Scikit-Learn, LightGBM, SHAP
- **Database**: PostgreSQL, Neo4j Graph DB, ChromaDB Vector DB
- **UI**: Streamlit Dashboard, Plotly
- **Infrastructure**: Docker, Docker Compose, MLflow

---

## Directory Structure

```text
mnip/
├── main.py                     # FastAPI entrypoint (lifespan model caching, routing)
├── config.py                   # Pydantic Settings
├── requirements.txt            # Pinned dependencies
├── docker-compose.yml          # Container configuration
├── Dockerfile                  # Multi-stage non-root container builder
├── .env.example                # Template for environment variables
├── README.md                   # Guide
├── ingestion/                  # Module 1
├── detection/                  # Module 2
├── risk/                       # Module 3
├── legal/                      # Module 4
├── database/                   # SQLAlchemy async connections
├── graph/                      # Neo4j pathway graphs
├── api/                        # Shared middleware, schemas
├── dashboard/                  # Streamlit frontend app
└── tests/                      # Pytest unit tests
```

---

## Running the Platform

### Option A: Using Docker Compose (Recommended)

1. Clone or copy the MNIP workspace files.
2. Initialize environment:
   ```bash
   cp .env.example .env
   ```
3. Start the entire container stack (Postgres, Neo4j, Ollama, API, UI):
   ```bash
   docker compose up -build -d
   ```
   *Note: On first boot, the Ollama container will pull the `mistral` LLM model automatically via its entrypoint script.*

### Option B: Running Locally (Without Docker)

1. Activate your Python 3.11 virtual environment and install dependencies:
   ```powershell
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Verify local database services (Postgres, Neo4j, Ollama) or allow fallback mode.
3. Seed the ChromaDB legal collection, train base models, and run:
   ```bash
   # 1. Seed Legal DB
   python -m backend.legal.ingestion
   # 2. Train Risk ML Model
   python -m backend.risk.trainer
   # 3. Launch FastAPI Server
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```
4. In a separate terminal, launch the modern React Web App:
   ```bash
   cd frontend
   npm install
   npm run dev
   # Access UI at http://localhost:5173
   ```
5. (Optional) Launch the Streamlit analytics console:
   ```bash
   streamlit run dashboard/app.py --server.port 8501
   ```

---

## REST API Documentation

Once started, access the swagger interactive docs at:
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

### API Contracts Summary

- **POST `/api/v1/ingest/fhir`**: Ingest and de-identify FHIR bundle.
- **POST `/api/v1/detect`**: Run Bio_ClinicalBERT negligence classification + explainability.
- **POST `/api/v1/risk/score`**: Stacking ML ensemble risk scoring + SHAP values.
- **POST `/api/v1/legal/query`**: Dense retrieval + Cross-encoder rerank + Mistral RAG.
- **GET `/api/v1/episodes/{episode_id}/report`**: Unified compilation report combining all 4 modules.

---

## Running Unit Tests

Run the test suite using `pytest`:
```bash
python -m pytest tests/
```
