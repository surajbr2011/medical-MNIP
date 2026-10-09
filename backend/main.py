import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database.connection import get_db, Base, engine
from backend.api.middleware import setup_middlewares
from backend.api.schemas import FullbackendReport, NegligenceResult, RiskScoreResponse, LegalQueryResponse

# Import routers
from backend.ingestion.router import router as ingestion_router
from backend.detection.router import router as detection_router
from backend.risk.router import router as risk_router, fetch_episode_from_db
from backend.legal.router import router as legal_router

# Import model singletons to initialize them during startup
from backend.detection.model import get_detection_model_and_tokenizer, predict_negligence
from backend.risk.model import load_risk_ensemble, predict_risk
from backend.legal.reranker import get_cross_encoder, rerank_chunks
from backend.legal.ingestion import ingest_corpus
from backend.legal.retriever import retrieve_chunks
from backend.legal.generator import generate_legal_advice

logger = logging.getLogger("backend.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle context manager executing actions on startup and shutdown."""
    logger.info("Initializing Medical Negligence Intelligence Platform...")
    
    # 1. Initialize databases and create tables if needed
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schemas verified/created.")
    
    # 2. Ingest legal precedents to ChromaDB vector store
    try:
        ingest_corpus()
    except Exception as e:
        logger.error(f"Failed to seed legal corpus to ChromaDB: {str(e)}")
        
    # 3. Pre-load Machine Learning Models (Singleton Lifespan Pattern)
    logger.info("Pre-loading Bio_ClinicalBERT tokenizer & weights...")
    get_detection_model_and_tokenizer()
    
    logger.info("Pre-loading Risk Stacking Ensemble...")
    load_risk_ensemble()
    
    logger.info("Pre-loading Cross-Encoder Reranker...")
    get_cross_encoder()
    
    logger.info("Lifespan startup sequences completed successfully.")
    yield
    logger.info("Platform shutting down.")

app = FastAPI(
    title="Medical Negligence Intelligence Platform (backend)",
    version="1.0.0",
    description="ABDM-compliant FHIR R4 clinical data ingestion, deep learning-based negligence classification, risk scoring, and legal advice generator.",
    lifespan=lifespan
)

# Setup middlewares (CORS + Logging)
setup_middlewares(app)

# Include routers
app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(detection_router, prefix="/api/v1")
app.include_router(risk_router, prefix="/api/v1")
app.include_router(legal_router, prefix="/api/v1")

@app.get("/", status_code=status.HTTP_200_OK, tags=["system"])
async def root():
    return {"message": "Welcome to the Medical Negligence Intelligence Platform API."}

@app.get("/health", status_code=status.HTTP_200_OK, tags=["system"])
@app.get("/api/v1/health", status_code=status.HTTP_200_OK, tags=["system"])
async def health_check():
    return {
        "status": "healthy",
        "service": "Medical Negligence Intelligence Platform API",
        "version": "1.0.0"
    }

@app.get("/api/v1/episodes/{episode_id}/report", response_model=FullbackendReport, tags=["report"])
async def get_episode_report(episode_id: str, db: AsyncSession = Depends(get_db)):
    """
    Returns a unified backend report combining data ingestion structures, NLP negligence detection,
    risk scoring, and RAG legal recommendations for a given episode_id.
    """
    try:
        # 1. Fetch structured episode details
        episode = await fetch_episode_from_db(episode_id, db)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error compiling report for episode '{episode_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch episode details: {str(e)}"
        )
        
    # 2. Run NLP detection if clinical notes exist
    detection_res = None
    note_text = ""
    if episode.clinical_notes:
        note_text = episode.clinical_notes[0]
        try:
            detection_res = predict_negligence(note_text)
        except Exception as e:
            logger.error(f"Report generation: Negligence detection failure: {str(e)}")
            
    # 3. Calculate Risk Score
    risk_res = None
    try:
        risk_res = predict_risk(episode)
    except Exception as e:
        logger.error(f"Report generation: Risk scoring failure: {str(e)}")
        
    # 4. Generate Legal advice
    legal_res = None
    if note_text:
        try:
            retrieved = retrieve_chunks(note_text)
            reranked = rerank_chunks(note_text, retrieved)
            advice = await generate_legal_advice(note_text, reranked)
            legal_res = LegalQueryResponse(
                citations=advice.get("citations", []),
                statutory_provisions=advice.get("statutory_provisions", []),
                standard_of_care_summary=advice.get("standard_of_care_summary", ""),
                liability_assessment=advice.get("liability_assessment", "")
            )
        except Exception as e:
            logger.error(f"Report generation: Legal advice generation failure: {str(e)}")

    # Ingestion stats dictionary
    ingestion_summary = {
        "patient_id": episode.patient_id,
        "encounter_id": episode.encounter_id,
        "note_count": len(episode.clinical_notes),
        "medication_count": len(episode.medications),
        "procedure_count": len(episode.procedures),
        "observation_count": len(episode.observations)
    }

    return FullbackendReport(
        episode_id=episode_id,
        ingestion=ingestion_summary,
        detection=detection_res,
        risk=risk_res,
        legal=legal_res
    )

# Global state for mock live analytics
analytics_state = {
    "total_incidents": 148,
    "critical_alerts": 14,
    "latency": 0.85,
    "risk_values": [82, 41, 18, 7],
    "cat_counts": [34, 21, 15, 12, 11, 8, 5, 2],
    "counts": [10, 8, 12, 14, 9, 11, 15, 13, 10, 16, 12, 18]
}

@app.get("/api/v1/analytics/live", tags=["analytics"])
async def get_live_analytics():
    # Simulate data fluctuation
    analytics_state["total_incidents"] += random.randint(0, 2)
    
    if random.random() > 0.8:
        analytics_state["critical_alerts"] += 1
        
    analytics_state["latency"] = max(0.4, min(1.5, analytics_state["latency"] + random.uniform(-0.1, 0.1)))
    
    # Fluctuate pie chart (risk)
    idx_risk = random.randint(0, 3)
    analytics_state["risk_values"][idx_risk] += random.randint(0, 1)
    
    # Fluctuate bar chart (domains)
    idx_cat = random.randint(0, 7)
    analytics_state["cat_counts"][idx_cat] += random.randint(0, 1)
    
    # Fluctuate line chart (time series)
    if random.random() > 0.7:
        analytics_state["counts"].pop(0)
        analytics_state["counts"].append(max(0, analytics_state["counts"][-1] + random.randint(-3, 4)))
        
    total_risk = sum(analytics_state["risk_values"]) or 1
    neg_rate = ((analytics_state["risk_values"][2] + analytics_state["risk_values"][3]) / total_risk) * 100
        
    return {
        "kpis": {
            "total_incidents": analytics_state["total_incidents"],
            "negligence_rate": round(neg_rate, 1),
            "critical_alerts": analytics_state["critical_alerts"],
            "latency": round(analytics_state["latency"], 2)
        },
        "risk_values": analytics_state["risk_values"],
        "cat_counts": analytics_state["cat_counts"],
        "time_series": analytics_state["counts"]
    }
