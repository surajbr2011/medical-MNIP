import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from mnip.config import settings
from mnip.database.connection import get_db, Base, engine
from mnip.api.middleware import setup_middlewares
from mnip.api.schemas import FullMNIPReport, NegligenceResult, RiskScoreResponse, LegalQueryResponse

# Import routers
from mnip.ingestion.router import router as ingestion_router
from mnip.detection.router import router as detection_router
from mnip.risk.router import router as risk_router, fetch_episode_from_db
from mnip.legal.router import router as legal_router

# Import model singletons to initialize them during startup
from mnip.detection.model import get_detection_model_and_tokenizer, predict_negligence
from mnip.risk.model import load_risk_ensemble, predict_risk
from mnip.legal.reranker import get_cross_encoder
from mnip.legal.ingestion import ingest_corpus
from mnip.legal.retriever import retrieve_chunks
from mnip.legal.generator import generate_legal_advice

logger = logging.getLogger("mnip.main")

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
    title="Medical Negligence Intelligence Platform (MNIP)",
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

@app.get("/api/v1/episodes/{episode_id}/report", response_model=FullMNIPReport, tags=["report"])
async def get_episode_report(episode_id: str, db: AsyncSession = Depends(get_db)):
    """
    Returns a unified MNIP report combining data ingestion structures, NLP negligence detection,
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

    return FullMNIPReport(
        episode_id=episode_id,
        ingestion=ingestion_summary,
        detection=detection_res,
        risk=risk_res,
        legal=legal_res
    )
