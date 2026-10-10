import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from backend.database.connection import get_db
from backend.api.schemas import RiskScoreRequest, RiskScoreResponse, FHIREpisode
from backend.ingestion.models import Encounter, Observation, MedicationRequest, Procedure, DocumentReference
from backend.risk.model import predict_risk
from backend.ingestion.router import EPISODE_CACHE

logger = logging.getLogger("backend.risk.router")
router = APIRouter(prefix="/risk", tags=["risk"])

async def fetch_episode_from_db(episode_id: str, db: AsyncSession) -> FHIREpisode:
    """Queries all database records associated with the episode_id, falling back to in-memory cache if offline."""
    # Fast path: check in-memory episode cache first (ensures instant response without DB connection latency)
    if episode_id in EPISODE_CACHE:
        return EPISODE_CACHE[episode_id]

    try:
        # 1. Fetch the encounter
        stmt = select(Encounter).where(Encounter.id == episode_id).options(
            selectinload(Encounter.observations),
            selectinload(Encounter.medication_requests),
            selectinload(Encounter.procedures),
            selectinload(Encounter.document_references)
        )
        result = await db.execute(stmt)
        encounter = result.scalar_one_or_none()
        
        if encounter:
            observations_dict = {}
            for obs in encounter.observations:
                obs_key = obs.display or obs.code
                if obs_key:
                    if obs.value_quantity is not None:
                        observations_dict[obs_key] = obs.value_quantity
                    elif obs.value_string is not None:
                        observations_dict[obs_key] = obs.value_string

            medications_list = [med.medication_name for med in encounter.medication_requests if med.medication_name]
            procedures_list = [proc.display for proc in encounter.procedures if proc.display]
            notes_list = [doc.content_text for doc in encounter.document_references if doc.content_text]
            
            return FHIREpisode(
                patient_id=encounter.patient_id,
                encounter_id=encounter.id,
                clinical_notes=notes_list,
                medications=medications_list,
                procedures=procedures_list,
                observations=observations_dict
            )
    except Exception as db_err:
        logger.warning(f"Database query failed ({str(db_err)}). Falling back to in-memory cache.")

    # Check in-memory episode cache (e.g. when PostgreSQL is offline)
    if episode_id in EPISODE_CACHE:
        return EPISODE_CACHE[episode_id]

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Encounter episode with ID '{episode_id}' not found in database or cache."
    )
        
    # 2. Map observations to a dictionary
    observations_dict = {}
    for obs in encounter.observations:
        obs_key = obs.display or obs.code
        if obs_key:
            if obs.value_quantity is not None:
                observations_dict[obs_key] = obs.value_quantity
            elif obs.value_string is not None:
                observations_dict[obs_key] = obs.value_string

    # 3. Map other resources
    medications_list = [med.medication_name for med in encounter.medication_requests if med.medication_name]
    procedures_list = [proc.display for proc in encounter.procedures if proc.display]
    notes_list = [doc.content_text for doc in encounter.document_references if doc.content_text]
    
    # 4. Formulate the episode object
    return FHIREpisode(
        patient_id=encounter.patient_id,
        encounter_id=encounter.id,
        clinical_notes=notes_list,
        medications=medications_list,
        procedures=procedures_list,
        observations=observations_dict
    )

@router.post("/score", response_model=RiskScoreResponse)
async def get_risk_score(payload: RiskScoreRequest, db: AsyncSession = Depends(get_db)):
    """
    Retrieves the clinical data for a given episode_id, extracts the 62-dimension risk vector,
    runs the LightGBM + Random Forest stacking ensemble risk model, and returns calibrated
    negligence risk predictions with SHAP narrative explanations.
    """
    episode_id = payload.episode_id
    
    try:
        # Fetch data from DB
        episode = await fetch_episode_from_db(episode_id, db)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error querying database for episode {episode_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database fetch failed: {str(e)}"
        )
        
    try:
        # Run prediction
        result = predict_risk(episode)
        return result
    except Exception as e:
        logger.error(f"Error during risk score calculation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk scoring inference engine failed: {str(e)}"
        )
