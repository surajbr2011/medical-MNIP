import logging
from fastapi import APIRouter, HTTPException, status
from backend.api.schemas import NegligenceDetectRequest, NegligenceResult
from backend.detection.model import predict_negligence

logger = logging.getLogger("backend.detection.router")
router = APIRouter(prefix="/detect", tags=["detection"])

@router.post("", response_model=NegligenceResult)
async def detect_negligence(payload: NegligenceDetectRequest):
    """
    Evaluates clinical note text for medical negligence. Uses fine-tuned Bio_ClinicalBERT
    to classify across 8 WHO ICPS categories, determine overall negligence probability,
    and returns token-level attribution scores for explainability.
    """
    raw_text = payload.note_text or payload.clinical_note or ""
    note_text = raw_text.strip()
    if not note_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Note text or clinical note cannot be empty."
        )
        
    try:
        # Run prediction
        result = predict_negligence(note_text)
        return result
    except Exception as e:
        logger.error(f"Error during negligence detection inference: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference engine failure: {str(e)}"
        )
