import logging
from fastapi import APIRouter, HTTPException, status
from mnip.api.schemas import LegalQueryRequest, LegalQueryResponse
from mnip.legal.retriever import retrieve_chunks
from mnip.legal.reranker import rerank_chunks
from mnip.legal.generator import generate_legal_advice

logger = logging.getLogger("mnip.legal.router")
router = APIRouter(prefix="/legal", tags=["legal"])

@router.post("/query", response_model=LegalQueryResponse)
async def query_legal_intelligence(payload: LegalQueryRequest):
    """
    Evaluates clinical incident descriptions against Indian case law precedents.
    Implements a 3-Stage RAG Pipeline:
    1. Retrieve the top 15 relevant chunks from ChromaDB using sentence-transformers.
    2. Rerank down to top 5 using a Cross-Encoder.
    3. Generate structured legal counsel (citations, standard of care, liability assessment) via local Ollama Mistral.
    """
    incident_description = payload.incident_description.strip()
    if not incident_description:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incident description cannot be empty."
        )
        
    try:
        # Stage 1: Dense retrieval (top-15)
        retrieved = retrieve_chunks(incident_description)
        
        # Stage 2: Cross-encoder reranking (top-5)
        reranked = rerank_chunks(incident_description, retrieved)
        
        # Stage 3: LLM prompt execution and output structure generation
        analysis = await generate_legal_advice(incident_description, reranked)
        
        return LegalQueryResponse(
            citations=analysis.get("citations", []),
            statutory_provisions=analysis.get("statutory_provisions", []),
            standard_of_care_summary=analysis.get("standard_of_care_summary", ""),
            liability_assessment=analysis.get("liability_assessment", "")
        )
    except Exception as e:
        logger.error(f"Failed to process legal intelligence RAG pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Legal intelligence module pipeline failure: {str(e)}"
        )
