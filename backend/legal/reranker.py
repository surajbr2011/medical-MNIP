import logging
from typing import List, Dict, Any
from functools import lru_cache
from sentence_transformers import CrossEncoder

logger = logging.getLogger("backend.legal.reranker")

@lru_cache(maxsize=1)
def get_cross_encoder() -> Any:
    """Loads the CrossEncoder model once (singleton pattern)."""
    try:
        # Load from HuggingFace
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return model
    except Exception as e:
        logger.warning(f"Failed to load CrossEncoder from HF: {str(e)}. Fallback to keyword relevance matching.")
        return None

def rerank_chunks(query: str, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage 2: Cross-encoder reranking.
    Takes top-15 chunks from Stage 1, reranks them using the Cross-Encoder,
    and returns the top-5 chunks.
    """
    if not retrieved_chunks:
        return []
        
    encoder = get_cross_encoder()
    
    if encoder is not None:
        try:
            # Construct pairs: [[query, text1], [query, text2], ...]
            pairs = [[query, chunk["text"]] for chunk in retrieved_chunks]
            
            # Predict scores
            scores = encoder.predict(pairs)
            
            # Update scores in chunks
            for idx, score in enumerate(scores):
                # Sigmoid to normalize score if needed, or raw values.
                # MS-MARCO CrossEncoder returns raw logits, but ranking by logit is mathematically identical to ranking by sigmoid.
                retrieved_chunks[idx]["rerank_score"] = float(score)
                
            # Sort by rerank_score descending
            reranked = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
            logger.info(f"CrossEncoder successfully reranked {len(reranked)} chunks.")
            return reranked[:5]
        except Exception as e:
            logger.error(f"Error during CrossEncoder predict: {str(e)}. Falling back to Stage 1 scores.")
            
    # Fallback: Sort by Stage 1 relevance score
    sorted_by_stage1 = sorted(retrieved_chunks, key=lambda x: x["relevance_score"], reverse=True)
    return sorted_by_stage1[:5]
