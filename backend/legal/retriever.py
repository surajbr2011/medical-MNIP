import logging
from typing import List, Dict, Any
from backend.legal.ingestion import get_chroma_collection

logger = logging.getLogger("backend.legal.retriever")

def retrieve_chunks(query: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Dense retrieval.
    Queries ChromaDB and retrieves the top-15 document chunks matching the query.
    """
    try:
        collection = get_chroma_collection()
        results = collection.query(
            query_texts=[query],
            n_results=15
        )
        
        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            
            for i in range(len(docs)):
                # Relevance score: 1.0 - normalized distance
                distance = distances[i] if i < len(distances) else 0.5
                relevance_score = max(0.0, 1.0 - (distance / 2.0)) # chromadb distance ranges [0, 2] for cosine
                
                retrieved.append({
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "relevance_score": float(relevance_score)
                })
        
        logger.info(f"Dense retrieval retrieved {len(retrieved)} chunks for query: '{query}'")
        return retrieved
    except Exception as e:
        logger.error(f"Error in retrieve_chunks: {str(e)}")
        return []
