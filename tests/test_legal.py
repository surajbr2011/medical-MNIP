import pytest
from backend.legal.ingestion import get_chroma_collection, MOCK_LEGAL_CORPUS
from backend.legal.retriever import retrieve_chunks
from backend.legal.reranker import rerank_chunks
from backend.legal.generator import generate_fallback_analysis

def test_legal_pipeline():
    # Ingest if not exists (collection initialization is covered)
    collection = get_chroma_collection()
    assert collection is not None
    
    query = "surgery done without consent"
    
    # 1. Retrieve
    retrieved = retrieve_chunks(query)
    # Since it may be empty in local test unless seeded, we handle fallback or check it's a list
    assert isinstance(retrieved, list)
    
    # 2. Rerank
    reranked = rerank_chunks(query, retrieved)
    assert isinstance(reranked, list)
    assert len(reranked) <= 5
    
    # 3. Fallback advice generator
    # We pass mock chunks to verify logic
    mock_chunks = [
        {
            "id": "case_samira_kohli",
            "text": "consent must be real, voluntary, and informed",
            "metadata": {
                "title": "Samira Kohli v. Dr. Prabha Manchanda (2008) 2 SCC 1",
                "court": "Supreme Court of India",
                "year": "2008",
                "statutory_provisions": "Consumer Protection Act",
                "standard_of_care": "Informed Consent Doctrine"
            }
        }
    ]
    advice = generate_fallback_analysis(query, mock_chunks)
    assert "citations" in advice
    assert len(advice["citations"]) > 0
    assert "Samira Kohli" in advice["citations"][0]["case_name"]
    assert "consent" in advice["liability_assessment"].lower()
