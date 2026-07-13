import pytest
from mnip.detection.model import predict_negligence, batch_predict_negligence
from mnip.api.schemas import NegligenceResult

def test_predict_negligence():
    text = "The doctor delayed the appendectomy by 48 hours without clinical justification, leading to perforation."
    
    # Run prediction (loads model via singleton/cache)
    result = predict_negligence(text)
    
    assert isinstance(result, NegligenceResult)
    assert result.confidence >= 0.0 and result.confidence <= 1.0
    assert "Clinical process/procedure" in result.categories
    assert len(result.token_attributions) >= 0

def test_batch_predict_negligence():
    texts = [
        "Patient was given correct medication.",
        "Doctor performed unauthorized surgery without consent."
    ]
    
    results = batch_predict_negligence(texts)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, NegligenceResult)
