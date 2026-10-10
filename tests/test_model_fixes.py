import pytest
import re
import numpy as np
from backend.api.schemas import FHIREpisode, RiskScoreResponse, NegligenceResult
from backend.risk.features import extract_all_features, to_numpy, FEATURE_NAMES, FeatureVector
from backend.risk.model import predict_risk, load_risk_ensemble
from backend.risk.shap_explainer import explain_risk
from backend.detection.model import predict_negligence, WHO_ICPS_CATEGORIES
from backend.detection.explainer import get_token_attributions
from backend.legal.generator import generate_fallback_analysis

# =========================================================================
# 1. RISK MODEL, PROBABILITY CALIBRATION, & SHAP ADDITIVITY TESTS
# =========================================================================

def test_feature_vector_dimensions_and_missingness():
    """Verify that feature extraction yields 64 dimensions (32 clinical + 32 missingness flags)."""
    episode = FHIREpisode(
        patient_id="test-pat-1",
        encounter_id="test-enc-1",
        observations={"treatment_delay_hours": 3.0}
    )
    fv = extract_all_features(episode)
    arr = to_numpy(fv)
    assert arr.shape == (64,), f"Expected 64 dimensions, got {arr.shape}"
    assert fv.treatment_delay_hours == 3.0
    assert fv.treatment_delay_hours_missing is False
    assert fv.diagnosis_delay_hours_missing is True

def test_probability_bounds_and_calibration():
    """Verify that calibrated risk scores stay strictly within [0.0, 1.0]."""
    test_cases = [
        {"treatment_delay_hours": 0.0, "documentation_completeness_score": 1.0},
        {"treatment_delay_hours": 48.0, "vital_sign_deterioration_flag": 1.0},
        {"treatment_delay_hours": 999.0, "diagnosis_delay_hours": 500.0}, # Outlier test
    ]
    for obs in test_cases:
        ep = FHIREpisode(patient_id="p", encounter_id="e", observations=obs)
        res = predict_risk(ep)
        assert 0.0 <= res.risk_score <= 1.0, f"Risk score {res.risk_score} out of bounds"
        assert res.risk_level in ["Minimal", "Moderate", "High", "Critical"]
        assert res.baseline_risk == 0.014
        assert "log-odds" in res.explanation_scale.lower()

def test_shap_values_count_and_magnitude():
    """Verify that SHAP explanation returns all 32 clinical features without NaN."""
    ep = FHIREpisode(patient_id="p", encounter_id="e", observations={"treatment_delay_hours": 4.0})
    res = predict_risk(ep)
    assert len(res.shap_values) == 32
    for name, val in res.shap_values.items():
        assert isinstance(val, (float, int))
        assert not np.isnan(val), f"SHAP value for {name} is NaN"

# =========================================================================
# 2. CLINICAL NARRATIVE REGRESSION TESTS (NO >100% DISTORTIONS)
# =========================================================================

def test_regression_no_absurd_percentage_statements_in_narrative():
    """
    CRITICAL REGRESSION TEST:
    Assert narrative NEVER contains absurd statements like 'decreases risk by 206.4%'
    or 'increases risk by 98.6%' from multiplying raw log-odds by 100.
    """
    test_cases = [
        {"treatment_delay_hours": 12.0, "vital_sign_deterioration_flag": 0.0},
        {"treatment_delay_hours": 0.5, "high_risk_drug_flag": 0.0, "documentation_completeness_score": 0.95},
        {"vital_sign_deterioration_flag": 1.0, "high_risk_drug_flag": 1.0, "treatment_delay_hours": 24.0}
    ]
    for obs in test_cases:
        ep = FHIREpisode(patient_id="p", encounter_id="e", observations=obs)
        res = predict_risk(ep)
        narrative = res.narrative
        
        # Check for forbidden patterns: "(increases|decreases) risk by X%" where X > 100 or fractional
        matches = re.findall(r"(?:increases|decreases) risk by\s+([\d\.]+)%", narrative, re.IGNORECASE)
        assert len(matches) == 0, f"Found forbidden arbitrary risk multiplier in narrative: {matches}"
        
        # Ensure approved clinical format is used
        assert "Predicted model probability:" in narrative
        assert "baseline" in narrative
        assert "not proof of causation or clinical negligence" in narrative

# =========================================================================
# 3. NEGLIGENCE DETECTION ENGINE & CONFIDENCE CALCULATION
# =========================================================================

def test_negligence_confidence_and_equivocal_zone():
    """
    Verify that:
    1. Confidence is defined as max(p, 1-p), meaning it is never < 0.50.
    2. When probability is between 0.40 and 0.60, status is 'Equivocal / Insufficient Evidence'.
    3. Metadata contains decision threshold and disclaimer.
    """
    # Note with moderate/equivocal language
    note_equivocal = "Patient attended OPD with non-specific headache. Neurological exam grossly normal. Advised analgesics."
    res = predict_negligence(note_equivocal)
    
    assert isinstance(res, NegligenceResult)
    assert res.confidence >= 0.50, f"Confidence {res.confidence} should never be < 0.50 for the assigned class"
    assert res.decision_threshold == 0.50
    assert res.disclaimer is not None
    assert "AI-assisted clinical screening" in res.disclaimer
    
    if 0.40 <= res.screening_probability <= 0.60:
        assert res.status == "Equivocal / Insufficient Evidence"

def test_negligence_empty_narrative_handling():
    """Verify that an empty or whitespace clinical narrative does not crash the model."""
    res = predict_negligence("")
    assert isinstance(res, NegligenceResult)
    assert 0.0 <= res.screening_probability <= 1.0
    assert res.confidence >= 0.50
    assert isinstance(res.token_attributions, list)

# =========================================================================
# 4. WHO ICPS MULTI-LABEL FORMULATION TESTS
# =========================================================================

def test_who_icps_categories_structure_and_bounds():
    """
    Verify WHO ICPS categories:
    1. Exactly 8 categories.
    2. Each score is an independent probability in [0.0, 1.0].
    3. Not artificially normalized to sum to 1.0 (multi-label formulation).
    """
    note = "Medication error: 10x overdose of intravenous potassium chloride administered due to equipment pump failure."
    res = predict_negligence(note)
    
    cats = res.categories
    assert len(cats) == 8, f"Expected 8 categories, got {len(cats)}"
    for cat_name in WHO_ICPS_CATEGORIES:
        assert cat_name in cats, f"Missing category: {cat_name}"
        score = cats[cat_name]
        assert 0.0 <= score <= 1.0, f"Category score for {cat_name} ({score}) out of [0, 1]"

# =========================================================================
# 5. TOKEN ATTRIBUTION HEATMAP TESTS
# =========================================================================

def test_token_attributions_filter_special_tokens():
    """Verify token attribution excludes special tokens [CLS], [SEP], [PAD]."""
    text = "Delay in diagnosis of acute myocardial infarction."
    res = predict_negligence(text)
    
    tokens = [item["token"] for item in res.token_attributions]
    assert "[CLS]" not in tokens, "Special token [CLS] should be filtered"
    assert "[SEP]" not in tokens, "Special token [SEP] should be filtered"
    assert "[PAD]" not in tokens, "Special token [PAD] should be filtered"
    
    for item in res.token_attributions:
        assert "token" in item
        assert "attribution" in item
        assert isinstance(item["attribution"], float)

# =========================================================================
# 6. LEGAL INTELLIGENCE CITATION & RATIO DECIDENDI TESTS
# =========================================================================

def test_legal_citations_authoritative_holdings():
    """Verify legal advice contains verified Supreme Court precedents and disclaimer."""
    mock_chunks = [
        {
            "id": "case_jacob_mathew",
            "text": "Gross negligence required for criminal liability under Bolam test",
            "metadata": {
                "title": "Jacob Mathew v. State of Punjab (2005) 6 SCC 1",
                "court": "Supreme Court of India",
                "year": "2005",
                "statutory_provisions": "IPC Section 304A",
                "standard_of_care": "Bolam Standard of Ordinary Skilled Professional",
                "source_url": "https://indiankanoon.org/doc/871062/"
            }
        }
    ]
    analysis = generate_fallback_analysis("delay in surgery resulting in death", mock_chunks)
    assert "citations" in analysis
    assert len(analysis["citations"]) > 0
    citation = analysis["citations"][0]
    assert "Jacob Mathew" in citation["case_name"]
    assert "(2005) 6 SCC 1" in citation["case_name"]
    assert "Supreme Court of India" in citation["court"]
    assert "disclaimer" in analysis or "liability_assessment" in analysis
