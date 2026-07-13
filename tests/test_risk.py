import pytest
import numpy as np
from mnip.api.schemas import FHIREpisode
from mnip.risk.features import extract_all_features, to_numpy, FeatureVector
from mnip.risk.model import predict_risk, load_risk_ensemble

def test_feature_extraction():
    episode = FHIREpisode(
        patient_id="pat-1",
        encounter_id="enc-1",
        medications=["aspirin", "insulin"],
        observations={
            "treatment_delay_hours": 4.5,
            "vital_sign_deterioration_flag": 1.0
        }
    )
    
    fv = extract_all_features(episode)
    assert isinstance(fv, FeatureVector)
    assert fv.treatment_delay_hours == 4.5
    assert fv.vital_sign_deterioration_flag == 1.0
    assert fv.high_risk_drug_flag == 1.0 # insulin keyword match
    assert fv.treatment_delay_hours_missing is False
    assert fv.diagnosis_delay_hours_missing is True
    
    # Check shape
    arr = to_numpy(fv)
    assert arr.shape == (62,)

def test_predict_risk():
    episode = FHIREpisode(
        patient_id="pat-1",
        encounter_id="enc-1",
        observations={"treatment_delay_hours": 2.0}
    )
    
    result = predict_risk(episode)
    assert result.risk_score >= 0.0 and result.risk_score <= 1.0
    assert result.risk_level in ["Minimal", "Moderate", "High", "Critical"]
    assert len(result.shap_values) == 31
    assert isinstance(result.narrative, str)
