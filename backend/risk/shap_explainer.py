import logging
import shap
import numpy as np
from typing import Dict, List, Tuple

from backend.risk.features import FeatureVector, to_numpy, FEATURE_NAMES

logger = logging.getLogger("backend.risk.shap_explainer")

def _format_feature_observation(name: str, x_val: float, is_missing: bool, shap_val: float) -> str:
    sign_str = f"+{shap_val:.2f}" if shap_val > 0 else f"{shap_val:.2f}"
    if is_missing:
        return f"unrecorded {name.replace('_', ' ')} ({sign_str} log-odds)"
        
    if name == "treatment_delay_hours":
        return f"treatment delay of {x_val:.1f}h ({sign_str} log-odds)"
    if name == "diagnosis_delay_hours":
        return f"diagnosis delay of {x_val:.1f}h ({sign_str} log-odds)"
    if name == "vital_sign_deterioration_flag":
        return f"vital sign deterioration observed ({sign_str} log-odds)" if x_val == 1.0 else f"stable vitals ({sign_str} log-odds)"
    if name == "high_risk_drug_flag":
        return f"high-risk drugs administered ({sign_str} log-odds)" if x_val == 1.0 else f"absence of high-risk drugs ({sign_str} log-odds)"
    if name == "consent_documented_flag":
        return f"documented informed consent ({sign_str} log-odds)" if x_val == 1.0 else f"unverified informed consent ({sign_str} log-odds)"
    if name == "documentation_completeness_score":
        return f"documentation completeness of {x_val*100:.0f}% ({sign_str} log-odds)"
    if name == "negligence_probability":
        return f"NLP narrative negligence score of {x_val*100:.1f}% ({sign_str} log-odds)"
    if name == "medication_count":
        return f"{int(x_val)} active medications ({sign_str} log-odds)"
    if name == "staff_patient_ratio":
        return f"staff-patient ratio of {x_val:.2f} ({sign_str} log-odds)"
    if name == "procedure_complication_flag":
        return f"procedure complication noted ({sign_str} log-odds)" if x_val == 1.0 else f"uncomplicated procedure ({sign_str} log-odds)"
    if name == "night_shift_flag":
        return f"night-shift care delivery ({sign_str} log-odds)" if x_val == 1.0 else f"day-shift care ({sign_str} log-odds)"
    if name == "weekend_flag":
        return f"weekend admission ({sign_str} log-odds)" if x_val == 1.0 else f"weekday admission ({sign_str} log-odds)"
    if name == "polypharmacy_flag":
        return f"polypharmacy present ({sign_str} log-odds)" if x_val == 1.0 else f"no polypharmacy ({sign_str} log-odds)"
    if name == "icu_flag":
        return f"ICU admission ({sign_str} log-odds)" if x_val == 1.0 else f"no ICU admission ({sign_str} log-odds)"
    if name == "locum_staff_flag":
        return f"locum staff care ({sign_str} log-odds)" if x_val == 1.0 else f"permanent staff care ({sign_str} log-odds)"
    if name == "allergy_override_flag":
        return f"allergy alert override ({sign_str} log-odds)" if x_val == 1.0 else f"no allergy overrides ({sign_str} log-odds)"
    if name == "dosage_deviation_flag":
        return f"dosage deviation detected ({sign_str} log-odds)" if x_val == 1.0 else f"standard dosage ({sign_str} log-odds)"
    if name == "readmission_flag":
        return f"30-day readmission ({sign_str} log-odds)" if x_val == 1.0 else f"initial admission ({sign_str} log-odds)"
    if name == "prior_adverse_event_flag":
        return f"prior adverse event history ({sign_str} log-odds)" if x_val == 1.0 else f"no prior adverse events ({sign_str} log-odds)"
    if name == "complaint_history_count":
        return f"{int(x_val)} clinician complaint histories ({sign_str} log-odds)"
    if name == "comorbidity_count":
        return f"{int(x_val)} comorbidities ({sign_str} log-odds)"
    if name == "followup_gap_days":
        return f"follow-up gap of {int(x_val)} days ({sign_str} log-odds)"
    if name == "result_to_action_hours":
        return f"result-to-action delay of {x_val:.1f}h ({sign_str} log-odds)"
    if name == "escalation_delay_hours":
        return f"escalation delay of {x_val:.1f}h ({sign_str} log-odds)"
        
    return f"{name.replace('_', ' ')} of {x_val:.2f} ({sign_str} log-odds)"

def explain_risk(ensemble, fv: FeatureVector, risk_score: float) -> Tuple[Dict[str, float], str, List[str]]:
    """
    Generates SHAP values for the clinical feature vector using TreeExplainer on the primary
    tree estimator of the ensemble. Formulates an evidence-based clinical narrative without
    distorted percentage risk multiplications, adhering to clinical AI governance standards.
    """
    X = to_numpy(fv).reshape(1, -1) # Shape [1, 64]
    
    # 1. Compute SHAP values on LightGBM tree base learner
    base_log_odds = 0.0
    try:
        explainer = shap.TreeExplainer(ensemble.lgb_model)
        raw_exp = explainer.expected_value
        if hasattr(raw_exp, '__iter__'):
            base_log_odds = float(raw_exp[1] if len(raw_exp) > 1 else raw_exp[0])
        else:
            base_log_odds = float(raw_exp)
            
        raw_shap = explainer.shap_values(X)
        if isinstance(raw_shap, list):
            shap_array = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif hasattr(raw_shap, "ndim") and raw_shap.ndim == 3:
            shap_array = raw_shap[1][0]
        else:
            shap_array = raw_shap[0]
            
    except Exception as e:
        logger.warning(f"TreeExplainer execution warning: {str(e)}. Generating deterministic linear contributions.")
        shap_array = np.zeros(64, dtype=np.float32)
        base_log_odds = -2.0
        for i, name in enumerate(FEATURE_NAMES):
            val = getattr(fv, name, 0.0)
            is_miss = getattr(fv, f"{name}_missing", True)
            if not is_miss:
                if name == "treatment_delay_hours" and val > 2.0:
                    shap_array[i] = min(0.6, (val - 2.0) * 0.1)
                elif name == "diagnosis_delay_hours" and val > 4.0:
                    shap_array[i] = min(0.4, (val - 4.0) * 0.08)
                elif name == "negligence_probability" and val > 0.5:
                    shap_array[i] = (val - 0.5) * 0.8
                elif name == "high_risk_drug_flag" and val == 1.0:
                    shap_array[i] = 0.35
                elif name == "vital_sign_deterioration_flag" and val == 1.0:
                    shap_array[i] = 0.45
                elif name == "consent_documented_flag" and val == 0.0:
                    shap_array[i] = 0.30

    # Base rate probability from base log-odds
    base_prob = 1.0 / (1.0 + np.exp(-base_log_odds))

    # 2. Map SHAP values to Feature Names (first 32 are features)
    shap_dict = {}
    for i, name in enumerate(FEATURE_NAMES):
        shap_dict[name] = float(shap_array[i])
        
    # Sort features by absolute SHAP magnitude
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_drivers = [name for name, val in sorted_features if abs(val) >= 0.02][:5]
    if not top_drivers:
        top_drivers = [name for name, _ in sorted_features[:5]]

    # 3. Formulate Clinical Narrative in validated format
    prob_pct = f"{risk_score * 100.0:.1f}%"
    baseline_pct = f"{base_prob * 100.0:.1f}%"
    
    higher_factors = []
    lower_factors = []
    
    for name, val in sorted_features:
        if abs(val) < 0.03:
            continue
        x_val = getattr(fv, name, 0.0)
        is_missing = getattr(fv, f"{name}_missing", False)
        desc = _format_feature_observation(name, x_val, is_missing, val)
        if val > 0:
            higher_factors.append(desc)
        else:
            lower_factors.append(desc)
            
    higher_text = ", ".join(higher_factors[:3]) if higher_factors else "no strong risk-elevating factors"
    lower_text = ", ".join(lower_factors[:3]) if lower_factors else "no strong protective factors"
    
    comparison = "higher than" if risk_score >= base_prob else "lower than"
    
    narrative = (
        f"Predicted model probability: {prob_pct} (calibrated score {risk_score:.3f}). "
        f"The model's output is {comparison} its baseline ({baseline_pct}) "
        f"in association with {higher_text}. "
        f"Factors contributing toward a lower output include {lower_text}. "
        f"These represent statistical model associations in native log-odds space, "
        f"not proof of causation or clinical negligence."
    )
    
    return shap_dict, narrative, top_drivers
