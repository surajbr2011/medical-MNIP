import numpy as np
from pydantic import BaseModel, Field
from typing import Tuple, Dict, Any, Optional

from mnip.api.schemas import FHIREpisode

# List of all 31 feature names for ordering
FEATURE_NAMES = [
    # Temporal (8)
    "treatment_delay_hours",
    "diagnosis_delay_hours",
    "followup_gap_days",
    "admission_to_procedure_hours",
    "symptom_to_presentation_hours",
    "discharge_delay_hours",
    "result_to_action_hours",
    "escalation_delay_hours",
    # NLP (4)
    "negligence_probability",
    "top_category",
    "anomaly_score",
    "documentation_completeness_score",
    # Medication (7)
    "medication_count",
    "polypharmacy_flag",
    "high_risk_drug_flag",
    "dosage_deviation_flag",
    "allergy_override_flag",
    "medication_complexity_index",
    "drug_interaction_count",
    # Admin (6)
    "night_shift_flag",
    "weekend_flag",
    "bed_occupancy_rate",
    "staff_patient_ratio",
    "icu_flag",
    "locum_staff_flag",
    # Clinical (6)
    "prior_adverse_event_flag",
    "comorbidity_count",
    "readmission_flag",
    "complaint_history_count",
    "procedure_complication_flag",
    "vital_sign_deterioration_flag",
    "consent_documented_flag"
]

class FeatureVector(BaseModel):
    # Features (31 float fields)
    treatment_delay_hours: float = Field(0.0, description="Hours elapsed between admission/incident and appropriate treatment.")
    diagnosis_delay_hours: float = Field(0.0, description="Hours elapsed between symptom onset/admission and definitive diagnosis.")
    followup_gap_days: float = Field(0.0, description="Number of days between discharge and scheduled follow-up appointment.")
    admission_to_procedure_hours: float = Field(0.0, description="Time delay in hours between patient admission and surgical/interventional procedure.")
    symptom_to_presentation_hours: float = Field(0.0, description="Time from symptom onset to presentation at emergency department.")
    discharge_delay_hours: float = Field(0.0, description="Administrative or clinical delay in hours during patient discharge.")
    result_to_action_hours: float = Field(0.0, description="Time in hours from lab/imaging result availability to clinician action.")
    escalation_delay_hours: float = Field(0.0, description="Delay in hours in escalating care to senior clinicians/ICU after deterioration.")
    
    negligence_probability: float = Field(0.0, description="Probability of negligence predicted by Bio_ClinicalBERT from clinical notes.")
    top_category: float = Field(0.0, description="Encoded index (0-7) of the predicted WHO ICPS category.")
    anomaly_score: float = Field(0.0, description="Quantified deviation of patient care sequence from standard protocols.")
    documentation_completeness_score: float = Field(0.0, description="Assess percentage of required note fields completed (0.0 to 1.0).")
    
    medication_count: float = Field(0.0, description="Total number of active medications prescribed during the episode.")
    polypharmacy_flag: float = Field(0.0, description="Binary flag indicating polypharmacy (usually > 5 concurrent drugs).")
    high_risk_drug_flag: float = Field(0.0, description="Binary flag indicating prescription of high-alert medications (e.g. insulin, anticoagulants).")
    dosage_deviation_flag: float = Field(0.0, description="Binary flag indicating mismatch between prescribed dose and typical range.")
    allergy_override_flag: float = Field(0.0, description="Binary flag indicating clinician override of electronic drug-allergy alert.")
    medication_complexity_index: float = Field(0.0, description="Calculated score reflecting frequency, route, and administration complexity.")
    drug_interaction_count: float = Field(0.0, description="Count of potential severe drug-drug interactions detected.")
    
    night_shift_flag: float = Field(0.0, description="Binary flag indicating if critical decisions/procedures occurred during night shift.")
    weekend_flag: float = Field(0.0, description="Binary flag indicating if admission/procedure occurred on a weekend.")
    bed_occupancy_rate: float = Field(0.0, description="Hospital/ward occupancy rate percentage at the time of patient stay.")
    staff_patient_ratio: float = Field(0.0, description="Nurse/physician to patient ratio in the department during stay.")
    icu_flag: float = Field(0.0, description="Binary flag indicating if patient required ICU admission.")
    locum_staff_flag: float = Field(0.0, description="Binary flag indicating if care was managed by temporary or locum staff.")
    
    prior_adverse_event_flag: float = Field(0.0, description="Binary flag indicating historical adverse events or complaints for the patient.")
    comorbidity_count: float = Field(0.0, description="Count of documented chronic comorbidities (e.g., diabetes, hypertension).")
    readmission_flag: float = Field(0.0, description="Binary flag indicating readmission within 30 days of previous discharge.")
    complaint_history_count: float = Field(0.0, description="Count of prior medico-legal complaints against the primary clinician.")
    procedure_complication_flag: float = Field(0.0, description="Binary flag indicating standard intra- or post-operative complications.")
    vital_sign_deterioration_flag: float = Field(0.0, description="Binary flag indicating MEWS/PEWS trigger indicating patient deterioration.")
    consent_documented_flag: float = Field(0.0, description="Binary flag indicating if informed consent document was verified.")

    # Missingness Flags (31 boolean fields)
    treatment_delay_hours_missing: bool = True
    diagnosis_delay_hours_missing: bool = True
    followup_gap_days_missing: bool = True
    admission_to_procedure_hours_missing: bool = True
    symptom_to_presentation_hours_missing: bool = True
    discharge_delay_hours_missing: bool = True
    result_to_action_hours_missing: bool = True
    escalation_delay_hours_missing: bool = True
    
    negligence_probability_missing: bool = True
    top_category_missing: bool = True
    anomaly_score_missing: bool = True
    documentation_completeness_score_missing: bool = True
    
    medication_count_missing: bool = True
    polypharmacy_flag_missing: bool = True
    high_risk_drug_flag_missing: bool = True
    dosage_deviation_flag_missing: bool = True
    allergy_override_flag_missing: bool = True
    medication_complexity_index_missing: bool = True
    drug_interaction_count_missing: bool = True
    
    night_shift_flag_missing: bool = True
    weekend_flag_missing: bool = True
    bed_occupancy_rate_missing: bool = True
    staff_patient_ratio_missing: bool = True
    icu_flag_missing: bool = True
    locum_staff_flag_missing: bool = True
    
    prior_adverse_event_flag_missing: bool = True
    comorbidity_count_missing: bool = True
    readmission_flag_missing: bool = True
    complaint_history_count_missing: bool = True
    procedure_complication_flag_missing: bool = True
    vital_sign_deterioration_flag_missing: bool = True
    consent_documented_flag_missing: bool = True

# Continuous feature medians for imputation
MEDIANS = {
    "treatment_delay_hours": 2.0,
    "diagnosis_delay_hours": 4.0,
    "followup_gap_days": 14.0,
    "admission_to_procedure_hours": 6.0,
    "symptom_to_presentation_hours": 12.0,
    "discharge_delay_hours": 1.0,
    "result_to_action_hours": 1.5,
    "escalation_delay_hours": 0.5,
    "negligence_probability": 0.1,
    "top_category": 0.0,
    "anomaly_score": 0.05,
    "documentation_completeness_score": 0.9,
    "medication_count": 3.0,
    "medication_complexity_index": 1.0,
    "drug_interaction_count": 0.0,
    "bed_occupancy_rate": 82.0,
    "staff_patient_ratio": 0.25,
    "comorbidity_count": 1.0,
    "complaint_history_count": 0.0
}

# Helper to extract a continuous or flag feature from observations
def get_obs_value(episode: FHIREpisode, key: str, is_flag: bool = False) -> Tuple[float, bool]:
    """Returns (value, is_missing). Uses MEDIANS or 0.0 for imputation."""
    val = episode.observations.get(key)
    if val is None:
        imputed_val = 0.0 if is_flag else MEDIANS.get(key, 0.0)
        return float(imputed_val), True
    return float(val), False

# Feature extractor functions
def get_treatment_delay(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "treatment_delay_hours")

def get_diagnosis_delay(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "diagnosis_delay_hours")

def get_followup_gap(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "followup_gap_days")

def get_admission_to_procedure(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "admission_to_procedure_hours")

def get_symptom_to_presentation(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "symptom_to_presentation_hours")

def get_discharge_delay(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "discharge_delay_hours")

def get_result_to_action(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "result_to_action_hours")

def get_escalation_delay(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "escalation_delay_hours")

# NLP Features
def get_negligence_probability(episode: FHIREpisode) -> Tuple[float, bool]:
    # If notes are present, we can look up predicted negligence or default to median
    return get_obs_value(episode, "negligence_probability")

def get_top_category(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "top_category")

def get_anomaly_score(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "anomaly_score")

def get_documentation_completeness(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "documentation_completeness_score")

# Medication Features
def get_medication_count(episode: FHIREpisode) -> Tuple[float, bool]:
    # We can count the medications in the parsed list directly!
    if episode.medications:
        return float(len(episode.medications)), False
    return get_obs_value(episode, "medication_count")

def get_polypharmacy_flag(episode: FHIREpisode, med_count: float, med_missing: bool) -> Tuple[float, bool]:
    if not med_missing:
        return (1.0 if med_count > 5 else 0.0), False
    return get_obs_value(episode, "polypharmacy_flag", is_flag=True)

def get_high_risk_drug_flag(episode: FHIREpisode) -> Tuple[float, bool]:
    # Look for known high risk drugs
    high_risk_keywords = ["insulin", "heparin", "warfarin", "digoxin", "morphine", "fentanyl"]
    if episode.medications:
        has_high_risk = any(any(kw in med.lower() for kw in high_risk_keywords) for med in episode.medications)
        return (1.0 if has_high_risk else 0.0), False
    return get_obs_value(episode, "high_risk_drug_flag", is_flag=True)

def get_dosage_deviation_flag(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "dosage_deviation_flag", is_flag=True)

def get_allergy_override_flag(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "allergy_override_flag", is_flag=True)

def get_medication_complexity_index(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "medication_complexity_index")

def get_drug_interaction_count(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "drug_interaction_count")

# Admin Features
def get_night_shift_flag(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "night_shift_flag", is_flag=True)

def get_weekend_flag(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "weekend_flag", is_flag=True)

def get_bed_occupancy_rate(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "bed_occupancy_rate")

def get_staff_patient_ratio(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "staff_patient_ratio")

def get_icu_flag(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "icu_flag", is_flag=True)

def get_locum_staff_flag(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "locum_staff_flag", is_flag=True)

# Clinical Features
def get_prior_adverse_event_flag(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "prior_adverse_event_flag", is_flag=True)

def get_comorbidity_count(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "comorbidity_count")

def get_readmission_flag(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "readmission_flag", is_flag=True)

def get_complaint_history_count(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "complaint_history_count")

def get_procedure_complication_flag(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "procedure_complication_flag", is_flag=True)

def get_vital_sign_deterioration_flag(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "vital_sign_deterioration_flag", is_flag=True)

def get_consent_documented_flag(episode: FHIREpisode) -> Tuple[float, bool]:
    return get_obs_value(episode, "consent_documented_flag", is_flag=True)


def extract_all_features(episode: FHIREpisode) -> FeatureVector:
    """Extracts all 31 features + 31 missingness flags into a FeatureVector Pydantic model."""
    
    treatment_val, treatment_miss = get_treatment_delay(episode)
    diag_val, diag_miss = get_diagnosis_delay(episode)
    follow_val, follow_miss = get_followup_gap(episode)
    adm_proc_val, adm_proc_miss = get_admission_to_procedure(episode)
    sym_pres_val, sym_pres_miss = get_symptom_to_presentation(episode)
    dis_val, dis_miss = get_discharge_delay(episode)
    res_val, res_miss = get_result_to_action(episode)
    esc_val, esc_miss = get_escalation_delay(episode)
    
    negl_val, negl_miss = get_negligence_probability(episode)
    top_cat_val, top_cat_miss = get_top_category(episode)
    anom_val, anom_miss = get_anomaly_score(episode)
    doc_val, doc_miss = get_documentation_completeness(episode)
    
    med_val, med_miss = get_medication_count(episode)
    poly_val, poly_miss = get_polypharmacy_flag(episode, med_val, med_miss)
    hr_val, hr_miss = get_high_risk_drug_flag(episode)
    dose_val, dose_miss = get_dosage_deviation_flag(episode)
    allergy_val, allergy_miss = get_allergy_override_flag(episode)
    med_comp_val, med_comp_miss = get_medication_complexity_index(episode)
    drug_int_val, drug_int_miss = get_drug_interaction_count(episode)
    
    night_val, night_miss = get_night_shift_flag(episode)
    week_val, week_miss = get_weekend_flag(episode)
    bed_val, bed_miss = get_bed_occupancy_rate(episode)
    staff_val, staff_miss = get_staff_patient_ratio(episode)
    icu_val, icu_miss = get_icu_flag(episode)
    locum_val, locum_miss = get_locum_staff_flag(episode)
    
    prior_val, prior_miss = get_prior_adverse_event_flag(episode)
    comorb_val, comorb_miss = get_comorbidity_count(episode)
    readm_val, readm_miss = get_readmission_flag(episode)
    compl_val, compl_miss = get_complaint_history_count(episode)
    proc_comp_val, proc_comp_miss = get_procedure_complication_flag(episode)
    vital_val, vital_miss = get_vital_sign_deterioration_flag(episode)
    consent_val, consent_miss = get_consent_documented_flag(episode)
    
    return FeatureVector(
        treatment_delay_hours=treatment_val,
        diagnosis_delay_hours=diag_val,
        followup_gap_days=follow_val,
        admission_to_procedure_hours=adm_proc_val,
        symptom_to_presentation_hours=sym_pres_val,
        discharge_delay_hours=dis_val,
        result_to_action_hours=res_val,
        escalation_delay_hours=esc_val,
        
        negligence_probability=negl_val,
        top_category=top_cat_val,
        anomaly_score=anom_val,
        documentation_completeness_score=doc_val,
        
        medication_count=med_val,
        polypharmacy_flag=poly_val,
        high_risk_drug_flag=hr_val,
        dosage_deviation_flag=dose_val,
        allergy_override_flag=allergy_val,
        medication_complexity_index=med_comp_val,
        drug_interaction_count=drug_int_val,
        
        night_shift_flag=night_val,
        weekend_flag=week_val,
        bed_occupancy_rate=bed_val,
        staff_patient_ratio=staff_val,
        icu_flag=icu_val,
        locum_staff_flag=locum_val,
        
        prior_adverse_event_flag=prior_val,
        comorbidity_count=comorb_val,
        readmission_flag=readm_val,
        complaint_history_count=compl_val,
        procedure_complication_flag=proc_comp_val,
        vital_sign_deterioration_flag=vital_val,
        consent_documented_flag=consent_val,
        
        # Flags
        treatment_delay_hours_missing=treatment_miss,
        diagnosis_delay_hours_missing=diag_miss,
        followup_gap_days_missing=follow_miss,
        admission_to_procedure_hours_missing=adm_proc_miss,
        symptom_to_presentation_hours_missing=sym_pres_miss,
        discharge_delay_hours_missing=dis_miss,
        result_to_action_hours_missing=res_miss,
        escalation_delay_hours_missing=esc_miss,
        
        negligence_probability_missing=negl_miss,
        top_category_missing=top_cat_miss,
        anomaly_score_missing=anom_miss,
        documentation_completeness_score_missing=doc_miss,
        
        medication_count_missing=med_miss,
        polypharmacy_flag_missing=poly_miss,
        high_risk_drug_flag_missing=hr_miss,
        dosage_deviation_flag_missing=dose_miss,
        allergy_override_flag_missing=allergy_miss,
        medication_complexity_index_missing=med_comp_miss,
        drug_interaction_count_missing=drug_int_miss,
        
        night_shift_flag_missing=night_miss,
        weekend_flag_missing=week_miss,
        bed_occupancy_rate_missing=bed_miss,
        staff_patient_ratio_missing=staff_miss,
        icu_flag_missing=icu_miss,
        locum_staff_flag_missing=locum_miss,
        
        prior_adverse_event_flag_missing=prior_miss,
        comorbidity_count_missing=comorb_miss,
        readmission_flag_missing=readm_miss,
        complaint_history_count_missing=compl_miss,
        procedure_complication_flag_missing=proc_comp_miss,
        vital_sign_deterioration_flag_missing=vital_miss,
        consent_documented_flag_missing=consent_miss
    )

def to_numpy(fv: FeatureVector) -> np.ndarray:
    """Converts a FeatureVector Pydantic model to a 1D float numpy array of shape [62]."""
    features = []
    missing_flags = []
    
    for name in FEATURE_NAMES:
        features.append(getattr(fv, name))
        missing_flags.append(float(getattr(fv, f"{name}_missing")))
        
    return np.array(features + missing_flags, dtype=np.float32)
