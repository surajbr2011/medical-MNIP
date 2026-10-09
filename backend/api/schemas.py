from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Module 1: Ingestion ---
class FHIRIngestRequest(BaseModel):
    fhir_bundle: Dict[str, Any]
    source_hospital_id: str

class FHIRIngestResponse(BaseModel):
    episode_id: str
    resources_processed: int
    de_id_audit_ref: str
    status: str

class FHIREpisode(BaseModel):
    patient_id: Optional[str] = None
    encounter_id: Optional[str] = None
    clinical_notes: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    procedures: List[str] = Field(default_factory=list)
    observations: Dict[str, Any] = Field(default_factory=dict)

# --- Module 2: Detection ---
class NegligenceDetectRequest(BaseModel):
    episode_id: Optional[str] = "ep-default"
    note_text: Optional[str] = None
    clinical_note: Optional[str] = None

class NegligenceResult(BaseModel):
    negligent: bool
    confidence: float
    categories: Dict[str, float]
    token_attributions: List[Dict[str, Any]]

# --- Module 3: Risk Scoring ---
class RiskScoreRequest(BaseModel):
    episode_id: str

class RiskScoreResponse(BaseModel):
    risk_score: float
    risk_level: str
    shap_values: Dict[str, float]
    narrative: str
    top_drivers: List[str]

# --- Module 4: Legal RAG ---
class LegalQueryRequest(BaseModel):
    incident_description: Optional[str] = None
    query: Optional[str] = None
    top_k: int = 5

class LegalQueryResponse(BaseModel):
    citations: List[Dict[str, Any]]
    statutory_provisions: List[str]
    standard_of_care_summary: str
    liability_assessment: str

# --- Combined Episode Report ---
class FullbackendReport(BaseModel):
    episode_id: str
    ingestion: Dict[str, Any]
    detection: Optional[NegligenceResult] = None
    risk: Optional[RiskScoreResponse] = None
    legal: Optional[LegalQueryResponse] = None
