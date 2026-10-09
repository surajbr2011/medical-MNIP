import base64
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import ValidationError

# Import fhir.resources models
from fhir.resources.bundle import Bundle
from fhir.resources.patient import Patient
from fhir.resources.encounter import Encounter
from fhir.resources.observation import Observation
from fhir.resources.medicationrequest import MedicationRequest
from fhir.resources.procedure import Procedure
from fhir.resources.documentreference import DocumentReference

from backend.api.schemas import FHIREpisode

logger = logging.getLogger("backend.fhir_parser")

def clean_reference_id(ref: Optional[str]) -> Optional[str]:
    """Extracts '123' from 'Patient/123' or 'Encounter/456'"""
    if not ref:
        return None
    if "/" in ref:
        return ref.split("/")[-1]
    return ref

def parse_fhir_resource(resource_dict: Dict[str, Any]) -> Any:
    """Parses a raw dict into a validated FHIR resource object based on its resourceType."""
    resource_type = resource_dict.get("resourceType")
    if not resource_type:
        raise ValidationError("Missing 'resourceType' in FHIR resource.")
        
    mapping = {
        "Bundle": Bundle,
        "Patient": Patient,
        "Encounter": Encounter,
        "Observation": Observation,
        "MedicationRequest": MedicationRequest,
        "Procedure": Procedure,
        "DocumentReference": DocumentReference
    }
    
    klass = mapping.get(resource_type)
    if not klass:
        logger.warning(f"Unsupported FHIR resourceType: {resource_type}")
        return None
        
    # Validates schema automatically using fhir.resources parse_obj
    try:
        return klass.parse_obj(resource_dict)
    except Exception as e:
        logger.error(f"Validation failed for resource {resource_type}: {str(e)}")
        raise e

def parse_fhir_bundle_to_episode(fhir_bundle_dict: Dict[str, Any]) -> FHIREpisode:
    """
    Parses a FHIR Bundle or a single resource, extracts key clinical elements,
    and returns a validated FHIREpisode model.
    """
    resource_type = fhir_bundle_dict.get("resourceType")
    if not resource_type:
        raise ValueError("Invalid FHIR JSON: resourceType is missing.")
        
    resources = []
    
    if resource_type == "Bundle":
        bundle_obj = parse_fhir_resource(fhir_bundle_dict)
        if bundle_obj and bundle_obj.entry:
            for entry in bundle_obj.entry:
                if entry.resource:
                    # Convert to dict to validate individually
                    res_dict = entry.resource.dict()
                    parsed = parse_fhir_resource(res_dict)
                    if parsed:
                        resources.append(parsed)
    else:
        # Single resource type
        parsed = parse_fhir_resource(fhir_bundle_dict)
        if parsed:
            resources.append(parsed)
            
    episode = FHIREpisode()
    
    for res in resources:
        if isinstance(res, Patient):
            episode.patient_id = res.id
            
        elif isinstance(res, Encounter):
            episode.encounter_id = res.id
            if not episode.patient_id and res.subject:
                episode.patient_id = clean_reference_id(res.subject.reference)
                
        elif isinstance(res, DocumentReference):
            # Extract note content
            note_content = None
            if res.content:
                for content_item in res.content:
                    attachment = content_item.attachment
                    if attachment:
                        if attachment.data:
                            try:
                                decoded = base64.b64decode(attachment.data).decode("utf-8")
                                note_content = decoded
                            except Exception:
                                pass
                        elif attachment.title:
                            note_content = attachment.title
            if note_content:
                episode.clinical_notes.append(note_content)
                
            if not episode.patient_id and res.subject:
                episode.patient_id = clean_reference_id(res.subject.reference)
            if not episode.encounter_id and res.context and res.context.encounter:
                episode.encounter_id = clean_reference_id(res.context.encounter[0].reference)
                
        elif isinstance(res, MedicationRequest):
            med_name = None
            if res.medicationCodeableConcept:
                if res.medicationCodeableConcept.coding:
                    med_name = res.medicationCodeableConcept.coding[0].display or res.medicationCodeableConcept.coding[0].code
                if not med_name:
                    med_name = res.medicationCodeableConcept.text
            elif res.medicationReference:
                med_name = clean_reference_id(res.medicationReference.reference)
                
            if med_name:
                episode.medications.append(med_name)
                
            if not episode.patient_id and res.subject:
                episode.patient_id = clean_reference_id(res.subject.reference)
            if not episode.encounter_id and res.encounter:
                episode.encounter_id = clean_reference_id(res.encounter.reference)
                
        elif isinstance(res, Procedure):
            proc_name = None
            if res.code:
                if res.code.coding:
                    proc_name = res.code.coding[0].display or res.code.coding[0].code
                if not proc_name:
                    proc_name = res.code.text
                    
            if proc_name:
                episode.procedures.append(proc_name)
                
            if not episode.patient_id and res.subject:
                episode.patient_id = clean_reference_id(res.subject.reference)
            if not episode.encounter_id and res.encounter:
                episode.encounter_id = clean_reference_id(res.encounter.reference)
                
        elif isinstance(res, Observation):
            obs_key = None
            if res.code:
                if res.code.coding:
                    obs_key = res.code.coding[0].display or res.code.coding[0].code
                if not obs_key:
                    obs_key = res.code.text
            
            val = None
            if res.valueQuantity:
                val = float(res.valueQuantity.value)
            elif res.valueString:
                val = res.valueString
            elif res.valueBoolean is not None:
                val = bool(res.valueBoolean)
                
            if obs_key and val is not None:
                episode.observations[obs_key] = val
                
            if not episode.patient_id and res.subject:
                episode.patient_id = clean_reference_id(res.subject.reference)
            if not episode.encounter_id and res.encounter:
                episode.encounter_id = clean_reference_id(res.encounter.reference)

    # Ensure defaults or fallbacks are correct
    if not episode.patient_id:
        episode.patient_id = "UNKNOWN_PATIENT"
    if not episode.encounter_id:
        episode.encounter_id = "UNKNOWN_ENCOUNTER"
        
    return episode

if __name__ == "__main__":
    # Inline Unit Test
    print("Running inline tests for fhir_parser.py...")
    sample_patient_json = {
        "resourceType": "Patient",
        "id": "pat-101",
        "gender": "male",
        "birthDate": "1980-05-15"
    }
    
    sample_encounter_json = {
        "resourceType": "Encounter",
        "id": "enc-202",
        "status": "finished",
        "subject": {
            "reference": "Patient/pat-101"
        }
    }
    
    sample_obs_json = {
        "resourceType": "Observation",
        "id": "obs-303",
        "status": "final",
        "code": {
            "coding": [{
                "code": "8867-4",
                "display": "Heart rate"
            }],
            "text": "Heart rate"
        },
        "subject": {
            "reference": "Patient/pat-101"
        },
        "encounter": {
            "reference": "Encounter/enc-202"
        },
        "valueQuantity": {
            "value": 72.0,
            "unit": "beats/minute"
        }
    }

    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {"resource": sample_patient_json},
            {"resource": sample_encounter_json},
            {"resource": sample_obs_json}
        ]
    }
    
    try:
        episode = parse_fhir_bundle_to_episode(bundle)
        assert episode.patient_id == "pat-101", f"Expected pat-101, got {episode.patient_id}"
        assert episode.encounter_id == "enc-202", f"Expected enc-202, got {episode.encounter_id}"
        assert episode.observations.get("Heart rate") == 72.0, f"Expected 72.0, got {episode.observations.get('Heart rate')}"
        print("All inline tests passed successfully!")
    except Exception as e:
        print(f"Test failed: {str(e)}")
