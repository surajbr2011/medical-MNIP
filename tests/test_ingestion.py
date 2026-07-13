import pytest
from mnip.ingestion.fhir_parser import parse_fhir_bundle_to_episode, clean_reference_id
from mnip.ingestion.deidentifier import deidentify_text
from mnip.api.schemas import FHIREpisode

def test_clean_reference_id():
    assert clean_reference_id("Patient/123") == "123"
    assert clean_reference_id("Encounter/456") == "456"
    assert clean_reference_id("789") == "789"
    assert clean_reference_id(None) is None

def test_deidentify_text():
    raw_text = "Patient Rahul Sharma (Aadhar 1234-5678-9012, ABHA 12-3456-7890-1234) presented at UHID-998877."
    anonymized, entities = deidentify_text(raw_text)
    
    # Check that identifiers are removed
    assert "Rahul Sharma" not in anonymized
    assert "1234-5678-9012" not in anonymized
    assert "12-3456-7890-1234" not in anonymized
    assert "UHID-998877" not in anonymized
    
    # Check that Presidio identified relevant entities
    assert len(entities) > 0

def test_parse_fhir_bundle_to_episode():
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-001",
                    "gender": "female"
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-100",
                    "status": "finished",
                    "subject": {"reference": "Patient/pat-001"}
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-1",
                    "status": "final",
                    "code": {
                        "coding": [{"code": "temp", "display": "Body Temperature"}]
                    },
                    "subject": {"reference": "Patient/pat-001"},
                    "encounter": {"reference": "Encounter/enc-100"},
                    "valueQuantity": {"value": 38.5}
                }
            }
        ]
    }
    
    episode = parse_fhir_bundle_to_episode(bundle)
    assert isinstance(episode, FHIREpisode)
    assert episode.patient_id == "pat-001"
    assert episode.encounter_id == "enc-100"
    assert episode.observations.get("Body Temperature") == 38.5
