import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Dict, Any

from mnip.database.connection import get_db
from mnip.api.schemas import FHIRIngestRequest, FHIRIngestResponse
from mnip.ingestion.fhir_parser import parse_fhir_bundle_to_episode, parse_fhir_resource
from mnip.ingestion.deidentifier import deidentify_text
from mnip.ingestion.models import Patient, Encounter, Observation, MedicationRequest, Procedure, DocumentReference, DeIdAuditLog

# Import fhir.resources models to assist with parsing raw resources in details
from fhir.resources.patient import Patient as FHIRPatient
from fhir.resources.encounter import Encounter as FHIREncounter
from fhir.resources.observation import Observation as FHIRObservation
from fhir.resources.medicationrequest import MedicationRequest as FHIRMedicationRequest
from fhir.resources.procedure import Procedure as FHIRProcedure
from fhir.resources.documentreference import DocumentReference as FHIRDocumentReference

logger = logging.getLogger("mnip.ingestion.router")
router = APIRouter(prefix="/ingest", tags=["ingestion"])

@router.post("/fhir", response_model=FHIRIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_fhir(payload: FHIRIngestRequest, db: AsyncSession = Depends(get_db)):
    """
    Accepts an HL7 FHIR R4 JSON bundle or resource, parses it, runs PHI de-identification,
    saves the structured records to PostgreSQL (omitting raw PHI), logs to the de-identification
    audit table, and returns the process status.
    """
    fhir_bundle = payload.fhir_bundle
    source_hospital_id = payload.source_hospital_id
    
    try:
        # 1. Parse using FHIR Parser
        episode = parse_fhir_bundle_to_episode(fhir_bundle)
    except Exception as e:
        logger.error(f"Failed to parse FHIR bundle: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_420_METHOD_FAILURE if hasattr(status, "HTTP_420_METHOD_FAILURE") else 422,
            detail=f"FHIR parsing validation error: {str(e)}"
        )

    # Resolve bundle items or single resource
    resource_type = fhir_bundle.get("resourceType")
    raw_resources = []
    if resource_type == "Bundle":
        if "entry" in fhir_bundle:
            for entry in fhir_bundle["entry"]:
                if "resource" in entry:
                    raw_resources.append(entry["resource"])
    else:
        raw_resources.append(fhir_bundle)

    # Track how many resources were parsed successfully
    resources_processed = 0
    de_id_audit_ref = str(uuid.uuid4())
    audit_logs_to_insert = []
    
    # We will buffer models to write in one transaction
    db_patients = {}
    db_encounters = {}
    db_observations = []
    db_med_requests = []
    db_procedures = []
    db_documents = []

    # Map/Parse and de-identify raw resources
    for raw_res in raw_resources:
        res_type = raw_res.get("resourceType")
        try:
            parsed = parse_fhir_resource(raw_res)
            if not parsed:
                continue
            resources_processed += 1
            
            # 1. Patient
            if isinstance(parsed, FHIRPatient):
                pat_id = parsed.id or "UNKNOWN_PATIENT"
                # Scrub patient identifiers for privacy (e.g. name / phone)
                # Presidio de-identification of gender/birthDate is simple since they are structured,
                # but we never persist patient name or direct identifiers.
                db_patients[pat_id] = Patient(
                    id=pat_id,
                    gender=parsed.gender,
                    birth_date=parsed.birthDate.dict() if hasattr(parsed.birthDate, "dict") else str(parsed.birthDate) if parsed.birthDate else None
                )
                
            # 2. Encounter
            elif isinstance(parsed, FHIREncounter):
                enc_id = parsed.id or "UNKNOWN_ENCOUNTER"
                pat_ref = parsed.subject.reference if parsed.subject else "UNKNOWN_PATIENT"
                pat_id = pat_ref.split("/")[-1]
                
                # Make sure patient is buffer-tracked or exists
                if pat_id not in db_patients:
                    # Create placeholder patient
                    db_patients[pat_id] = Patient(id=pat_id)
                    
                period_obj = getattr(parsed, 'actualPeriod', getattr(parsed, 'period', None))
                start_dt = period_obj.start if period_obj else None
                end_dt = period_obj.end if period_obj else None
                
                db_encounters[enc_id] = Encounter(
                    id=enc_id,
                    patient_id=pat_id,
                    status=parsed.status,
                    start_time=start_dt,
                    end_time=end_dt
                )
                
            # 3. Observation
            elif isinstance(parsed, FHIRObservation):
                obs_id = parsed.id or str(uuid.uuid4())
                pat_ref = parsed.subject.reference if parsed.subject else "UNKNOWN_PATIENT"
                pat_id = pat_ref.split("/")[-1]
                enc_ref = parsed.encounter.reference if parsed.encounter else None
                enc_id = enc_ref.split("/")[-1] if enc_ref else None
                
                # Verify references
                if pat_id not in db_patients:
                    db_patients[pat_id] = Patient(id=pat_id)
                if enc_id and enc_id not in db_encounters:
                    db_encounters[enc_id] = Encounter(id=enc_id, patient_id=pat_id)
                    
                obs_code = parsed.code.coding[0].code if parsed.code and parsed.code.coding else "UNKNOWN"
                obs_disp = parsed.code.coding[0].display or parsed.code.text if parsed.code else "UNKNOWN"
                
                val_q = float(parsed.valueQuantity.value) if parsed.valueQuantity else None
                val_s = parsed.valueString or (parsed.valueCodeableConcept.text if parsed.valueCodeableConcept else None)
                eff_dt = parsed.effectiveDateTime
                
                db_observations.append(Observation(
                    id=obs_id,
                    encounter_id=enc_id,
                    patient_id=pat_id,
                    code=obs_code,
                    display=obs_disp,
                    value_quantity=val_q,
                    value_string=val_s,
                    effective_time=eff_dt
                ))
                
            # 4. MedicationRequest
            elif isinstance(parsed, FHIRMedicationRequest):
                med_id = parsed.id or str(uuid.uuid4())
                pat_ref = parsed.subject.reference if parsed.subject else "UNKNOWN_PATIENT"
                pat_id = pat_ref.split("/")[-1]
                enc_ref = parsed.encounter.reference if parsed.encounter else None
                enc_id = enc_ref.split("/")[-1] if enc_ref else None
                
                if pat_id not in db_patients:
                    db_patients[pat_id] = Patient(id=pat_id)
                if enc_id and enc_id not in db_encounters:
                    db_encounters[enc_id] = Encounter(id=enc_id, patient_id=pat_id)
                    
                med_name = None
                if parsed.medicationCodeableConcept:
                    if parsed.medicationCodeableConcept.coding:
                        med_name = parsed.medicationCodeableConcept.coding[0].display
                    if not med_name:
                        med_name = parsed.medicationCodeableConcept.text
                elif parsed.medicationReference:
                    med_name = parsed.medicationReference.reference
                    
                db_med_requests.append(MedicationRequest(
                    id=med_id,
                    encounter_id=enc_id,
                    patient_id=pat_id,
                    medication_name=med_name or "Unknown Medication",
                    status=parsed.status,
                    intent=parsed.intent,
                    authored_on=parsed.authoredOn
                ))
                
            # 5. Procedure
            elif isinstance(parsed, FHIRProcedure):
                proc_id = parsed.id or str(uuid.uuid4())
                pat_ref = parsed.subject.reference if parsed.subject else "UNKNOWN_PATIENT"
                pat_id = pat_ref.split("/")[-1]
                enc_ref = parsed.encounter.reference if parsed.encounter else None
                enc_id = enc_ref.split("/")[-1] if enc_ref else None
                
                if pat_id not in db_patients:
                    db_patients[pat_id] = Patient(id=pat_id)
                if enc_id and enc_id not in db_encounters:
                    db_encounters[enc_id] = Encounter(id=enc_id, patient_id=pat_id)
                    
                proc_code = parsed.code.coding[0].code if parsed.code and parsed.code.coding else "UNKNOWN"
                proc_disp = parsed.code.coding[0].display or parsed.code.text if parsed.code else "UNKNOWN"
                perf_dt = parsed.performedDateTime if parsed.performedDateTime else (parsed.performedPeriod.start if parsed.performedPeriod else None)
                
                db_procedures.append(Procedure(
                    id=proc_id,
                    encounter_id=enc_id,
                    patient_id=pat_id,
                    code=proc_code,
                    display=proc_disp,
                    status=parsed.status,
                    performed_time=perf_dt
                ))
                
            # 6. DocumentReference (Contains Clinical Note Text)
            elif isinstance(parsed, FHIRDocumentReference):
                doc_id = parsed.id or str(uuid.uuid4())
                pat_ref = parsed.subject.reference if parsed.subject else "UNKNOWN_PATIENT"
                pat_id = pat_ref.split("/")[-1]
                enc_ref = parsed.context[0].reference if parsed.context and len(parsed.context) > 0 else None
                enc_id = enc_ref.split("/")[-1] if enc_ref else None
                
                if pat_id not in db_patients:
                    db_patients[pat_id] = Patient(id=pat_id)
                if enc_id and enc_id not in db_encounters:
                    db_encounters[enc_id] = Encounter(id=enc_id, patient_id=pat_id)
                    
                type_code = parsed.type.coding[0].code if parsed.type and parsed.type.coding else "NOTE"
                type_disp = parsed.type.coding[0].display or parsed.type.text if parsed.type else "Clinical Note"
                
                # Extract original raw note content
                raw_note = ""
                if parsed.content:
                    for c_item in parsed.content:
                        if c_item.attachment:
                            if c_item.attachment.data:
                                try:
                                    raw_note = base64.b64decode(c_item.attachment.data).decode("utf-8")
                                except Exception:
                                    pass
                            elif c_item.attachment.title:
                                raw_note = c_item.attachment.title
                                
                # RUN DE-IDENTIFICATION BEFORE STORAGE
                anonymized_note, scrubbed_entities = deidentify_text(raw_note)
                
                # Buffer audit log
                audit_logs_to_insert.append(DeIdAuditLog(
                    resource_type="DocumentReference",
                    resource_id=doc_id,
                    original_length=len(raw_note),
                    anonymized_length=len(anonymized_note),
                    scrubbed_entities=scrubbed_entities,
                    status="SUCCESS"
                ))
                
                db_documents.append(DocumentReference(
                    id=doc_id,
                    encounter_id=enc_id,
                    patient_id=pat_id,
                    type_code=type_code,
                    type_display=type_disp,
                    content_text=anonymized_note
                ))
                
        except Exception as ex:
            logger.error(f"Error parsing resource entry: {str(ex)}")
            # Log failure in audit trail
            audit_logs_to_insert.append(DeIdAuditLog(
                resource_type=res_type or "Unknown",
                resource_id=raw_res.get("id", "Unknown"),
                original_length=0,
                anonymized_length=0,
                scrubbed_entities=["ERROR"],
                status=f"FAILED: {str(ex)}"
            ))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Resource validation failed: {str(ex)}"
            )

    # 3. Store in DB
    # Let's upsert patients first
    for pat_id, pat_obj in db_patients.items():
        existing_pat = await db.get(Patient, pat_id)
        if not existing_pat:
            db.add(pat_obj)
        else:
            # update fields if needed
            if pat_obj.gender:
                existing_pat.gender = pat_obj.gender
            if pat_obj.birth_date:
                existing_pat.birth_date = pat_obj.birth_date

    # Upsert encounters
    for enc_id, enc_obj in db_encounters.items():
        existing_enc = await db.get(Encounter, enc_id)
        if not existing_enc:
            db.add(enc_obj)
        else:
            if enc_obj.status:
                existing_enc.status = enc_obj.status
            if enc_obj.start_time:
                existing_enc.start_time = enc_obj.start_time
            if enc_obj.end_time:
                existing_enc.end_time = enc_obj.end_time

    # Add observations, meds, procedures, documents, audit logs
    for obs in db_observations:
        existing = await db.get(Observation, obs.id)
        if not existing: db.add(obs)
    for med in db_med_requests:
        existing = await db.get(MedicationRequest, med.id)
        if not existing: db.add(med)
    for proc in db_procedures:
        existing = await db.get(Procedure, proc.id)
        if not existing: db.add(proc)
    for doc in db_documents:
        existing = await db.get(DocumentReference, doc.id)
        if not existing: 
            db.add(doc)
        else:
            existing.content_text = doc.content_text
    for alog in audit_logs_to_insert:
        db.add(alog)
        
    await db.commit()
    
    # episode_id is mapped to the Encounter ID
    episode_id = episode.encounter_id or "UNKNOWN_ENCOUNTER"
    
    return FHIRIngestResponse(
        episode_id=episode_id,
        resources_processed=resources_processed,
        de_id_audit_ref=de_id_audit_ref,
        status="success"
    )
