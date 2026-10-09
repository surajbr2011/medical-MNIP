import logging
from typing import Tuple, List
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger("backend.deidentifier")

# Initialize engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Custom Pattern Recognizers for Indian ABDM compliance
# 1. Ayushman Bharat Health Account (ABHA ID): 14-digit format XX-XXXX-XXXX-XXXX
abha_pattern = Pattern(
    name="abha_id_pattern",
    regex=r"\b\d{2}-\d{4}-\d{4}-\d{4}\b",
    score=0.9
)
abha_recognizer = PatternRecognizer(
    supported_entity="ABHA_ID",
    patterns=[abha_pattern]
)
analyzer.registry.add_recognizer(abha_recognizer)

# 2. Unique Hospital Identification Number (UHID) e.g., UHID-123456789 or UHID/2026/00456
uhid_pattern = Pattern(
    name="uhid_pattern",
    regex=r"\bUHID[-/]\d{4,10}([-/]\d{4,10})?\b",
    score=0.85
)
uhid_recognizer = PatternRecognizer(
    supported_entity="UHID",
    patterns=[uhid_pattern]
)
analyzer.registry.add_recognizer(uhid_recognizer)

# 3. Aadhaar Card Number: 12-digit format XXXX-XXXX-XXXX or XXXX XXXX XXXX
aadhaar_pattern = Pattern(
    name="aadhaar_pattern",
    regex=r"\b\d{4}[ -]\d{4}[ -]\d{4}\b",
    score=0.8
)
aadhaar_recognizer = PatternRecognizer(
    supported_entity="AADHAAR",
    patterns=[aadhaar_pattern]
)
analyzer.registry.add_recognizer(aadhaar_recognizer)

def deidentify_text(text: str) -> Tuple[str, List[str]]:
    """
    De-identify a clinical note, removing PHI (Names, Phones, Emails, ABHA IDs, UHIDs, Aadhaar).
    Returns a tuple of (anonymized_text, list_of_scrubbed_entities).
    """
    if not text:
        return "", []
        
    try:
        # Run Presidio analysis
        # We explicitly request our custom entities in addition to default ones
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "ABHA_ID", "UHID", "AADHAAR", "DATE_TIME"]
        )
        
        # Run anonymizer
        anonymized_res = anonymizer.anonymize(text=text, analyzer_results=results)
        
        # Retrieve the list of distinct scrubbed entities
        scrubbed_entities = list(set([r.entity_type for r in results]))
        
        logger.info(f"De-identified text length {len(text)} -> {len(anonymized_res.text)}. Entities: {scrubbed_entities}")
        return anonymized_res.text, scrubbed_entities
    except Exception as e:
        logger.error(f"Error in deidentification: {str(e)}")
        # Fallback to returning original text if something goes wrong, or a sanitized message
        # Since rule 5 says: "Never persist raw PHI", it's safer to raise or return a fully scrubbed string
        raise RuntimeError("De-identification process failed. Aborting database persistence to prevent PHI leak.") from e
