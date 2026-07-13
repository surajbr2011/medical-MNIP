import asyncio
import base64
from mnip.ingestion.router import ingest_fhir
from mnip.api.schemas import FHIRIngestRequest
from mnip.database.connection import AsyncSessionLocal

payload = {
    'fhir_bundle': {
        'resourceType': 'Bundle',
        'type': 'transaction',
        'entry': [
            {
                'resource': {
                    'resourceType': 'Patient',
                    'id': 'pat-temp',
                    'gender': 'unknown'
                }
            },
            {
                'resource': {
                    'resourceType': 'Encounter',
                    'id': 'enc-temp',
                    'status': 'finished',
                    'subject': {'reference': 'Patient/pat-temp'}
                }
            },
            {
                'resource': {
                    'resourceType': 'DocumentReference',
                    'id': 'doc-temp',
                    'status': 'current',
                    'subject': {'reference': 'Patient/pat-temp'},
                    'context': [
                        {'reference': 'Encounter/enc-temp'}
                    ],
                    'content': [{
                        'attachment': {
                            'title': 'Patient presented with acute appendicitis...'
                        }
                    }]
                }
            }
        ]
    },
    'source_hospital_id': 'HOSP-IND-99'
}

async def run_test():
    req = FHIRIngestRequest(**payload)
    async with AsyncSessionLocal() as db:
        try:
            res = await ingest_fhir(req, db)
            print("Success!", res)
        except Exception as e:
            import traceback
            traceback.print_exc()

asyncio.run(run_test())
