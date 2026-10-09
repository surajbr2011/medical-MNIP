import os
import json
import logging
import chromadb
from chromadb.utils import embedding_functions

from backend.config import settings

logger = logging.getLogger("backend.legal.ingestion")

# Standard Indian Medico-Legal Precedents
MOCK_LEGAL_CORPUS = [
    {
        "id": "case_jacob_mathew",
        "title": "Jacob Mathew v. State of Punjab (2005) 6 SCC 1",
        "court": "Supreme Court of India",
        "year": "2005",
        "excerpt": "Negligence in the context of the medical profession necessarily calls for a treatment with a difference. A simple lack of care, an error of judgment or an accident, is not proof of negligence on the part of a medical professional. So long as a doctor follows a practice acceptable to the medical profession of that day, he cannot be held liable for negligence merely because a better alternative course of treatment was available or because a more skilled doctor would not have chosen to follow or adopt the practice. Section 304A IPC requires proof of rashness or gross negligence.",
        "statutory_provisions": "Section 304A of Indian Penal Code (IPC), Section 80 and 88 of IPC.",
        "standard_of_care": "Bolam Test - standard of an ordinary skilled man exercising and professing to have that special skill."
    },
    {
        "id": "case_martin_dsouza",
        "title": "Martin F. D'Souza v. Mohd. Ishfaq (2009) 3 SCC 549",
        "court": "Supreme Court of India",
        "year": "2009",
        "excerpt": "A medical practitioner faces harassment from disgruntled patients. The Bolam Test applies. The court must consult a medical expert panel of independent specialists before issuing notice to a doctor accused of medical negligence to avoid unnecessary harassment. The standard of care required is that of a reasonably competent practitioner.",
        "statutory_provisions": "Consumer Protection Act, Section 2(1)(g) (Deficiency of Service).",
        "standard_of_care": "A professional is not liable for negligence if he performs his duty with a reasonable degree of care and competence."
    },
    {
        "id": "case_samira_kohli",
        "title": "Samira Kohli v. Dr. Prabha Manchanda (2008) 2 SCC 1",
        "court": "Supreme Court of India",
        "year": "2008",
        "excerpt": "A doctor cannot perform an additional surgery or procedure (e.g., hysterectomy during diagnostic laparoscopy) without explicit consent from the patient, even if it is beneficial, unless there is an immediate threat to life. Consent must be real, voluntary, and informed. Performing unauthorized surgery constitutes a tort of assault/battery and deficiency of service.",
        "statutory_provisions": "Consumer Protection Act, Law of Torts (Assault and Battery).",
        "standard_of_care": "Informed Consent Doctrine - patient has the right to decide what happens to their body."
    },
    {
        "id": "case_vp_shantha",
        "title": "Indian Medical Association v. V.P. Shantha (1995) 6 SCC 651",
        "court": "Supreme Court of India",
        "year": "1995",
        "excerpt": "Medical services rendered by doctors and hospitals constitute a 'service' under Section 2(1)(o) of the Consumer Protection Act. Free treatment in government hospitals or to the poor does not change its nature if others are charged. Patients can sue doctors for deficiency of service in Consumer Forums, which are faster and cheaper than Civil Courts.",
        "statutory_provisions": "Section 2(1)(o) and Section 2(1)(g) of Consumer Protection Act.",
        "standard_of_care": "Doctors must provide competent professional services. Failure to do so constitutes deficiency of service."
    },
    {
        "id": "case_laxman_joshi",
        "title": "Dr. Laxman Balkrishna Joshi v. Dr. Trimbak Bapu Godbole AIR 1969 SC 128",
        "court": "Supreme Court of India",
        "year": "1969",
        "excerpt": "A doctor has certain duties to a patient who consults him: (a) a duty of care in deciding whether to undertake the case, (b) a duty of care in deciding what treatment to give, and (c) a duty of care in the administration of that treatment. A breach of any of these duties gives a right of action for negligence to the patient.",
        "statutory_provisions": "Law of Torts - Negligence.",
        "standard_of_care": "Adherence to the duties of undertaking, deciding, and administering care with reasonable skill."
    },
    {
        "id": "case_kunhal_kutty",
        "title": "Poonam Verma v. Ashwin Patel (1996) 4 SCC 332",
        "court": "Supreme Court of India",
        "year": "1996",
        "excerpt": "A practitioner of Homeopathic medicine who prescribes Allopathic drugs is guilty of negligence per se. A person who does not have knowledge of a particular system of medicine but practices in that system is a quack and acts with rashness, making them liable for criminal and civil consequences.",
        "statutory_provisions": "Section 15(3) of Indian Medical Council Act, 1956.",
        "standard_of_care": "Practicing outside one's registered medical system constitutes negligence per se."
    }
]

def get_chroma_collection():
    """Initializes and returns the ChromaDB client and collection."""
    os.makedirs(settings.CHROMADB_PATH, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
    
    # Embedding model all-MiniLM-L6-v2
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    collection = chroma_client.get_or_create_collection(
        name="legal_corpus",
        embedding_function=emb_fn
    )
    return collection

def ingest_corpus():
    """Ingests the medical-legal documents into ChromaDB."""
    collection = get_chroma_collection()
    
    # Retrieve existing document IDs to avoid duplicate insertions
    existing = collection.get()
    existing_ids = set(existing.get("ids", []))
    
    ids = []
    documents = []
    metadatas = []
    
    for case in MOCK_LEGAL_CORPUS:
        doc_id = case["id"]
        if doc_id not in existing_ids:
            # We store the main excerpt and summary text in document chunk
            doc_text = f"Title: {case['title']}\nCourt: {case['court']}\nYear: {case['year']}\nExcerpt: {case['excerpt']}\nStatutes: {case['statutory_provisions']}\nStandard of Care: {case['standard_of_care']}"
            
            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append({
                "title": case["title"],
                "court": case["court"],
                "year": case["year"],
                "statutory_provisions": case["statutory_provisions"],
                "standard_of_care": case["standard_of_care"]
            })
            
    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully ingested {len(ids)} Indian medico-legal case documents into ChromaDB.")
    else:
        print("Legal corpus is already fully ingested in ChromaDB.")

if __name__ == "__main__":
    ingest_corpus()
