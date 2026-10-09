import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/mnip"
    DATABASE_SYNC_URL: str = "postgresql://postgres:postgres@localhost:5432/mnip"

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "mnip_neo4j"

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    LLM_MODEL: str = "mistral"

    # ChromaDB
    CHROMADB_PATH: str = "data/chromadb"

    # Models
    CLINICALBERT_MODEL_PATH: str = "emilyalsentzer/Bio_ClinicalBERT"
    RISK_ENSEMBLE_PATH: str = "models/risk_ensemble"

    # API Configuration
    JWT_SECRET: str = "supersecretkey"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **values):
        super().__init__(**values)
        
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        
        # Clean DATABASE_URL
        if self.DATABASE_URL.startswith("postgresql://"):
            self.DATABASE_URL = self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
            
        try:
            parsed = urlparse(self.DATABASE_URL)
            if parsed.query:
                query_params = parse_qs(parsed.query)
                query_params.pop("schema", None) # Remove schema parameter
                new_query = urlencode(query_params, doseq=True)
                self.DATABASE_URL = urlunparse(parsed._replace(query=new_query))
        except Exception:
            pass

        # Clean DATABASE_SYNC_URL
        if not self.DATABASE_SYNC_URL.startswith("postgresql://"):
            self.DATABASE_SYNC_URL = self.DATABASE_SYNC_URL.replace("postgresql+asyncpg://", "postgresql://", 1)
            
        try:
            parsed_sync = urlparse(self.DATABASE_SYNC_URL)
            if parsed_sync.query:
                query_params = parse_qs(parsed_sync.query)
                query_params.pop("schema", None)
                new_query = urlencode(query_params, doseq=True)
                self.DATABASE_SYNC_URL = urlunparse(parsed_sync._replace(query=new_query))
        except Exception:
            pass

settings = Settings()
