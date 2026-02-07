from chromadb.config import Settings

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duck_db+parquet",
    persist_directory="db",
    anonymized_telemetry=False
)
