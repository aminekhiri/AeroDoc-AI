"""AeroDoc-AI - shared configuration (embedding model + pgvector store)."""
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data/raw"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

PG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "user": os.getenv("PG_USER", "aerodoc"),
    "password": os.getenv("PG_PASSWORD", "aerodoc"),
    "database": os.getenv("PG_DB", "aerodoc"),
}
PG_TABLE = os.getenv("PG_TABLE", "chunks")


def setup_settings() -> None:
    """Local embeddings only, no LLM needed at this stage."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        normalize=True,  # unit vectors -> cosine similarity = dot product
    )
    Settings.llm = None


def get_vector_store() -> PGVectorStore:
    """Exact (brute-force) cosine search: enough for a few thousand chunks."""
    return PGVectorStore.from_params(
        **PG,
        table_name=PG_TABLE,
        embed_dim=EMBED_DIM,
    )
