"""AeroDoc-AI - shared configuration (embedding model + pgvector store)."""
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data/raw"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "512"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

def _required(name: str) -> str:
    """Read a mandatory variable from the environment (.env), with no default."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


PG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "user": _required("PG_USER"),
    "password": _required("PG_PASSWORD"),
    "database": _required("PG_DB"),
}
PG_TABLE = os.getenv("PG_TABLE", "chunks")


def get_device() -> str:
    """Best available torch device: cuda, then mps (Apple), then cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_settings() -> None:
    """Local embeddings only, no LLM needed at this stage."""
    device = get_device()
    is_e5 = "e5" in EMBED_MODEL.lower()
    print(f"[embed] model={EMBED_MODEL} dim={EMBED_DIM} device={device} "
          f"batch={EMBED_BATCH_SIZE} max_length={EMBED_MAX_LENGTH}")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        max_length=EMBED_MAX_LENGTH,  # chunks are ~512 tokens: no need for the model's full context
        embed_batch_size=EMBED_BATCH_SIZE,
        device=device,
        normalize=True,  # unit vectors -> cosine similarity = dot product
        # E5 models expect these prefixes on queries and passages
        query_instruction="query: " if is_e5 else None,
        text_instruction="passage: " if is_e5 else None,
    )
    Settings.llm = None


def get_vector_store() -> PGVectorStore:
    """Exact (brute-force) cosine search: enough for a few thousand chunks."""
    return PGVectorStore.from_params(
        **PG,
        table_name=PG_TABLE,
        embed_dim=EMBED_DIM,
    )
