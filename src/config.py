"""AeroDoc-AI - shared configuration (embedding model, pgvector store, Gemini LLM)."""
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
EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "512"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# Answer generation (Google Gemini API). The key is read from .env and never printed.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))  # 0 = most faithful to the sources
# Generous on purpose: Gemini "thinking" tokens count against this limit and could cut the answer short.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
# Minimum pause between two real LLM calls (flash-lite: 15 requests/min, 500/day).
LLM_PAUSE_S = float(os.getenv("LLM_PAUSE_S", "5"))

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
    """Local embeddings. No LLM here: ingestion, search and evaluation do not need one
    (the answer step builds its own with get_llm())."""
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


def get_llm(model: str | None = None):
    """Gemini chat model (LLM_MODEL by default). Needs GEMINI_API_KEY in .env."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env (see .env.example).")
    from llama_index.llms.google_genai import GoogleGenAI

    model = model or LLM_MODEL
    return GoogleGenAI(
        model=model,
        api_key=GEMINI_API_KEY,
        # gemini-3 models may reject an explicit temperature: leave it to the API default there.
        temperature=None if "gemini-3" in model else LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


def get_vector_store() -> PGVectorStore:
    """Exact (brute-force) cosine search: enough for a few thousand chunks."""
    return PGVectorStore.from_params(
        **PG,
        table_name=PG_TABLE,
        embed_dim=EMBED_DIM,
    )
