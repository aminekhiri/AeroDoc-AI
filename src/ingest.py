"""
AeroDoc-AI - Ingestion pipeline
PDF -> pages -> chunks -> local embeddings (bge-m3) -> pgvector

Usage:
    python src/ingest.py            # add documents
    python src/ingest.py --reset    # drop the table and re-index everything
"""
import argparse
import sys
import time

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import create_engine, text

from config import (
    CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBED_MODEL, PG, PG_TABLE,
    get_vector_store, setup_settings,
)


def reset_table() -> None:
    url = (f"postgresql+psycopg2://{PG['user']}:{PG['password']}"
           f"@{PG['host']}:{PG['port']}/{PG['database']}")
    engine = create_engine(url)
    with engine.begin() as conn:
        # LlamaIndex prefixes table names with "data_"
        conn.execute(text(f'DROP TABLE IF EXISTS "data_{PG_TABLE}"'))
    print(f"[reset] table data_{PG_TABLE} dropped")


def load_documents():
    if not DATA_DIR.exists() or not any(DATA_DIR.rglob("*.pdf")):
        sys.exit(f"[error] no PDF found in {DATA_DIR.resolve()}")
    reader = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".pdf"],
        recursive=True,
        filename_as_id=True,
    )
    docs = reader.load_data()  # one Document per page, metadata: file_name, page_label
    for d in docs:
        d.excluded_embed_metadata_keys = [
            k for k in d.metadata if k not in ("file_name", "page_label")
        ]
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="drop table before indexing")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[config] model={EMBED_MODEL} chunk={CHUNK_SIZE} overlap={CHUNK_OVERLAP}")
    setup_settings()

    if args.reset:
        reset_table()

    docs = load_documents()
    n_files = len({d.metadata.get("file_name") for d in docs})
    print(f"[load] {n_files} PDF(s), {len(docs)} page(s)")

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs, show_progress=True)
    nodes = [n for n in nodes if len(n.get_content().strip()) > 50]  # drop empty/noise chunks
    print(f"[chunk] {len(nodes)} chunk(s)")

    storage_context = StorageContext.from_defaults(vector_store=get_vector_store())
    VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

    print(f"[done] indexed {len(nodes)} chunks in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
