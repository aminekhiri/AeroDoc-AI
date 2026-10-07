"""
AeroDoc-AI - Retrieval test (no LLM): shows the top-k chunks for a question.

Usage:
    python src/search.py "What are the requirements for emergency exits?" --k 5
"""
import argparse
import textwrap

from llama_index.core import VectorStoreIndex

from config import get_vector_store, setup_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    setup_settings()
    index = VectorStoreIndex.from_vector_store(get_vector_store())
    retriever = index.as_retriever(similarity_top_k=args.k)
    results = retriever.retrieve(args.question)

    print(f"\nQ: {args.question}\n")
    for i, r in enumerate(results, 1):
        meta = r.node.metadata
        src = f"{meta.get('file_name', '?')} p.{meta.get('page_label', '?')}"
        snippet = textwrap.shorten(r.node.get_content().replace("\n", " "), width=300)
        print(f"[{i}] score={r.score:.3f} | {src}\n    {snippet}\n")


if __name__ == "__main__":
    main()
