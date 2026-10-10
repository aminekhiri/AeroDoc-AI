"""
AeroDoc-AI - Ask a question: retrieval (pgvector) + answer written by Gemini, with citations.

Usage:
    python src/ask.py "What are the requirements for emergency exits?"
    python src/ask.py "Combien d'avions Airbus a-t-il livrés en 2023 ?" --k 5
    python src/ask.py "..." --show-prompt     # print the prompt, do not call the LLM (no API key needed)
    python src/ask.py "..." --no-cache        # always call the API, ignore the answer cache

Needs GEMINI_API_KEY in .env (see .env.example).
"""
import argparse
import re
import sys

print("[ask] chargement des bibliothèques (10 à 30 s la première fois)...", flush=True)

from llama_index.core import VectorStoreIndex

from config import LLM_MODEL, get_vector_store, setup_settings
from llm_client import error_hint, make_client
from prompts import SYSTEM_PROMPT, build_prompt, to_passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5, help="number of passages given to the LLM")
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt and stop")
    parser.add_argument("--no-cache", action="store_true", help="always call the API, ignore the answer cache")
    args = parser.parse_args()

    setup_settings()
    index = VectorStoreIndex.from_vector_store(get_vector_store())
    nodes = index.as_retriever(similarity_top_k=args.k).retrieve(args.question)
    prompt = build_prompt(args.question, to_passages(nodes))

    if args.show_prompt:
        print(f"\n--- system ---\n{SYSTEM_PROMPT}\n\n--- user ---\n{prompt}")
        return

    print(f"[ask] {len(nodes)} passages trouvés, appel à Gemini ({LLM_MODEL})...", flush=True)
    try:
        client = make_client(use_cache=not args.no_cache)
        answer = client.ask(SYSTEM_PROMPT, prompt)
    except RuntimeError as exc:  # missing key
        sys.exit(f"[error] {exc}")
    except Exception as exc:  # network, quota, invalid key... (the client already retries 3 times)
        sys.exit(f"[error] Gemini call failed ({LLM_MODEL}): {type(exc).__name__}: {exc}{error_hint(exc)}")

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    origin = f"{LLM_MODEL}, depuis le cache" if client.cache_hits else LLM_MODEL

    print(f"\nQ: {args.question}\n\n{answer}\n\nSources ({origin}) :")
    for i, r in enumerate(nodes, 1):
        meta = r.node.metadata
        mark = "*" if i in cited else " "
        print(f" {mark}[{i}] {meta.get('file_name', '?')} p.{meta.get('page_label', '?')}  (score {r.score:.3f})")
    print("(* = source citée dans la réponse)")


if __name__ == "__main__":
    main()
