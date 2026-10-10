"""
AeroDoc-AI - Ask a question through the agent graph: retrieve -> write -> verify (up to 2 drafts).

Same interface as ask.py. Every step is printed: the draft, the verifier's verdict and comment,
then the final answer and its sources.

Usage:
    python src/ask_agent.py "Combien d'avions Airbus a-t-il livrés en 2023 ?"
    python src/ask_agent.py "..." --k 5 --no-cache
    python src/ask_agent.py --mermaid docs/graph.md     # export the graph diagram (no LLM, no database)

Models: WRITER_MODEL and VERIFIER_MODEL in .env (both LLM_MODEL by default).
"""
import argparse
import logging
import re
import sys
from pathlib import Path

print("[agent] chargement des bibliothèques (10 à 30 s la première fois)...", flush=True)

from agent.graph import MAX_ATTEMPTS, build_graph, to_mermaid

GRAPH_DOC = """# Graphe de l'agent AeroDoc

Généré par `python src/ask_agent.py --mermaid docs/graph.md` à partir du graphe compilé (`src/agent/graph.py`).

```mermaid
{mermaid}
```

- **retrieve** : recherche des passages (bge-m3 + pgvector, la même recherche qu'`ask.py`).
- **write** : le rédacteur écrit la réponse avec citations `[n]`, ou « Je ne trouve pas cette information dans les documents. » sans source. Au 2e essai, il reçoit le commentaire du vérificateur et son brouillon précédent.
- **verify** : le vérificateur renvoie `{{"verdict": "ok" | "a_corriger", "probleme": "..."}}`. Un JSON illisible est considéré comme `ok` et signalé dans les logs.
- Flèche en pointillés vers **write** : verdict `a_corriger` et moins de {max_attempts} essais. Sinon, fin : la réponse finale est le dernier brouillon.
"""


def print_sources(passages, draft):
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", draft)}
    for i, p in enumerate(passages, 1):
        mark = "*" if i in cited else " "
        print(f" {mark}[{i}] {p['file']} p.{p['page']}  (score {p['score']:.3f})")
    print("(* = source citée dans la réponse)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Question -> agent graph (writer + verifier)")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--k", type=int, default=5, help="number of passages given to the LLM")
    parser.add_argument("--no-cache", action="store_true", help="always call the API, ignore the answer cache")
    parser.add_argument("--mermaid", type=Path, metavar="PATH", help="write the graph diagram and stop")
    args = parser.parse_args()

    if args.mermaid:
        args.mermaid.parent.mkdir(parents=True, exist_ok=True)
        args.mermaid.write_text(GRAPH_DOC.format(mermaid=to_mermaid().strip(), max_attempts=MAX_ATTEMPTS),
                                encoding="utf-8")
        print(f"[agent] schéma écrit dans {args.mermaid}")
        return
    if not args.question:
        parser.error("a question is required (or use --mermaid)")

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")
    from llama_index.core import VectorStoreIndex

    from config import VERIFIER_MODEL, WRITER_MODEL, get_vector_store, setup_settings
    from llm_client import LLMClient, error_hint, make_client

    setup_settings()
    retriever = VectorStoreIndex.from_vector_store(get_vector_store()).as_retriever(similarity_top_k=args.k)
    try:
        writer = make_client(WRITER_MODEL, use_cache=not args.no_cache)
        verifier = (LLMClient(writer.llm, VERIFIER_MODEL, writer.pause_s, writer.cache)  # same model: one client
                    if VERIFIER_MODEL == WRITER_MODEL else make_client(VERIFIER_MODEL, use_cache=not args.no_cache))
    except RuntimeError as exc:  # missing key
        sys.exit(f"[error] {exc}")
    except Exception as exc:
        sys.exit(f"[error] Gemini setup failed: {type(exc).__name__}: {exc}{error_hint(exc)}")

    print(f"[agent] rédacteur : {WRITER_MODEL} | vérificateur : {VERIFIER_MODEL} | {MAX_ATTEMPTS} essais max\n")
    graph = build_graph(retriever, writer, verifier)
    final = {}
    try:
        for update in graph.stream({"question": args.question}, stream_mode="updates"):
            for node, change in update.items():
                final.update({k: v for k, v in change.items() if k != "history"})
                if node == "retrieve":
                    pages = ", ".join(f"[{i}] p.{p['page']}" for i, p in enumerate(change["passages"], 1))
                    print(f"[retrieve] {len(change['passages'])} passages : {pages}")
                elif node == "write":
                    print(f"\n[write] essai {change['attempts']} - brouillon :\n{change['draft']}")
                elif node == "verify":
                    print(f"[verify] verdict : {change['verdict']}")
                    print(f"[verify] commentaire : {change['feedback'] or '(aucun)'}")
    except Exception as exc:  # network, quota... (the client already retries 3 times)
        sys.exit(f"[error] Gemini call failed: {type(exc).__name__}: {exc}{error_hint(exc)}")

    print(f"\n=== Réponse finale (après {final['attempts']} essai(s), dernier verdict : {final['verdict']}) ===")
    if final["verdict"] != "ok":
        print(f"[!] le vérificateur n'a pas validé la réponse après {MAX_ATTEMPTS} essais : "
              "dernier brouillon affiché tel quel.")
    print(f"\nQ: {args.question}\n\n{final['draft']}\n\nSources :")
    print_sources(final["passages"], final["draft"])
    calls = writer.real_calls + (verifier.real_calls if verifier is not writer else 0)
    hits = writer.cache_hits + (verifier.cache_hits if verifier is not writer else 0)
    print(f"[agent] appels API : {calls} | réponses servies par le cache : {hits}")


if __name__ == "__main__":
    main()
