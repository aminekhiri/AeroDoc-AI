"""
AeroDoc-AI - Ask a question through the agent graph: retrieve -> write -> verify (up to 2 drafts).

Same interface as ask.py. Every step is printed: the draft, the verifier's verdict and comment,
then the final answer and its sources. For several questions in a row, use chat.py: the models
are loaded only once.

Usage:
    python src/ask_agent.py "Combien d'avions Airbus a-t-il livrés en 2023 ?"
    python src/ask_agent.py "..." --k 5 --no-cache
    python src/ask_agent.py --mermaid docs/graph.md     # export the graph diagram (no LLM, no database)

Models: WRITER_MODEL and VERIFIER_MODEL in .env (both LLM_MODEL by default).
"""
import argparse
import logging
import sys
from pathlib import Path

print("[agent] chargement des bibliothèques (10 à 30 s la première fois)...", flush=True)

from agent.graph import MAX_ATTEMPTS, to_mermaid
from agent.runtime import load_agent, print_sources
from config import HISTORY_TURNS

GRAPH_DOC = """# Graphe de l'agent AeroDoc

Généré par `python src/ask_agent.py --mermaid docs/graph.md` à partir du graphe compilé (`src/agent/graph.py`).

```mermaid
{mermaid}
```

- **route** : routeur à sortie structurée (schéma Pydantic `RouteDecision`) : `documents`, `conversation` ou `hors_perimetre`. En cas de doute ou d'échec : `documents`. Il voit les {history_turns} derniers échanges.
- **condense** : reformule une relance (« et en 2023 ? », « reviens à ma première question ») en question autonome, à partir des {history_turns} derniers échanges. Sans historique : aucun appel.
- **answer_from_history** : répond uniquement à partir de la conversation (« liste mes dernières questions »), sans recherche ni citation de PDF.
- **refuse** : hors périmètre, refus direct sans recherche ni appel LLM.
- **finish** : ajoute la réponse à la conversation (`messages`, conservés par le checkpointer `MemorySaver` par `thread_id` dans le chat).
- **retrieve** : recherche des passages (bge-m3 + pgvector, la même recherche qu'`ask.py`).
- **write** : le rédacteur écrit la réponse avec citations `[n]`, ou « Je ne trouve pas cette information dans les documents. » sans source. Au 2e essai, il reçoit le commentaire du vérificateur et son brouillon précédent.
- **verify** : le vérificateur renvoie `{{"verdict": "ok" | "a_corriger", "probleme": "..."}}`. Un JSON illisible est considéré comme `ok` et signalé dans les logs.
- Après **verify** : retour à **write** si verdict `a_corriger` et moins de {max_attempts} essais ; sinon **finish** : la réponse finale est le dernier brouillon.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Question -> agent graph (writer + verifier)")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--k", type=int, default=5, help="number of passages given to the LLM")
    parser.add_argument("--no-cache", action="store_true", help="always call the API, ignore the answer cache")
    parser.add_argument("--mermaid", type=Path, metavar="PATH", help="write the graph diagram and stop")
    args = parser.parse_args()

    if args.mermaid:
        args.mermaid.parent.mkdir(parents=True, exist_ok=True)
        args.mermaid.write_text(GRAPH_DOC.format(mermaid=to_mermaid().strip(), max_attempts=MAX_ATTEMPTS,
                                                 history_turns=HISTORY_TURNS), encoding="utf-8")
        print(f"[agent] schéma écrit dans {args.mermaid}")
        return
    if not args.question:
        parser.error("a question is required (or use --mermaid)")

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")
    from config import VERIFIER_MODEL, WRITER_MODEL
    from llm_client import error_hint

    try:
        agent = load_agent(k=args.k, use_cache=not args.no_cache)
    except RuntimeError as exc:  # missing key
        sys.exit(f"[error] {exc}")
    except Exception as exc:
        sys.exit(f"[error] Gemini setup failed: {type(exc).__name__}: {exc}{error_hint(exc)}")

    print(f"[agent] rédacteur : {WRITER_MODEL} | vérificateur : {VERIFIER_MODEL} | {MAX_ATTEMPTS} essais max\n")
    try:
        final = agent.run(args.question)
    except Exception as exc:  # network, quota... (the client already retries 3 times)
        sys.exit(f"[error] Gemini call failed: {type(exc).__name__}: {exc}{error_hint(exc)}")

    if final.get("route") != "documents":
        print(f"\n=== Réponse finale (route {final.get('route')}, sans recherche dans les PDF) ===")
    else:
        print(f"\n=== Réponse finale (après {final['attempts']} essai(s), dernier verdict : {final['verdict']}) ===")
    if final.get("route") == "documents" and final["verdict"] != "ok":
        print(f"[!] le vérificateur n'a pas validé la réponse après {MAX_ATTEMPTS} essais : "
              "dernier brouillon affiché tel quel.")
    print(f"\nQ: {args.question}\n\n{final['draft']}\n\nSources :")
    print_sources(final["passages"], final["draft"])
    calls, hits = agent.api_stats()
    print(f"[agent] appels API : {calls} | réponses servies par le cache : {hits}")


if __name__ == "__main__":
    main()
