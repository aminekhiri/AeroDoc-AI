"""
AeroDoc-AI - Chat in the terminal. The embedding model, the database connection and the Gemini
clients are loaded ONCE at start-up (~20 s); then each question only costs the LLM calls.

The conversation is kept by a LangGraph checkpointer (MemorySaver, in memory: it is lost when the
chat ends), one thread_id per conversation. At each turn a router chooses:
  - documents      : rewriting of the question from the history, then search in the PDF + answer + verification;
  - conversation   : answer from the conversation history only (no search, no PDF citation);
  - hors_perimetre : direct refusal, no search.
The router and the rewriter see the last HISTORY_TURNS exchanges (.env, 10 by default).

Usage:
    python src/chat.py
    python src/chat.py --details      # also show the drafts and the verifier's comments

Commands during the chat:
    /reset    start a new conversation         /details  show or hide the intermediate steps
    /help     list the commands                /quit     leave (Ctrl+C also works)
"""
import argparse
import logging
import sys
import time
import uuid

print("[chat] chargement des modèles, une seule fois (10 à 30 s)...", flush=True)

from langgraph.checkpoint.memory import MemorySaver

from agent.graph import MAX_ATTEMPTS
from agent.runtime import load_agent, print_sources
from config import HISTORY_TURNS, VERIFIER_MODEL, WRITER_MODEL
from llm_client import error_hint

HELP = ("Commandes : /reset (nouvelle conversation), /details (afficher ou masquer les étapes), "
        "/help, /quit")


def new_thread() -> str:
    return f"chat-{uuid.uuid4().hex[:8]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="AeroDoc chat (router + agent + conversation memory)")
    parser.add_argument("--k", type=int, default=5, help="number of passages given to the LLM")
    parser.add_argument("--details", action="store_true", help="show drafts and verifier comments")
    parser.add_argument("--no-cache", action="store_true", help="always call the API, ignore the answer cache")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")

    try:
        agent = load_agent(k=args.k, use_cache=not args.no_cache, checkpointer=MemorySaver(),
                           history_turns=HISTORY_TURNS)
    except RuntimeError as exc:  # missing key
        sys.exit(f"[error] {exc}")
    except Exception as exc:
        sys.exit(f"[error] Gemini setup failed: {type(exc).__name__}: {exc}{error_hint(exc)}")

    thread_id, details = new_thread(), args.details
    print(f"\nAeroDoc prêt. Rédacteur : {WRITER_MODEL} | vérificateur : {VERIFIER_MODEL} | "
          f"mémoire : {HISTORY_TURNS} derniers échanges | conversation {thread_id}\n{HELP}\n")

    while True:
        try:
            question = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        command = question.lower()
        if command in ("/quit", "/exit", "/q"):
            break
        if command == "/help":
            print(HELP)
            continue
        if command == "/reset":
            thread_id = new_thread()
            print(f"[chat] nouvelle conversation ({thread_id}) : l'historique précédent est oublié.\n")
            continue
        if command == "/details":
            details = not details
            print(f"[chat] étapes intermédiaires {'affichées' if details else 'masquées'}.\n")
            continue

        start, calls_before = time.monotonic(), agent.api_stats()[0]
        try:
            final = agent.run(question, thread_id=thread_id, show_steps=details)
        except Exception as exc:  # network, quota... the chat goes on
            print(f"[error] Gemini call failed: {type(exc).__name__}: {exc}{error_hint(exc)}\n")
            continue

        route = final.get("route", "?")
        print(f"[route] {route} ({final.get('route_reason') or 'sans raison'})")
        if route == "documents" and final.get("question") != question:
            print(f"[mémoire] question comprise comme : {final['question']}")
        answer = final["draft"]
        print(f"\nAeroDoc > {answer}\n")
        if route == "documents":
            if final["verdict"] != "ok":
                print(f"[!] le vérificateur n'a pas validé cette réponse après {MAX_ATTEMPTS} essais.")
            print_sources(final["passages"], answer)
            detail = f"{final['attempts']} essai(s), verdict {final['verdict']}"
        else:
            detail = "sans recherche dans les PDF"
        calls = agent.api_stats()[0] - calls_before
        print(f"[{time.monotonic() - start:.1f} s | route {route} | {detail} | {calls} appel(s) API]\n")

    calls, hits = agent.api_stats()
    print(f"[chat] fin. Appels API : {calls} | réponses servies par le cache : {hits}")


if __name__ == "__main__":
    main()
