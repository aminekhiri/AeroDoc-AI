"""Load the agent once (embedding model, pgvector, Gemini clients) and run questions through it.

Used by ask_agent.py (one question, no memory) and chat.py (many questions, models loaded once,
conversation kept by a LangGraph checkpointer per thread_id).
"""
import re
from dataclasses import dataclass

from agent.graph import DEFAULT_HISTORY_TURNS, build_graph, turn_input


@dataclass
class Agent:
    graph: object
    writer: object
    verifier: object

    def run(self, question: str, thread_id: str | None = None, show_steps: bool = True) -> dict:
        """Run one turn; print each step if asked; return the final state of the turn."""
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        final = {}
        for update in self.graph.stream(turn_input(question), config=config, stream_mode="updates"):
            for node, change in update.items():
                final.update({k: v for k, v in (change or {}).items() if k not in ("messages", "steps")})
                if show_steps:
                    _print_step(node, change or {}, final)
        return final

    def api_stats(self) -> tuple[int, int]:
        """(calls sent to the API, answers served by the cache) since the agent was loaded."""
        clients = [self.writer] if self.verifier is self.writer else [self.writer, self.verifier]
        return sum(c.real_calls for c in clients), sum(c.cache_hits for c in clients)


def _print_step(node: str, change: dict, state: dict) -> None:
    if node == "route":
        print(f"[route] {change['route']} ({change.get('route_reason') or 'sans raison'})")
    elif node == "condense" and change["question"] != state.get("user_question", change["question"]):
        print(f"[mémoire] question reformulée : {change['question']}")
    elif node == "retrieve":
        pages = ", ".join(f"[{i}] p.{p['page']}" for i, p in enumerate(change["passages"], 1))
        print(f"[retrieve] {len(change['passages'])} passages : {pages}")
    elif node == "write":
        print(f"\n[write] essai {change['attempts']} - brouillon :\n{change['draft']}")
    elif node == "verify":
        print(f"[verify] verdict : {change['verdict']}")
        print(f"[verify] commentaire : {change['feedback'] or '(aucun)'}")
    elif node == "answer_from_history":
        print("[conversation] réponse tirée de l'historique, sans recherche dans les PDF")
    elif node == "refuse":
        print("[hors périmètre] refus direct, sans recherche")


def load_agent(k: int = 5, use_cache: bool = True, checkpointer=None,
               history_turns: int = DEFAULT_HISTORY_TURNS) -> Agent:
    """Load the embedding model, connect to pgvector and create the Gemini clients (slow: ~20 s)."""
    from llama_index.core import VectorStoreIndex

    from config import VERIFIER_MODEL, WRITER_MODEL, get_vector_store, setup_settings
    from llm_client import make_client

    setup_settings()
    retriever = VectorStoreIndex.from_vector_store(get_vector_store()).as_retriever(similarity_top_k=k)
    writer = make_client(WRITER_MODEL, use_cache=use_cache)
    verifier = (writer if VERIFIER_MODEL == WRITER_MODEL  # same model: one client, one quota counter
                else make_client(VERIFIER_MODEL, use_cache=use_cache))
    graph = build_graph(retriever, writer, verifier, history_turns=history_turns, checkpointer=checkpointer)
    return Agent(graph, writer, verifier)


def print_sources(passages: list[dict], answer: str) -> None:
    if not passages:
        return
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    for i, p in enumerate(passages, 1):
        mark = "*" if i in cited else " "
        print(f" {mark}[{i}] {p['file']} p.{p['page']}  (score {p['score']:.3f})")
    print("(* = source citée dans la réponse)")
