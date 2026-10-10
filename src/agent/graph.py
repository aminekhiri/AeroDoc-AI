"""Agent graph.

    route --documents--------> condense -> retrieve -> write -> verify --ok / 2 essais--> finish
          |                                             ^          |
          |                                             +--a_corriger (essais < 2)
          +--conversation-----> answer_from_history -------------------------------------> finish
          +--hors_perimetre---> refuse ------------------------------------------------> finish

finish stores the answer in `messages`. With a checkpointer, the conversation is kept per thread_id.
"""
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agent.nodes import (finish, make_answer_from_history, make_condense, make_retrieve, make_route,
                         make_verify, make_write, refuse)
from agent.state import AgentState

MAX_ATTEMPTS = 2
DEFAULT_HISTORY_TURNS = 10
ROUTES = {"documents": "condense", "conversation": "answer_from_history", "hors_perimetre": "refuse"}


def route_after_router(state: AgentState) -> str:
    return ROUTES.get(state.get("route"), "condense")  # unknown route: documents


def route_after_verify(state: AgentState) -> str:
    if state.get("verdict") == "a_corriger" and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "write"
    return "finish"


def turn_input(question: str) -> dict:
    """Input of one turn: adds the question to the conversation and resets the per-turn fields."""
    return {"messages": [HumanMessage(content=question)], "user_question": question, "question": question,
            "route": "", "route_reason": "", "passages": [], "draft": "", "verdict": "", "feedback": "",
            "attempts": 0, "steps": None}


def build_graph(retriever, writer, verifier, router=None, history_turns: int = DEFAULT_HISTORY_TURNS,
                checkpointer=None):
    """router (routing, rewriting, answers from the history) defaults to the writer's LLM."""
    router = router or writer
    graph = StateGraph(AgentState)
    graph.add_node("route", make_route(router, history_turns))
    graph.add_node("condense", make_condense(router, history_turns))
    graph.add_node("retrieve", make_retrieve(retriever))
    graph.add_node("write", make_write(writer))
    graph.add_node("verify", make_verify(verifier))
    graph.add_node("answer_from_history", make_answer_from_history(router, history_turns))
    graph.add_node("refuse", refuse)
    graph.add_node("finish", finish)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", route_after_router, list(ROUTES.values()))
    graph.add_edge("condense", "retrieve")
    graph.add_edge("retrieve", "write")
    graph.add_edge("write", "verify")
    graph.add_conditional_edges("verify", route_after_verify, ["write", "finish"])
    graph.add_edge("answer_from_history", "finish")
    graph.add_edge("refuse", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


def to_mermaid() -> str:
    """Mermaid diagram of the graph (no retriever or LLM needed: the nodes are never called)."""
    return build_graph(None, None, None).get_graph().draw_mermaid()
