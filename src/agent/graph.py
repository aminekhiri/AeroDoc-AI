"""Agent graph: retrieve -> write -> verify; back to write with the verifier's comment while the
verdict is "a_corriger" and fewer than MAX_ATTEMPTS drafts have been written; otherwise end.
"""
from langgraph.graph import END, START, StateGraph

from agent.nodes import make_retrieve, make_verify, make_write
from agent.state import AgentState

MAX_ATTEMPTS = 2


def route_after_verify(state: AgentState) -> str:
    if state.get("verdict") == "a_corriger" and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "write"
    return END


def build_graph(retriever, writer, verifier):
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", make_retrieve(retriever))
    graph.add_node("write", make_write(writer))
    graph.add_node("verify", make_verify(verifier))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "write")
    graph.add_edge("write", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"write": "write", END: END})
    return graph.compile()


def to_mermaid() -> str:
    """Mermaid diagram of the graph (no retriever or LLM needed: the nodes are never called)."""
    return build_graph(None, None, None).get_graph().draw_mermaid()
