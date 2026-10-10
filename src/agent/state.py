"""Shared state of the agent graph.

With a checkpointer (chat), the state is kept between the turns of a conversation (thread_id):
`messages` accumulates the whole conversation (add_messages); the other fields describe the
current turn and are reset by turn_input() at the start of each question.
"""
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class Passage(TypedDict):
    file: str
    page: str
    text: str
    score: float


class Step(TypedDict):
    """One writing attempt and what the verifier said about it (kept for display and logs)."""
    attempt: int
    draft: str
    verdict: str
    feedback: str


def steps_of_turn(left: list | None, right: list | None) -> list:
    """Steps are appended during a turn; None (sent by turn_input) resets them for a new turn."""
    if right is None:
        return []
    return (left or []) + right


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # conversation: user questions and final answers
    user_question: str           # question as typed by the user
    route: str                   # "documents" | "conversation" | "hors_perimetre"
    route_reason: str            # short reason given by the router
    question: str                # standalone question used for the search (after rewriting)
    passages: list[Passage]      # retrieved sources, numbered [1], [2]... in the prompts
    draft: str                   # brouillon : latest answer written
    verdict: str                 # "ok" | "a_corriger" ("" when there is no verification)
    feedback: str                # commentaire du vérificateur on the latest draft
    attempts: int                # number of writing attempts in this turn
    steps: Annotated[list[Step], steps_of_turn]  # writing attempts of this turn

