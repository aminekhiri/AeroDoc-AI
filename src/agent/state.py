"""Shared state of the agent graph."""
from operator import add
from typing import Annotated, TypedDict


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


class AgentState(TypedDict, total=False):
    question: str
    passages: list[Passage]      # retrieved sources, numbered [1], [2]... in the prompts
    draft: str                   # brouillon : latest answer written
    verdict: str                 # "ok" | "a_corriger"
    feedback: str                # commentaire du vérificateur on the latest draft
    attempts: int                # number of writing attempts so far
    history: Annotated[list[Step], add]  # appended by each verification
