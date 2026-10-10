"""Nodes of the agent graph. Dependencies are injected, so the tests can use fakes without network.

- retriever: object with .retrieve(question) -> LlamaIndex results (the same retriever as ask.py);
- writer, verifier, router: objects with .ask(system, user) -> str and, for the router,
  .ask_structured(system, user, schema) -> Pydantic instance (an LLMClient, or a fake in the tests).
"""
import json
import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agent.memory import standalone_question, turns_from_messages
from prompts import (CONVERSATION_SYSTEM_PROMPT, REFUSAL_TEXT, ROUTER_SYSTEM_PROMPT, SYSTEM_PROMPT,
                     VERIFIER_SYSTEM_PROMPT, build_prompt, history_prompt, revision_prompt,
                     to_passages, verification_prompt)

log = logging.getLogger("aerodoc.agent")
VERDICTS = ("ok", "a_corriger")


class RouteDecision(BaseModel):
    """Structured output of the router."""
    route: Literal["documents", "conversation", "hors_perimetre"] = Field(
        description="documents : contenu des PDF ; conversation : la conversation elle-même ; "
                    "hors_perimetre : sans rapport. En cas de doute : documents.")
    raison: str = Field(default="", description="raison très courte du choix")


def _history(state, max_turns: int) -> list[tuple[str, str]]:
    """Previous exchanges of the conversation (the last message is the current question)."""
    return turns_from_messages(state.get("messages", [])[:-1], max_turns)


def make_route(router, max_turns: int):
    def route(state):
        question = state["user_question"]
        try:
            decision = router.ask_structured(ROUTER_SYSTEM_PROMPT,
                                             history_prompt(_history(state, max_turns), question), RouteDecision)
            return {"route": decision.route, "route_reason": decision.raison}
        except Exception as exc:  # invalid structured output, API error...: when in doubt, documents
            log.warning("router failed (%s: %s), route forced to 'documents'", type(exc).__name__, exc)
            return {"route": "documents", "route_reason": "(routeur en échec : documents par défaut)"}
    return route


def make_condense(router, max_turns: int):
    def condense(state):
        question = state["user_question"]
        return {"question": standalone_question(router, _history(state, max_turns), question)}
    return condense


def make_answer_from_history(router, max_turns: int):
    def answer_from_history(state):
        prompt = history_prompt(_history(state, max_turns), state["user_question"])
        return {"draft": router.ask(CONVERSATION_SYSTEM_PROMPT, prompt), "passages": [], "attempts": 1}
    return answer_from_history


def refuse(state):
    """Out of scope: direct refusal, no search and no LLM call."""
    return {"draft": REFUSAL_TEXT, "passages": [], "attempts": 0}


def finish(state):
    """Store the final answer in the conversation."""
    return {"messages": [AIMessage(content=state["draft"])]}


def make_retrieve(retriever):
    def retrieve(state):
        return {"passages": to_passages(retriever.retrieve(state["question"]))}
    return retrieve


def make_write(writer):
    def write(state):
        question, passages = state["question"], state["passages"]
        if state.get("verdict") == "a_corriger" and state.get("feedback"):
            user = revision_prompt(question, passages, state["draft"], state["feedback"])
        else:
            user = build_prompt(question, passages)
        return {"draft": writer.ask(SYSTEM_PROMPT, user), "attempts": state.get("attempts", 0) + 1}
    return write


def parse_verdict(raw: str) -> tuple[str, str] | None:
    """Read {"verdict": ..., "probleme": ...} from the verifier output; None if it is not valid."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    verdict = str(data.get("verdict", "")).strip().lower() if isinstance(data, dict) else ""
    if verdict not in VERDICTS:
        return None
    return verdict, str(data.get("probleme", "") or "").strip()


def make_verify(verifier):
    def verify(state):
        raw = verifier.ask(VERIFIER_SYSTEM_PROMPT,
                           verification_prompt(state["question"], state["passages"], state["draft"]))
        parsed = parse_verdict(raw)
        if parsed is None:
            # Spec: an unreadable verdict must not block the answer; it is accepted and logged.
            log.warning("verifier returned invalid JSON, verdict forced to 'ok': %r", raw[:300])
            verdict, feedback = "ok", "(réponse du vérificateur illisible : verdict considéré comme ok)"
        else:
            verdict, feedback = parsed
        step = {"attempt": state["attempts"], "draft": state["draft"], "verdict": verdict, "feedback": feedback}
        return {"verdict": verdict, "feedback": feedback, "steps": [step]}
    return verify
