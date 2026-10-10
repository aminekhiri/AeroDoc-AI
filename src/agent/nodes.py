"""Nodes of the agent graph. Dependencies are injected, so the tests can use fakes without network.

- retriever: object with .retrieve(question) -> LlamaIndex results (the same retriever as ask.py);
- writer, verifier: objects with .ask(system, user) -> str (an LLMClient, or a fake in the tests).
"""
import json
import logging
import re

from prompts import (SYSTEM_PROMPT, VERIFIER_SYSTEM_PROMPT, build_prompt, revision_prompt,
                     to_passages, verification_prompt)

log = logging.getLogger("aerodoc.agent")
VERDICTS = ("ok", "a_corriger")


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
        return {"verdict": verdict, "feedback": feedback, "history": [step]}
    return verify
