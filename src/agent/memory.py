"""Conversation memory: the history lives in the graph state (`messages`, kept by the LangGraph
checkpointer per thread_id). These helpers turn it into (question, answer) exchanges and rewrite a
follow-up question ("Et en 2022 ?") into a standalone one before the search.
"""
import logging

from prompts import CONDENSE_SYSTEM_PROMPT, history_prompt

log = logging.getLogger("aerodoc.chat")
MAX_STANDALONE_CHARS = 500


def turns_from_messages(messages: list, max_turns: int) -> list[tuple[str, str]]:
    """Last `max_turns` (user question, answer) exchanges, oldest first. A question without answer
    (a turn that failed) is skipped."""
    turns, pending = [], None
    for message in messages:
        if message.type == "human":
            pending = message.content
        elif message.type == "ai" and pending is not None:
            turns.append((pending, message.content))
            pending = None
    return turns[-max_turns:] if max_turns > 0 else []


def standalone_question(llm, turns: list[tuple[str, str]], question: str) -> str:
    """Rewrite a follow-up question from the conversation; unchanged (and free) without history."""
    if not turns:
        return question
    rewritten = llm.ask(CONDENSE_SYSTEM_PROMPT, history_prompt(turns, question))
    rewritten = rewritten.strip().splitlines()[0].strip().strip('"«»“” ') if rewritten.strip() else ""
    if not rewritten or len(rewritten) > MAX_STANDALONE_CHARS:
        log.warning("unusable rewritten question, original kept: %r", rewritten[:200])
        return question
    return rewritten
