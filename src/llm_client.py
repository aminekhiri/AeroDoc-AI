"""AeroDoc-AI - LLM calls shared by every script that calls the LLM.

LLMClient.ask(system, user) -> text. It waits at least `pause_s` seconds between two real
calls, to stay under the API rate limit (gemini-3.5-flash-lite: 15 requests/min, 500/day).
The pause is shared by all clients of the process (writer and verifier use the same quota).
"""
import time

_last_call = 0.0  # time.monotonic() of the last real call in this process


class LLMClient:
    def __init__(self, llm, model: str, pause_s: float = 0.0):
        self.llm = llm          # any LlamaIndex chat LLM (or a test double with .chat)
        self.model = model
        self.pause_s = pause_s
        self.real_calls = 0     # calls actually sent to the API

    def _wait_turn(self) -> None:
        wait = _last_call + self.pause_s - time.monotonic()
        if _last_call and wait > 0:
            time.sleep(wait)

    def ask(self, system: str, user: str) -> str:
        global _last_call
        from llama_index.core.llms import ChatMessage, MessageRole

        self._wait_turn()
        try:
            response = self.llm.chat([
                ChatMessage(role=MessageRole.SYSTEM, content=system),
                ChatMessage(role=MessageRole.USER, content=user),
            ])
        finally:  # a failed call also counts against the quota
            _last_call = time.monotonic()
            self.real_calls += 1
        return (response.message.content or "").strip()
