"""AeroDoc-AI - LLM calls shared by every script that calls the LLM.

LLMClient.ask(system, user) -> text, with:
  - a pause of at least `pause_s` seconds between two real calls, to stay under the API rate limit
    (gemini-3.5-flash-lite: 15 requests/min, 500/day). The pause is shared by all clients of the
    process (writer and verifier use the same quota);
  - an optional disk cache of the answers (data/cache/llm/ by default): the same model, system
    prompt and user prompt give the cached answer, without any API call. Any change of model or
    prompt is a new key, so a modified prompt is never answered from an old cache entry.
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

_last_call = 0.0  # time.monotonic() of the last real call in this process


class ResponseCache:
    """One JSON file per answer, named by the hash of (model, system, user)."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @staticmethod
    def key(model: str, system: str, user: str) -> str:
        payload = json.dumps([model, system, user], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["answer"]

    def put(self, key: str, model: str, answer: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {"model": model, "created": datetime.now(timezone.utc).isoformat(), "answer": answer}
        (self.directory / f"{key}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


class LLMClient:
    def __init__(self, llm, model: str, pause_s: float = 0.0, cache: ResponseCache | None = None):
        self.llm = llm          # any LlamaIndex chat LLM (or a test double with .chat)
        self.model = model
        self.pause_s = pause_s
        self.cache = cache
        self.real_calls = 0     # calls actually sent to the API
        self.cache_hits = 0     # answers served by the cache

    def _wait_turn(self) -> None:
        wait = _last_call + self.pause_s - time.monotonic()
        if _last_call and wait > 0:
            time.sleep(wait)

    def ask(self, system: str, user: str) -> str:
        global _last_call
        key = ResponseCache.key(self.model, system, user) if self.cache else None
        if key:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached

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
        answer = (response.message.content or "").strip()
        if key:
            self.cache.put(key, self.model, answer)
        return answer


def make_client(model: str | None = None, use_cache: bool = True) -> LLMClient:
    """Client for the real Gemini API, configured from .env (model, pause, cache)."""
    from config import LLM_CACHE, LLM_CACHE_DIR, LLM_MODEL, LLM_PAUSE_S, get_llm

    model = model or LLM_MODEL
    cache = ResponseCache(LLM_CACHE_DIR) if (use_cache and LLM_CACHE) else None
    return LLMClient(get_llm(model), model, pause_s=LLM_PAUSE_S, cache=cache)


def error_hint(exc: Exception) -> str:
    """Short advice for the most common Gemini API errors."""
    text = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429 or "429" in text or "resource_exhausted" in text or "quota" in text:
        return ("\n[hint] Limite de débit ou de quota atteinte sur votre compte Gemini : "
                "attendez une minute, vérifiez vos quotas sur aistudio.google.com, "
                "ou essayez un autre modèle (LLM_MODEL dans .env).")
    if code == 404 or "not found" in text:
        return "\n[hint] Modèle introuvable : vérifiez LLM_MODEL dans .env."
    if "api key" in text or code in (401, 403):
        return "\n[hint] Clé refusée : vérifiez GEMINI_API_KEY dans .env."
    return ""
