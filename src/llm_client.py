"""AeroDoc-AI - LLM calls shared by every script that calls the LLM.

LLMClient.ask(system, user) -> text, with:
  - a rate limit: at most LLM_CALLS_PER_MIN real calls in any sliding window of 60 s
    (gemini-3.5-flash-lite: 15 requests/min, 500/day). The limiter is shared by all clients of the
    process (router, writer and verifier use the same quota). A call only waits when the last
    minute is already full: a single question never waits;
  - an optional disk cache of the answers (data/cache/llm/ by default): the same model, system
    prompt and user prompt give the cached answer, without any API call. Any change of model or
    prompt is a new key, so a modified prompt is never answered from an old cache entry.
"""
import hashlib
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

WINDOW_S = 60.0
SAFETY_MARGIN_S = 1.0  # the API clock and ours are not exactly in sync


class RateLimiter:
    """At most `max_calls` calls in any sliding window of `window_s` seconds (0 = no limit)."""

    def __init__(self, max_calls: int, window_s: float = WINDOW_S + SAFETY_MARGIN_S,
                 clock=time.monotonic, sleep=time.sleep, announce=print):
        self.max_calls, self.window_s = max_calls, window_s
        self.clock, self.sleep, self.announce = clock, sleep, announce
        self.starts: deque[float] = deque()  # start times of the calls still inside the window

    def _forget_old(self, now: float) -> None:
        while self.starts and self.starts[0] <= now - self.window_s:
            self.starts.popleft()

    def acquire(self) -> float:
        """Wait if the window is full, then record a call. Returns the seconds waited."""
        if self.max_calls <= 0:
            return 0.0
        now = self.clock()
        self._forget_old(now)
        waited = 0.0
        if len(self.starts) >= self.max_calls:
            waited = self.starts[0] + self.window_s - now
            self.announce(f"[llm] limite de {self.max_calls} appels par minute atteinte : attente {waited:.0f} s",
                          flush=True)
            self.sleep(waited)
            now = self.clock()
            self._forget_old(now)
        self.starts.append(now)  # a failed call also counts against the quota
        return waited


_shared_limiters: dict[int, RateLimiter] = {}


def shared_limiter(max_calls: int) -> RateLimiter:
    """One limiter per process (and per limit value): all clients share the same API quota."""
    if max_calls not in _shared_limiters:
        _shared_limiters[max_calls] = RateLimiter(max_calls)
    return _shared_limiters[max_calls]


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
    def __init__(self, llm, model: str, cache: ResponseCache | None = None, limiter: RateLimiter | None = None):
        self.llm = llm          # any LlamaIndex chat LLM (or a test double with .chat)
        self.model = model
        self.limiter = limiter  # None: no rate limit (tests)
        self.cache = cache
        self.real_calls = 0     # calls actually sent to the API
        self.cache_hits = 0     # answers served by the cache

    def _wait_turn(self) -> None:
        if self.limiter:
            self.limiter.acquire()

    def ask(self, system: str, user: str) -> str:
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
            self.real_calls += 1
        answer = (response.message.content or "").strip()
        if key:
            self.cache.put(key, self.model, answer)
        return answer

    def ask_structured(self, system: str, user: str, schema):
        """Structured output: the API is asked for JSON following the Pydantic `schema`
        (Gemini response_schema); returns a validated instance of `schema`."""
        tag = f"structured:{schema.__name__}:{json.dumps(schema.model_json_schema(), sort_keys=True)}\n{system}"
        key = ResponseCache.key(self.model, tag, user) if self.cache else None
        if key:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                return schema.model_validate_json(cached)

        from llama_index.core.llms import ChatMessage, MessageRole
        from llama_index.core.prompts import ChatPromptTemplate

        # One user message: the Gemini structured call has no separate system instruction.
        prompt = ChatPromptTemplate(message_templates=[
            ChatMessage(role=MessageRole.USER, content=f"{system}\n\n{user}")])
        predict = getattr(self.llm, "structured_predict_without_function_calling", None) or self.llm.structured_predict
        self._wait_turn()
        try:
            result = predict(schema, prompt)
        finally:
            self.real_calls += 1
        if key:
            self.cache.put(key, self.model, result.model_dump_json())
        return result


def make_client(model: str | None = None, use_cache: bool = True) -> LLMClient:
    """Client for the real Gemini API, configured from .env (model, rate limit, cache)."""
    from config import LLM_CACHE, LLM_CACHE_DIR, LLM_CALLS_PER_MIN, LLM_MODEL, get_llm

    model = model or LLM_MODEL
    cache = ResponseCache(LLM_CACHE_DIR) if (use_cache and LLM_CACHE) else None
    return LLMClient(get_llm(model), model, cache=cache, limiter=shared_limiter(LLM_CALLS_PER_MIN))


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
