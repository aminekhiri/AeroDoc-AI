"""Tests of the sliding-window rate limiter, with a simulated clock: no real waiting, no network."""
import types

from llm_client import LLMClient, RateLimiter, ResponseCache


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def limiter(max_calls, clock, window_s=60.0):
    return RateLimiter(max_calls, window_s=window_s, clock=clock, sleep=clock.sleep, announce=lambda *a, **k: None)


def test_calls_under_the_limit_never_wait():
    clock = FakeClock()
    rl = limiter(15, clock)
    for _ in range(15):
        assert rl.acquire() == 0.0
        clock.now += 0.5  # 15 calls in 7.5 s: still allowed
    assert clock.slept == []


def test_the_16th_call_waits_until_the_oldest_leaves_the_window():
    clock = FakeClock()
    rl = limiter(15, clock)
    for _ in range(15):
        rl.acquire()
        clock.now += 1.0           # calls at t=0..14 s
    waited = rl.acquire()           # t=15 s: the window is full
    assert waited == 45.0           # the call made at t=0 leaves the window at t=60
    assert clock.slept == [45.0]


def test_the_window_slides():
    clock = FakeClock()
    rl = limiter(2, clock)
    rl.acquire()                    # t=0
    clock.now += 50
    rl.acquire()                    # t=50
    clock.now += 15                 # t=65: the first call is out of the window
    assert rl.acquire() == 0.0
    assert rl.acquire() == 45.0     # t=65: full again (50 and 65); 50 leaves at 110


def test_a_single_question_of_4_calls_never_waits():
    """What the chat does for a follow-up question: route, rewriting, write, verify."""
    clock = FakeClock()
    rl = limiter(15, clock)
    for _ in range(4):
        assert rl.acquire() == 0.0
        clock.now += 2.0  # API answer time
    assert clock.slept == []


def test_zero_means_no_limit():
    clock = FakeClock()
    rl = limiter(0, clock)
    assert all(rl.acquire() == 0.0 for _ in range(100))


def test_client_uses_the_limiter_but_not_for_cached_answers(tmp_path):
    class CountingLimiter:
        acquired = 0

        def acquire(self):
            CountingLimiter.acquired += 1
            return 0.0

    class ChatLLM:
        def chat(self, messages):
            return types.SimpleNamespace(message=types.SimpleNamespace(content="réponse"))

    client = LLMClient(ChatLLM(), "fake", cache=ResponseCache(tmp_path), limiter=CountingLimiter())
    client.ask("S", "U")
    client.ask("S", "U")   # served by the cache: no slot of the quota is used
    assert CountingLimiter.acquired == 1 and client.real_calls == 1 and client.cache_hits == 1


def test_a_failed_call_still_takes_a_slot():
    clock = FakeClock()
    rl = limiter(1, clock)

    class Boom:
        def chat(self, messages):
            raise RuntimeError("503")

    client = LLMClient(Boom(), "fake", limiter=rl)
    try:
        client.ask("S", "U")
    except RuntimeError:
        pass
    assert len(rl.starts) == 1 and client.real_calls == 1
