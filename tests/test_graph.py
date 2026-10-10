"""Tests of the agent graph with fake LLMs and a fake retriever: no network, no database, no model."""
import json
import logging
import types

import pytest

from agent.graph import MAX_ATTEMPTS, build_graph, to_mermaid
from agent.nodes import parse_verdict
from llm_client import LLMClient, ResponseCache
from prompts import SYSTEM_PROMPT, VERIFIER_SYSTEM_PROMPT

QUESTION = "Combien d'avions Airbus a-t-il livrés en 2023 ?"
OK = json.dumps({"verdict": "ok", "probleme": ""})
COMPLETE = ("En 2023, Airbus a livré 735 avions commerciaux [1] : 68 A220, 571 A320, 32 A330 et 64 A350 [1].")
INCOMPLETE = "En 2023, Airbus a livré 735 avions [1], répartis ainsi : 68 A220, 571 A320 et 32 A330 [1]."
SUM_PROBLEM = "La liste est présentée comme complète mais 68 + 571 + 32 = 671 ≠ 735 : il manque les 64 A350."


class FakeLLM:
    """Returns scripted answers in order and records every (system, user) prompt it receives."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def ask(self, system, user):
        self.prompts.append((system, user))
        if not self.answers:
            raise AssertionError("LLM called more often than expected")
        return self.answers.pop(0)


class FakeRetriever:
    def __init__(self):
        self.questions = []

    def retrieve(self, question):
        self.questions.append(question)
        node = types.SimpleNamespace(
            metadata={"file_name": "airbus_urd_2023_en.pdf", "page_label": "44"},
            get_content=lambda: "A220 family: 68 A220 delivered; A320 family: 571; A330: 32; A350: 64.")
        return [types.SimpleNamespace(node=node, score=0.73)]


def run(writer, verifier):
    retriever = FakeRetriever()
    final = build_graph(retriever, writer, verifier).invoke({"question": QUESTION})
    return final, retriever


def test_correct_answer_ends_after_one_attempt():
    writer, verifier = FakeLLM(COMPLETE), FakeLLM(OK)
    final, retriever = run(writer, verifier)

    assert retriever.questions == [QUESTION]
    assert final["verdict"] == "ok"
    assert final["attempts"] == 1
    assert final["draft"] == COMPLETE
    assert len(writer.prompts) == 1 and len(verifier.prompts) == 1
    assert writer.prompts[0][0] == SYSTEM_PROMPT
    assert verifier.prompts[0][0] == VERIFIER_SYSTEM_PROMPT
    assert "[1] airbus_urd_2023_en.pdf, page 44" in writer.prompts[0][1]
    assert COMPLETE in verifier.prompts[0][1]  # the verifier sees the draft


def test_incomplete_list_is_rewritten_then_accepted():
    writer = FakeLLM(INCOMPLETE, COMPLETE)
    verifier = FakeLLM(json.dumps({"verdict": "a_corriger", "probleme": SUM_PROBLEM}), OK)
    final, _ = run(writer, verifier)

    assert final["attempts"] == 2
    assert final["verdict"] == "ok"
    assert final["draft"] == COMPLETE and "64 A350" in final["draft"]
    second_prompt = writer.prompts[1][1]
    assert SUM_PROBLEM in second_prompt and INCOMPLETE in second_prompt  # the writer got the comment
    assert [s["verdict"] for s in final["history"]] == ["a_corriger", "ok"]
    assert final["history"][0]["draft"] == INCOMPLETE and final["history"][0]["feedback"] == SUM_PROBLEM


def test_verifier_refusing_twice_stops_after_two_attempts():
    refuse = json.dumps({"verdict": "a_corriger", "probleme": "chiffre absent des sources citées"})
    writer, verifier = FakeLLM("brouillon 1 [1]", "brouillon 2 [1]"), FakeLLM(refuse, refuse)
    final, _ = run(writer, verifier)  # a third writer call would raise in FakeLLM

    assert MAX_ATTEMPTS == 2
    assert final["attempts"] == 2
    assert final["verdict"] == "a_corriger"
    assert final["draft"] == "brouillon 2 [1]"
    assert len(writer.prompts) == 2 and len(verifier.prompts) == 2
    assert len(final["history"]) == 2


def test_invalid_verifier_json_counts_as_ok_and_is_logged(caplog):
    writer, verifier = FakeLLM(COMPLETE), FakeLLM("Tout me semble correct !")
    with caplog.at_level(logging.WARNING, logger="aerodoc.agent"):
        final, _ = run(writer, verifier)

    assert final["verdict"] == "ok" and final["attempts"] == 1
    assert "invalid JSON" in caplog.text


def test_refusal_without_sources_goes_through():
    refusal = "Je ne trouve pas cette information dans les documents."
    final, _ = run(FakeLLM(refusal), FakeLLM(OK))
    assert final["draft"] == refusal and final["verdict"] == "ok"


@pytest.mark.parametrize("raw, expected", [
    ('{"verdict": "ok", "probleme": ""}', ("ok", "")),
    ('```json\n{"verdict": "a_corriger", "probleme": "il manque l\'A350"}\n```', ("a_corriger", "il manque l'A350")),
    ('Voici mon verdict : {"verdict": "OK", "probleme": null}', ("ok", "")),
    ('{"verdict": "peut-etre", "probleme": "x"}', None),
    ("pas de json", None),
    ('{"verdict": "ok", ', None),
])
def test_parse_verdict(raw, expected):
    assert parse_verdict(raw) == expected


def test_cache_avoids_a_second_api_call(tmp_path):
    class ChatLLM:  # LlamaIndex-like: .chat(messages) -> response.message.content
        calls = 0

        def chat(self, messages):
            ChatLLM.calls += 1
            return types.SimpleNamespace(message=types.SimpleNamespace(content=f"réponse {ChatLLM.calls}"))

    client = LLMClient(ChatLLM(), "fake-model", pause_s=0, cache=ResponseCache(tmp_path))
    assert client.ask("S", "U") == "réponse 1"
    assert client.ask("S", "U") == "réponse 1"      # same prompt: served by the cache
    assert client.ask("S", "U bis") == "réponse 2"  # different prompt: new call
    assert ChatLLM.calls == 2 and client.real_calls == 2 and client.cache_hits == 1

    other_model = LLMClient(ChatLLM(), "other-model", pause_s=0, cache=ResponseCache(tmp_path))
    assert other_model.ask("S", "U") == "réponse 3"  # the model is part of the cache key


def test_mermaid_export_has_the_three_nodes():
    mermaid = to_mermaid()
    for node in ("retrieve", "write", "verify"):
        assert node in mermaid
