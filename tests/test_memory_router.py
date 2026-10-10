"""Tests of the conversation memory (LangGraph checkpointer) and of the router, with fakes:
no network, no database, no model."""
import json
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

from agent.graph import build_graph, turn_input
from agent.memory import turns_from_messages
from agent.nodes import RouteDecision
from llm_client import LLMClient, ResponseCache
from prompts import CONDENSE_SYSTEM_PROMPT, CONVERSATION_SYSTEM_PROMPT, REFUSAL_TEXT, ROUTER_SYSTEM_PROMPT

from fakes import FakeLLM, FakeRetriever

OK = json.dumps({"verdict": "ok", "probleme": ""})
DOC, CONV, OUT = ({"route": r, "raison": "test"} for r in ("documents", "conversation", "hors_perimetre"))


class Conversation:
    """A chat with a MemorySaver checkpointer: the history is kept between the turns of one thread."""

    def __init__(self, router, writer, verifier, history_turns=10):
        self.retriever = FakeRetriever()
        self.graph = build_graph(self.retriever, writer, verifier, router=router,
                                 history_turns=history_turns, checkpointer=MemorySaver())
        self.thread = "t1"

    def ask(self, question):
        return self.graph.invoke(turn_input(question), {"configurable": {"thread_id": self.thread}})


def test_list_my_last_3_questions_goes_to_conversation_with_the_exact_questions():
    q1, q2, q3 = ("Combien d'avions Airbus a-t-il livrés en 2023 ?", "Et en 2022 ?",
                  "Combien de salariés Thales comptait-il fin 2025 ?")
    listing = f"1. {q1}\n2. {q2}\n3. {q3}"
    router = FakeLLM(DOC,
                     DOC, "Combien d'avions Airbus a-t-il livrés en 2022 ?",  # q2 rewritten for the search
                     DOC, q3,
                     CONV, listing)                                             # route + answer from history
    writer = FakeLLM("735 avions [1].", "661 avions [1].", "84 958 salariés [1].")
    chat = Conversation(router, writer, FakeLLM(OK, OK, OK))
    for q in (q1, q2, q3):
        chat.ask(q)

    final = chat.ask("liste mes 3 dernières questions")

    assert final["route"] == "conversation"
    assert final["draft"] == listing
    system, user = router.prompts[-1]
    assert system == CONVERSATION_SYSTEM_PROMPT
    # the questions given to the LLM are the ones TYPED by the user (q2 is "Et en 2022 ?", not its rewrite)
    expected = [f"Échange 1 - Utilisateur : {q1}", f"Échange 2 - Utilisateur : {q2}", f"Échange 3 - Utilisateur : {q3}"]
    positions = [user.index(line) for line in expected]
    assert positions == sorted(positions)
    assert user.endswith("Dernière question : liste mes 3 dernières questions")
    assert chat.retriever.questions == [q1, "Combien d'avions Airbus a-t-il livrés en 2022 ?", q3]  # no search
    assert writer.answers == [] and len(final["messages"]) == 8  # 4 questions + 4 answers


def test_follow_up_et_en_2023_goes_to_documents_rewritten_with_airbus_and_2023():
    q1 = "combien d'avions Airbus a livrés en 2022 ?"
    router = FakeLLM(DOC, DOC, "Combien d'avions Airbus a-t-il livrés en 2023 ?")
    chat = Conversation(router, FakeLLM("En 2022, Airbus a livré 661 avions [1].", "735 avions [1]."),
                        FakeLLM(OK, OK))
    chat.ask(q1)

    final = chat.ask("et en 2023 ?")

    assert final["route"] == "documents"
    assert "Airbus" in final["question"] and "2023" in final["question"]
    assert chat.retriever.questions[-1] == final["question"]
    system, user = router.prompts[-1]
    assert system == CONDENSE_SYSTEM_PROMPT
    assert q1 in user and "661 avions" in user and user.endswith("Dernière question : et en 2023 ?")


def test_reference_to_a_question_asked_4_turns_before_is_resolved():
    first = "Combien d'avions Airbus a-t-il livrés en 2023 ?"
    others = ["Combien de salariés Thales comptait-il fin 2025 ?",
              "Combien de moteurs LEAP Safran a-t-il livrés en 2025 ?",
              "En combien de secondes doit-on évacuer un grand avion selon CS-25 ?"]
    follow_up = "Et pour ma première question, quel était le chiffre en 2022 ?"
    rewritten = "Combien d'avions Airbus a-t-il livrés en 2022 ?"
    router = FakeLLM(DOC, *[x for q in others for x in (DOC, q)], DOC, rewritten)
    chat = Conversation(router, FakeLLM("735 [1].", "84 958 [1].", "1 802 [1].", "90 secondes [1].", "661 [1]."),
                        FakeLLM(OK, OK, OK, OK, OK))
    for q in [first, *others]:
        chat.ask(q)

    final = chat.ask(follow_up)  # turn 5: the question referred to is 4 turns back

    system, user = router.prompts[-1]
    assert system == CONDENSE_SYSTEM_PROMPT
    assert f"Échange 1 - Utilisateur : {first}" in user
    assert final["question"] == rewritten and chat.retriever.questions[-1] == rewritten


def test_history_window_follows_history_turns():
    """With the old window of 3 exchanges, the question asked 4 turns before is no longer visible."""
    first = "Combien d'avions Airbus a-t-il livrés en 2023 ?"
    router = FakeLLM(DOC, DOC, "q2", DOC, "q3", DOC, "q4", DOC, "question réécrite")
    chat = Conversation(router, FakeLLM("a1", "a2", "a3", "a4", "a5"), FakeLLM(OK, OK, OK, OK, OK), history_turns=3)
    for q in (first, "q2", "q3", "q4", "et en 2022 ?"):
        chat.ask(q)
    assert first not in router.prompts[-1][1]
    assert "Échange 3 - Utilisateur : q4" in router.prompts[-1][1]


def test_out_of_scope_is_refused_without_search_nor_writer():
    writer = FakeLLM()  # any call would raise
    chat = Conversation(FakeLLM(OUT), writer, FakeLLM())
    final = chat.ask("Qui a gagné la dernière finale de la Ligue des champions ?")
    assert final["route"] == "hors_perimetre"
    assert final["draft"] == REFUSAL_TEXT and final["passages"] == []
    assert chat.retriever.questions == [] and writer.prompts == []
    assert [m.type for m in final["messages"]] == ["human", "ai"]


def test_router_failure_falls_back_to_documents_and_is_logged(caplog):
    router = FakeLLM("ceci n'est pas du JSON")
    with caplog.at_level(logging.WARNING, logger="aerodoc.agent"):
        final = Conversation(router, FakeLLM("735 avions [1]."), FakeLLM(OK)).ask("Combien d'avions Airbus en 2023 ?")
    assert final["route"] == "documents" and final["draft"] == "735 avions [1]."
    assert "route forced to 'documents'" in caplog.text


def test_router_sees_the_history_and_threads_are_isolated():
    router = FakeLLM(DOC, DOC, "q2 réécrite", DOC)
    chat = Conversation(router, FakeLLM("a1", "a2", "a3"), FakeLLM(OK, OK, OK))
    chat.ask("q1")
    chat.ask("q2")
    route_system, route_user = router.prompts[1]
    assert route_system == ROUTER_SYSTEM_PROMPT and "Échange 1 - Utilisateur : q1" in route_user

    chat.thread = "t2"  # /reset in the chat: new thread_id, empty history, no rewriting call
    final = chat.ask("q3")
    assert "(aucun échange précédent)" in router.prompts[-1][1]
    assert final["question"] == "q3" and len(final["messages"]) == 2


def test_per_turn_fields_are_reset_between_turns():
    refuse = json.dumps({"verdict": "a_corriger", "probleme": "x"})
    chat = Conversation(FakeLLM(DOC, DOC, "q2"), FakeLLM("d1", "d2", "d3"), FakeLLM(refuse, refuse, OK))
    first = chat.ask("q1")
    assert first["attempts"] == 2 and len(first["steps"]) == 2
    second = chat.ask("q2")  # without the reset, attempts would start at 2 and the loop would stop
    assert second["attempts"] == 1 and len(second["steps"]) == 1 and second["verdict"] == "ok"


def test_turns_from_messages_pairs_questions_and_answers():
    messages = [HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2 sans réponse (erreur)"),
                HumanMessage("q3"), AIMessage("a3"), HumanMessage("q4"), AIMessage("a4")]
    assert turns_from_messages(messages, 10) == [("q1", "a1"), ("q3", "a3"), ("q4", "a4")]
    assert turns_from_messages(messages, 2) == [("q3", "a3"), ("q4", "a4")]
    assert turns_from_messages(messages, 0) == []


def test_structured_output_is_validated_and_cached(tmp_path):
    class StructuredLLM:
        calls = 0

        def structured_predict_without_function_calling(self, schema, prompt):
            StructuredLLM.calls += 1
            assert "SYSTEM" in prompt.format_messages()[0].content
            return schema(route="conversation", raison="question sur l'échange")

    client = LLMClient(StructuredLLM(), "fake", pause_s=0, cache=ResponseCache(tmp_path))
    first = client.ask_structured("SYSTEM", "liste mes questions", RouteDecision)
    second = client.ask_structured("SYSTEM", "liste mes questions", RouteDecision)
    assert isinstance(first, RouteDecision) and first.route == "conversation"
    assert second == first and StructuredLLM.calls == 1 and client.cache_hits == 1


def test_route_decision_rejects_unknown_routes():
    with pytest.raises(ValidationError):
        RouteDecision.model_validate({"route": "meteo"})

