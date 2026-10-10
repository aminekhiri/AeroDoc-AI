"""Test doubles shared by the tests: no network, no database, no model."""
import types


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

    def ask_structured(self, system, user, schema):
        """Structured output: the scripted answer is a dict, a JSON string or already a schema instance."""
        answer = self.ask(system, user)
        if isinstance(answer, schema):
            return answer
        if isinstance(answer, dict):
            return schema.model_validate(answer)
        return schema.model_validate_json(answer)


def router(route, *more):
    """FakeLLM for the router: first the route decision, then the other scripted answers
    (rewritten question, answer from the history...)."""
    return FakeLLM({"route": route, "raison": "test"}, *more)


class FakeRetriever:
    """Always returns the same Airbus passage and records the questions it was asked."""

    def __init__(self):
        self.questions = []

    def retrieve(self, question):
        self.questions.append(question)
        node = types.SimpleNamespace(
            metadata={"file_name": "airbus_urd_2023_en.pdf", "page_label": "44"},
            get_content=lambda: "A220 family: 68 A220 delivered; A320 family: 571; A330: 32; A350: 64.")
        return [types.SimpleNamespace(node=node, score=0.73)]
