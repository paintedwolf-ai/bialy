from bialy import llm


def calls(chat):
    sent = []

    class Completions:
        def create(self, **kw):
            sent.append(kw)
            reply = type("Reply", (), {})()
            reply.id, reply.usage = "r", None
            reply.choices = [type("Choice", (), {"message": type("Message", (), {"content": "{}"})(), "finish_reason": "stop"})()]
            return reply

    chat.client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    return sent


def test_a_configured_effort_goes_on_every_call():
    chat = llm.Chat("https://example.invalid/v1", "m", hosted=True, reasoning_effort="none")
    sent = calls(chat)
    chat.complete("s", "u", thinking=True)
    chat.complete("s", "u", thinking=False)
    assert [kw["extra_body"] for kw in sent] == [{"reasoning_effort": "none"}] * 2


def test_without_one_a_hosted_model_reasons_low_only_when_asked_not_to_think():
    chat = llm.Chat("https://example.invalid/v1", "m", hosted=True)
    sent = calls(chat)
    chat.complete("s", "u", thinking=True)
    chat.complete("s", "u", thinking=False)
    assert "extra_body" not in sent[0] and sent[1]["extra_body"] == {"reasoning_effort": "low"}
