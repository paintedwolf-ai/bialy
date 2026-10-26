from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("reason", ["length", "content_filter", None])
def test_training_turns_refuse_incomplete_provider_output(reason):
    chat = llm.Chat("https://example.invalid/v1", "m")
    reply = SimpleNamespace(id="r", usage=None, choices=[
        SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content="partial", tool_calls=[])),
    ])
    chat.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: reply)))
    with pytest.raises(ValueError, match="incomplete assistant turn"):
        chat.turn([{"role": "user", "content": "request"}])


def test_training_turns_preserve_calls_and_disable_local_reasoning():
    chat = llm.Chat("https://example.invalid/v1", "m")
    seen = []
    reply = SimpleNamespace(id="r", usage=None, choices=[
        SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(content=None, tool_calls=[
            SimpleNamespace(id="call-1", function=SimpleNamespace(name="read", arguments='{"path":"a.py"}')),
        ])),
    ])

    def create(**kwargs):
        seen.append(kwargs)
        return reply

    chat.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    output = chat.turn([{"role": "user", "content": "request"}], tools=tools)
    assert output["tool_calls"] == [{"id": "call-1", "name": "read", "arguments": '{"path":"a.py"}'}]
    assert seen[0]["tools"] == tools
    assert seen[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_structured_turn_sends_schema_and_preserves_finish_validation():
    chat = llm.Chat("https://example.invalid/v1", "m", hosted=True, reasoning_effort="none")
    sent = calls(chat)
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    assert chat.turn([{"role": "user", "content": "judge"}], schema=schema)["text"] == "{}"
    assert sent[0]["response_format"] == {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": True}}
    assert sent[0]["extra_body"] == {"reasoning_effort": "none"}
