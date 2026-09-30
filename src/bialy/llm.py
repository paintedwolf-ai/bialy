"""Chat calls to OpenAI-compatible endpoints: the factory's own vLLM servers, or a
provider serving a pinned open-weights model."""

import json
import re

from openai import OpenAI

from .usage import record_usage

FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class Chat:
    def __init__(self, base_url, model_id, api_key="unused", hosted=False):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=600, max_retries=8)
        self.model = model_id
        self.hosted = hosted

    def complete(self, system, user, temperature=0.0, seed=None, max_tokens=8192, thinking=True, schema=None):
        """The reply text, with reasoning kept out: the servers run a reasoning parser, so
        `content` is the answer alone. `thinking=False` asks the chat template to skip the
        reasoning phase where the model supports it; a provider takes the lowest reasoning
        effort instead, since some hosted models only answer after reasoning."""
        extra = {"seed": seed} if seed is not None else {}
        if not thinking:
            extra["extra_body"] = {"reasoning_effort": "low"} if self.hosted else {"chat_template_kwargs": {"enable_thinking": False}}
        if schema is not None:
            # Constrained decoding: the server only emits JSON matching the schema.
            extra["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": True}}
        reply = self.client.chat.completions.create(
            model=self.model, temperature=temperature, top_p=1.0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **extra)
        record_usage(self.model, reply)
        return reply.choices[0].message.content or ""

    def json(self, system, user, **kw):
        """A reply parsed as one JSON value, tolerating a code fence around it."""
        text = FENCE.sub("", self.complete(system, user, **kw).strip())
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        if start < 0:
            raise ValueError("no JSON in reply: %r" % text[:200])
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return value
