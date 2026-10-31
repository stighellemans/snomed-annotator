"""Thin OpenAI structured-output helper (mirrors the prototype's working pattern)."""
from __future__ import annotations

import json
from typing import Any, Dict

from openai import OpenAI

from .config import settings

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (see backend/.env.example).")
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def call_structured(system: str, user: str, schema_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    client = get_client()
    kwargs: Dict[str, Any] = dict(
        model=settings.llm_model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )
    # gpt-5 series: cap reasoning effort so simple structured tasks don't burn latency.
    if settings.llm_reasoning_effort:
        kwargs["reasoning"] = {"effort": settings.llm_reasoning_effort}
    response = client.responses.create(**kwargs)
    try:
        return json.loads(response.output_text)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise RuntimeError("OpenAI response was not valid JSON") from exc
