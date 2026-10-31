"""Provider adapters for clinical extraction and concept selection.

LLM credentials/endpoints are independent of the optional OpenAI embeddings.
No request is silently retried against a different provider or output mode.
"""
from __future__ import annotations

from functools import lru_cache
import json
from typing import Any
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft202012Validator
from openai import APIConnectionError, APIStatusError, OpenAI

from .config import settings


def api_style() -> str:
    return settings.llm_api or {
        "openai": "responses",
        "openai_compatible": "chat_completions",
        "anthropic": "messages",
    }.get(settings.llm_provider, "")


def base_url() -> str:
    return settings.llm_base_url or {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
    }.get(settings.llm_provider, "")


def api_key() -> str:
    if settings.llm_api_key:
        return settings.llm_api_key
    if settings.llm_provider == "openai":
        return settings.openai_api_key
    if settings.llm_provider == "anthropic":
        return settings.anthropic_api_key
    # Never send an OpenAI key to a different service by accident.
    return ""


def configuration_errors() -> list[str]:
    errors = []
    if settings.llm_provider not in {"openai", "openai_compatible", "anthropic"}:
        errors.append("LLM_PROVIDER: kies openai, openai_compatible of anthropic.")
    if not settings.llm_model.strip():
        errors.append("Vul LLM_MODEL in met de modelnaam van je aanbieder.")
    try:
        url = urlsplit(base_url())
        valid_url = url.scheme in {"http", "https"} and bool(url.netloc)
    except ValueError:
        valid_url, url = False, None
    if not valid_url:
        errors.append("LLM_BASE_URL moet een volledig http(s)-adres zijn, inclusief API-pad (bijvoorbeeld /v1).")
    if url and (url.username or url.password or url.query or url.fragment):
        errors.append("Zet de sleutel in LLM_API_KEY; LLM_BASE_URL mag geen login, query of fragment bevatten.")
    if settings.llm_provider != "openai_compatible" and not api_key():
        errors.append("Vul LLM_API_KEY in (of OPENAI_API_KEY / ANTHROPIC_API_KEY voor de gekozen aanbieder).")
    allowed = {"messages"} if settings.llm_provider == "anthropic" else {"responses", "chat_completions"}
    if api_style() not in allowed:
        errors.append("LLM_API past niet bij deze provider: gebruik responses, chat_completions of messages.")
    if settings.llm_json_mode not in {"json_schema", "json_object", "prompt"}:
        errors.append("LLM_JSON_MODE: kies json_schema, json_object of prompt.")
    if settings.llm_provider == "anthropic" and settings.llm_json_mode != "json_schema":
        errors.append("De Anthropic-adapter gebruikt LLM_JSON_MODE=json_schema (een verplichte tool met schema).")
    if settings.llm_timeout_seconds <= 0 or settings.llm_max_tokens <= 0:
        errors.append("LLM_TIMEOUT_SECONDS en LLM_MAX_TOKENS moeten groter zijn dan nul.")
    return errors


def cache_identity() -> tuple:
    """Include every model setting, but never a credential, in annotation keys."""
    return (
        settings.llm_provider, base_url().rstrip("/"), api_style(), settings.llm_model,
        settings.llm_json_mode, settings.llm_reasoning_effort, settings.llm_max_tokens,
    )


@lru_cache(maxsize=4)
def _llm_client(endpoint: str, key: str, timeout: float) -> OpenAI:
    return OpenAI(base_url=endpoint, api_key=key or "local", timeout=timeout, max_retries=2)


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY ontbreekt voor de optionele OpenAI-zoekindex. BioLORD heeft die niet nodig.")
    return OpenAI(api_key=settings.openai_api_key, base_url="https://api.openai.com/v1")


def _validated(data: Any, schema: dict, name: str) -> dict:
    try:
        json.dumps(data, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"LLM-antwoord voor {name} bevat geen geldige JSON-waarden.") from exc
    error = next(Draft202012Validator(schema).iter_errors(data), None)
    if error is not None:
        path = ".".join(str(part) for part in error.absolute_path) or "antwoord"
        raise RuntimeError(f"LLM-antwoord voor {name} voldoet niet aan het schema bij {path}.")
    return data


def _reject_constant(value: str):
    raise ValueError(f"Invalid JSON constant: {value}")


def _decode(text: str, schema: dict, name: str) -> dict:
    # Prompt-only servers often wrap otherwise valid JSON in a Markdown fence.
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"LLM gaf geen geldig JSON-antwoord voor {name}; controleer LLM_JSON_MODE en het model.") from exc
    return _validated(data, schema, name)


def _anthropic(system: str, user: str, name: str, schema: dict) -> dict:
    # Tool input is used as a structured return value; no external tool is executed.
    with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
        response = client.post(
            base_url().rstrip("/") + "/messages",
            headers={"x-api-key": api_key(), "anthropic-version": "2023-06-01"},
            json={
                "model": settings.llm_model, "max_tokens": settings.llm_max_tokens,
                "system": system, "messages": [{"role": "user", "content": user}],
                "tools": [{"name": name, "description": "Return the structured result.", "input_schema": schema}],
                "tool_choice": {"type": "tool", "name": name},
            },
        )
        response.raise_for_status()
        data = response.json()
    if data.get("stop_reason") == "max_tokens":
        raise RuntimeError("LLM-antwoord afgebroken; verhoog LLM_MAX_TOKENS.")
    for block in data.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == name:
            return _validated(block.get("input"), schema, name)
    raise RuntimeError("De Anthropic-API gaf geen gestructureerd antwoord voor de gevraagde taak.")


def call_structured(system: str, user: str, schema_name: str, schema: dict) -> dict:
    errors = configuration_errors()
    if errors:
        raise RuntimeError(" ".join(errors))
    try:
        if settings.llm_provider == "anthropic":
            return _anthropic(system, user, schema_name, schema)

        client = _llm_client(base_url(), api_key(), settings.llm_timeout_seconds)
        mode = settings.llm_json_mode
        if mode != "json_schema":
            system += "\nReturn only a JSON object matching this schema:\n" + json.dumps(schema)
        output_format = None
        if mode == "json_schema":
            output_format = {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}
        elif mode == "json_object":
            output_format = {"type": "json_object"}

        if api_style() == "responses":
            kwargs: dict[str, Any] = {
                "model": settings.llm_model,
                "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_output_tokens": settings.llm_max_tokens,
            }
            if output_format:
                kwargs["text"] = {"format": output_format}
            if settings.llm_reasoning_effort:
                kwargs["reasoning"] = {"effort": settings.llm_reasoning_effort}
            response = client.responses.create(**kwargs)
            if response.status != "completed":
                raise RuntimeError("LLM-antwoord onvolledig; controleer model en LLM_MAX_TOKENS.")
            text = response.output_text
        else:
            kwargs = {
                "model": settings.llm_model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            token_field = "max_completion_tokens" if settings.llm_provider == "openai" else "max_tokens"
            kwargs[token_field] = settings.llm_max_tokens
            if output_format:
                kwargs["response_format"] = (
                    {"type": "json_schema", "json_schema": {key: value for key, value in output_format.items() if key != "type"}}
                    if mode == "json_schema" else output_format
                )
            if settings.llm_reasoning_effort:
                kwargs["reasoning_effort"] = settings.llm_reasoning_effort
            response = client.chat.completions.create(**kwargs)
            if not response.choices or response.choices[0].finish_reason != "stop":
                raise RuntimeError("LLM-antwoord ontbreekt of is afgebroken; controleer model en LLM_MAX_TOKENS.")
            text = response.choices[0].message.content or ""
        return _decode(text, schema, schema_name)
    except (APIStatusError, httpx.HTTPStatusError) as exc:
        status = exc.status_code if isinstance(exc, APIStatusError) else exc.response.status_code
        raise RuntimeError(f"LLM-API gaf HTTP {status}; controleer provider, endpoint, model en sleutel. Er is geen andere provider gebruikt.") from exc
    except (APIConnectionError, httpx.RequestError) as exc:
        raise RuntimeError("LLM-API niet bereikbaar; controleer LLM_BASE_URL, de verbinding en LLM_TIMEOUT_SECONDS.") from exc
