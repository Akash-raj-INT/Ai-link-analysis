"""Provider-agnostic LLM client (Anthropic / OpenAI / Gemini via plain HTTPS with httpx).

Responsibilities: call the model, parse + validate JSON against a Pydantic model, repair once if invalid,
fall back to a second provider if the first fails, and report measured latency / token usage.
No SDKs are used on purpose: fewer dependencies and the request shape is explicit for review.
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings, estimate_cost_usd
from app.errors import LLMConfigError, LLMOutputError, LLMServiceError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

RETRY_STATUS = {429, 500, 502, 503, 504, 529}


@dataclass
class RawCompletion:
    text: str
    input_tokens: int
    output_tokens: int
    truncated: bool = False


@dataclass
class LLMUsage:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 0
    used_fallback: bool = False

    @property
    def estimated_cost_usd(self) -> Optional[float]:
        return estimate_cost_usd(self.model, self.input_tokens, self.output_tokens)


# --------------------------------------------------------------------------- HTTP helper
def _error_detail(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        err = data.get("error", data)
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return (msg or resp.text)[:200]
    except Exception:
        return resp.text[:200]


async def _post_json(client: httpx.AsyncClient, provider: str, url: str, headers: dict, payload: dict) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            resp = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            last_exc = LLMServiceError(f"{provider} request timed out.")
        except httpx.RequestError as exc:
            last_exc = LLMServiceError(f"Could not reach {provider} ({type(exc).__name__}).")
        else:
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (401, 403):
                raise LLMServiceError(f"{provider} rejected the API key (HTTP {resp.status_code}). Check your credentials.")
            if resp.status_code == 400:
                raise LLMServiceError(f"{provider} rejected the request: {_error_detail(resp)}")
            last_exc = LLMServiceError(f"{provider} returned HTTP {resp.status_code}: {_error_detail(resp)}")
            if resp.status_code not in RETRY_STATUS:
                raise last_exc
        if attempt == 0:
            await asyncio.sleep(1.5)
    raise last_exc  # type: ignore[misc]


# --------------------------------------------------------------------------- providers
class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str, settings: Settings):
        self.model, self.api_key, self.s = model, api_key, settings

    async def complete(self, client, system: str, messages: list[dict]) -> RawCompletion:
        payload = {
            "model": self.model,
            "max_tokens": self.s.llm_max_output_tokens,
            "system": system,
            "messages": messages,
        }
        if self.s.llm_temperature is not None:
            payload["temperature"] = self.s.llm_temperature
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = await _post_json(client, self.name, "https://api.anthropic.com/v1/messages", headers, payload)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return RawCompletion(
            text, usage.get("input_tokens", 0), usage.get("output_tokens", 0), data.get("stop_reason") == "max_tokens"
        )


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, api_key: str, settings: Settings):
        self.model, self.api_key, self.s = model, api_key, settings

    async def complete(self, client, system: str, messages: list[dict]) -> RawCompletion:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": self.s.llm_max_output_tokens,
        }
        if self.s.llm_temperature is not None:
            payload["temperature"] = self.s.llm_temperature
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = await _post_json(client, self.name, "https://api.openai.com/v1/chat/completions", headers, payload)
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage", {})
        return RawCompletion(
            (choice.get("message") or {}).get("content") or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            choice.get("finish_reason") == "length",
        )


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, api_key: str, settings: Settings):
        self.model, self.api_key, self.s = model, api_key, settings

    async def complete(self, client, system: str, messages: list[dict]) -> RawCompletion:
        gen_cfg = {"responseMimeType": "application/json", "maxOutputTokens": self.s.llm_max_output_tokens}
        if self.s.llm_temperature is not None:
            gen_cfg["temperature"] = self.s.llm_temperature
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages
            ],
            "generationConfig": gen_cfg,
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        data = await _post_json(client, self.name, url, headers, payload)
        cand = (data.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
        usage = data.get("usageMetadata", {})
        # "thinking" tokens are billed as output tokens
        out_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        return RawCompletion(text, usage.get("promptTokenCount", 0), out_tokens, cand.get("finishReason") == "MAX_TOKENS")


PROVIDERS = {"anthropic": AnthropicProvider, "openai": OpenAIProvider, "gemini": GeminiProvider}


def get_provider(name: str, settings: Settings):
    key = settings.api_key_for(name)
    if not key:
        raise LLMConfigError(f"No API key configured for provider '{name}'. Set {name.upper()}_API_KEY in your .env file.")
    return PROVIDERS[name](settings.model_for(name), key, settings)


def configured_providers(settings: Settings) -> list:
    """Primary provider first, then the optional fallback - only those that have an API key."""
    names = [settings.llm_provider]
    if settings.llm_fallback_provider and settings.llm_fallback_provider != settings.llm_provider:
        names.append(settings.llm_fallback_provider)
    providers = [get_provider(n, settings) for n in names if settings.api_key_for(n)]
    if not providers:
        raise LLMConfigError(
            f"No API key configured for '{settings.llm_provider}'. Copy .env.example to .env and set {settings.llm_provider.upper()}_API_KEY."
        )
    return providers


# --------------------------------------------------------------------------- JSON handling
def extract_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences or stray prose around it."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


async def _structured_with_repair(provider, client, system: str, user: str, model_cls: Type[T], usage: LLMUsage) -> T:
    messages = [{"role": "user", "content": user}]
    last_error = ""
    for attempt in range(2):  # first try + one repair
        t0 = time.perf_counter()
        raw = await provider.complete(client, system, messages)
        usage.latency_ms += int((time.perf_counter() - t0) * 1000)
        usage.input_tokens += raw.input_tokens
        usage.output_tokens += raw.output_tokens
        usage.attempts += 1
        if raw.truncated:
            raise LLMOutputError("The model's answer was cut off (output token limit). Increase LLM_MAX_OUTPUT_TOKENS.")
        try:
            return model_cls.model_validate(extract_json(raw.text))
        except (json.JSONDecodeError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                last_error = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:8])
            else:
                last_error = f"not valid JSON ({exc})"
            log.warning("LLM output invalid on attempt %d: %s", attempt + 1, last_error)
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": raw.text[:20000]},
                {
                    "role": "user",
                    "content": f"Your previous reply was invalid: {last_error}. Return the complete corrected JSON object only.",
                },
            ]
    raise LLMOutputError(f"The model did not return valid structured output: {last_error}")


async def run_structured(
    system: str, user: str, model_cls: Type[T], settings: Settings, client: Optional[httpx.AsyncClient] = None
) -> tuple[T, LLMUsage]:
    providers = configured_providers(settings)
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=settings.llm_timeout_seconds)
    last_error: Optional[Exception] = None
    try:
        for i, provider in enumerate(providers):
            usage = LLMUsage(provider=provider.name, model=provider.model, used_fallback=i > 0)
            try:
                return await _structured_with_repair(provider, client, system, user, model_cls, usage), usage
            except (LLMServiceError, LLMOutputError) as exc:
                log.warning("Provider %s failed: %s", provider.name, exc.message)
                last_error = exc
        raise last_error  # type: ignore[misc]
    finally:
        if own_client:
            await client.aclose()
