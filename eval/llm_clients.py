from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass


@dataclass
class GenerationRequest:
    model: str
    system_prompt: str
    user_prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None


@dataclass
class GenerationResponse:
    ok: bool
    output_text: str
    provider: str
    model_name: str
    model_version: str | None
    finish_reason: str | None
    seed_supported: bool
    latency_ms: float | None
    error_type: str | None
    error_message: str | None


FIXED_PROSE = (
    "The room kept its composure though every face within it seemed to have lost one. "
    "A servant moved quietly at the edge of the carpet, as if discretion were the chief article of the household. "
    "No one spoke at first, and that silence itself became the event. "
    "When the first reply at last arrived it was civil enough in phrase, yet the civility had been sharpened to a private use. "
    "One heard in it not peace but management. "
    "The curtains admitted a pale light that made every gesture appear slower and more deliberate than it was. "
    "A question was asked with care; the answer, though brief, enlarged the uneasiness of the table. "
    "Each person seemed aware that something had already been decided elsewhere, and that the present conversation existed only to give it a respectable shape."
)


def _echo_prompt_body(user_prompt: str, *, max_words: int = 220) -> str:
    words = user_prompt.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _stub_response(request: GenerationRequest, model_name: str) -> str:
    if model_name == "echo_prompt":
        return _echo_prompt_body(request.user_prompt)
    seed_prefix = hashlib.sha1(f"{request.seed}|{request.user_prompt}".encode("utf-8")).hexdigest()[:8]
    return f"{FIXED_PROSE} The hour carried the sign {seed_prefix} as if the day itself had agreed to remember it."


def _openai_generate(request: GenerationRequest, model_name: str) -> GenerationResponse:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.responses.create(
        model=model_name,
        instructions=request.system_prompt,
        input=request.user_prompt,
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=request.max_tokens,
    )
    output_text = getattr(response, "output_text", None)
    if output_text is None:
        output_parts = []
        for item in getattr(response, "output", []):
            for content_item in getattr(item, "content", []):
                if getattr(content_item, "type", None) == "output_text":
                    output_parts.append(getattr(content_item, "text", ""))
        output_text = "".join(output_parts)
    return GenerationResponse(
        ok=True,
        output_text=(output_text or "").strip(),
        provider="openai",
        model_name=model_name,
        model_version=getattr(response, "model", None),
        finish_reason=None,
        seed_supported=False,
        latency_ms=None,
        error_type=None,
        error_message=None,
    )


def _anthropic_generate(request: GenerationRequest, model_name: str) -> GenerationResponse:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=model_name,
        system=request.system_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        messages=[{"role": "user", "content": request.user_prompt}],
    )
    text_parts = [block.text for block in message.content if getattr(block, "type", None) == "text"]
    return GenerationResponse(
        ok=True,
        output_text="".join(text_parts).strip(),
        provider="anthropic",
        model_name=model_name,
        model_version=getattr(message, "model", None),
        finish_reason=getattr(message, "stop_reason", None),
        seed_supported=False,
        latency_ms=None,
        error_type=None,
        error_message=None,
    )


def generate_text(request: GenerationRequest) -> GenerationResponse:
    start = time.perf_counter()
    provider, _, model_name = request.model.partition(":")
    try:
        if provider == "stub":
            output = _stub_response(request, model_name or "fixed_prose")
            return GenerationResponse(
                ok=True,
                output_text=output.strip(),
                provider="stub",
                model_name=model_name or "fixed_prose",
                model_version="local",
                finish_reason="stop",
                seed_supported=True,
                latency_ms=(time.perf_counter() - start) * 1000,
                error_type=None,
                error_message=None,
            )
        if provider == "openai":
            response = _openai_generate(request, model_name or request.model)
        elif provider == "anthropic":
            response = _anthropic_generate(request, model_name or request.model)
        else:
            return GenerationResponse(
                ok=False,
                output_text="",
                provider=provider or "unknown",
                model_name=model_name or request.model,
                model_version=None,
                finish_reason=None,
                seed_supported=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                error_type="provider_unavailable",
                error_message=f"Unsupported provider: {provider}",
            )
        response.latency_ms = (time.perf_counter() - start) * 1000
        return response
    except Exception as exc:
        return GenerationResponse(
            ok=False,
            output_text="",
            provider=provider or "unknown",
            model_name=model_name or request.model,
            model_version=None,
            finish_reason=None,
            seed_supported=False,
            latency_ms=(time.perf_counter() - start) * 1000,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
