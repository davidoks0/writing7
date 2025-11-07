import os
import random
from typing import Optional
import time


class LLMError(Exception):
    pass


def _coerce_seed(seed: Optional[int]) -> Optional[int]:
    try:
        return int(seed) if seed is not None else None
    except Exception:
        return None


def generate(
    model: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    max_tokens: int = 800,
    temperature: float = 0.9,
    top_p: float = 0.95,
    seed: Optional[int] = None,
) -> str:
    """Generate text from a provider-specific model.

    model: provider:model_name, e.g. "openai:gpt-4o-mini", "anthropic:claude-3-5-sonnet",
           "gemini:1.5-pro", "kimi:moonshot-v1-8k".
    """
    if ":" not in model:
        raise LLMError("Model must be prefixed with provider, e.g. 'openai:gpt-4o-mini'")
    provider, name = model.split(":", 1)
    provider = provider.lower()
    seed = _coerce_seed(seed)

    if provider == "openai":
        return _gen_openai(name, prompt, system=system, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed)
    if provider == "anthropic":
        return _gen_anthropic(name, prompt, system=system, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed)
    if provider == "gemini":
        return _gen_gemini(name, prompt, system=system, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed)
    if provider == "kimi":
        return _gen_kimi(name, prompt, system=system, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed)
    raise LLMError(f"Unknown provider: {provider}")


def _gen_openai(model: str, prompt: str, *, system: Optional[str], max_tokens: int, temperature: float, top_p: float, seed: Optional[int]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY not set")
    base_url = os.getenv("OPENAI_BASE_URL")
    from openai import OpenAI
    org = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
    project = os.getenv("OPENAI_PROJECT")
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url, organization=org, project=project)
    else:
        client = OpenAI(api_key=api_key, organization=org, project=project)
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    # Basic retry for transient errors; insufficient_quota won't recover
    delays = [0.5, 1.5, 3.0]
    last_err: Exception | None = None
    for attempt, delay in enumerate([0.0] + delays):
        if delay:
            time.sleep(delay)
        try:
            is_gpt5 = "gpt-5" in (model or "").lower()
            if is_gpt5:
                # Prefer Responses API for GPT-5. Omit temperature/top_p.
                # Use `instructions` for system prompt and only user content in `input`.
                def _extract_text(resp_obj):
                    t = getattr(resp_obj, "output_text", None)
                    if t:
                        return t
                    try:
                        chunks = []
                        output = getattr(resp_obj, "output", None) or []
                        for item in output:
                            msg = getattr(item, "content", None) or []
                            for c in msg:
                                if getattr(c, "type", None) == "output_text" and hasattr(c, "text"):
                                    chunks.append(c.text)
                                elif getattr(c, "type", None) == "text" and hasattr(c, "text"):
                                    chunks.append(c.text)
                        return "\n".join(chunks)
                    except Exception:
                        return None

                # Try Responses up to 2 times before falling back
                for _attempt in range(2):
                    create_args = dict(
                        model=model,
                        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                        max_output_tokens=max_tokens,
                    )
                    if system:
                        create_args["instructions"] = system
                    resp = client.responses.create(**create_args)
                    text = (_extract_text(resp) or "").strip()
                    if text:
                        return text
                try:
                    # Chat Completions fallback with GPT-5: use max_completion_tokens and no sampling params
                    cc = client.chat.completions.create(
                        model=model,
                        messages=msgs,
                        max_completion_tokens=max_tokens,
                    )
                    content = cc.choices[0].message.content or ""
                    return content.strip()
                except Exception:
                    # Final fallback: try without token cap
                    try:
                        cc2 = client.chat.completions.create(
                            model=model,
                            messages=msgs,
                        )
                        content = cc2.choices[0].message.content or ""
                        return content.strip()
                    except Exception as _e2:
                        raise _e2
            else:
                # Chat Completions for non-GPT-5 models
                create_args = dict(
                    model=model,
                    messages=msgs,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                cc = client.chat.completions.create(**create_args)
                content = cc.choices[0].message.content or ""
                return content.strip()
        except Exception as e:
            # Map provider error to our error type after last attempt
            last_err = e
            # If clearly quota/permission, break early
            msg = str(e).lower()
            if "insufficient_quota" in msg or "insufficient quota" in msg or "invalid api key" in msg:
                break
            continue
    raise LLMError(f"openai error: {last_err}")


def _gen_anthropic(model: str, prompt: str, *, system: Optional[str], max_tokens: int, temperature: float, top_p: float, seed: Optional[int]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY not set")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    kwargs = {}
    if seed is not None:
        kwargs["metadata"] = {"user_id": f"seed-{seed}"}
    delays = [0.5, 1.5, 3.0]
    last_err: Exception | None = None
    for attempt, delay in enumerate([0.0] + delays):
        if delay:
            time.sleep(delay)
        try:
            # Some Anthropic models disallow specifying both temperature and top_p.
            # Prefer temperature and omit top_p to maximize compatibility.
            create_args = dict(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            create_args.update(kwargs)
            msg = client.messages.create(**create_args)
            parts = []
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
                elif hasattr(block, "text"):
                    parts.append(block.text)
            return ("\n".join(parts)).strip()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "rate limit" in msg or "insufficient" in msg:
                # still retry per schedule
                pass
            continue
    raise LLMError(f"anthropic error: {last_err}")


def _gen_gemini(model: str, prompt: str, *, system: Optional[str], max_tokens: int, temperature: float, top_p: float, seed: Optional[int]) -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    safety = None  # Use defaults
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_tokens,
    }
    m = genai.GenerativeModel(model)
    contents = []
    if system:
        contents.append(system)
    contents.append(prompt)
    delays = [0.5, 1.5, 3.0]
    last_err: Exception | None = None
    for attempt, delay in enumerate([0.0] + delays):
        if delay:
            time.sleep(delay)
        try:
            resp = m.generate_content(contents, safety_settings=safety, generation_config=generation_config)
            text = getattr(resp, "text", None)
            if not text and hasattr(resp, "candidates") and resp.candidates:
                try:
                    text = resp.candidates[0].content.parts[0].text
                except Exception:
                    text = ""
            return (text or "").strip()
        except Exception as e:
            # If the SDK rejects unknown fields (e.g., seed in GenerationConfig), retry without extras
            msg = str(e).lower()
            if "unknown field" in msg or "generationconfig" in msg:
                try:
                    # strip any non-standard keys just in case
                    generation_config = {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_output_tokens": max_tokens,
                    }
                except Exception:
                    pass
            last_err = e
            continue
    raise LLMError(f"gemini error: {last_err}")


def _gen_kimi(model: str, prompt: str, *, system: Optional[str], max_tokens: int, temperature: float, top_p: float, seed: Optional[int]) -> str:
    """Kimi (Moonshot) client.

    - Defaults to the global API base unless KIMI_BASE_URL is set.
    - Uses Chat Completions for moonshot-v1-* models.
    - Uses Responses API for K2 models (e.g., kimi-k2-thinking) per platform docs.
    """
    api_key = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise LLMError("KIMI_API_KEY (or MOONSHOT_API_KEY) not set")
    base_url = os.getenv("KIMI_BASE_URL") or "https://api.moonshot.ai/v1"
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    use_responses = model.lower().startswith("kimi-k2") or ("k2" in model.lower() and not model.lower().startswith("moonshot-v1"))
    delays = [0.5, 1.5, 3.0]
    last_err: Exception | None = None

    if not use_responses:
        # Chat Completions path (moonshot-v1-*)
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        for attempt, delay in enumerate([0.0] + delays):
            if delay:
                time.sleep(delay)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or ""
                return content.strip()
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "insufficient_quota" in msg or "invalid api key" in msg or "invalid authentication" in msg:
                    break
                continue
        raise LLMError(f"kimi error: {last_err}")

    # Responses API path for K2 models
    def _to_input(system_text: Optional[str], user_text: str):
        parts = []
        if system_text:
            parts.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
        parts.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        return parts

    input_payload = _to_input(system, prompt)

    for attempt, delay in enumerate([0.0] + delays):
        if delay:
            time.sleep(delay)
        try:
            resp = client.responses.create(
                model=model,
                input=input_payload,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
            )
            # Prefer SDK convenience attr if present
            text = getattr(resp, "output_text", None)
            if not text:
                # Fallback: concatenate any text parts in output/messages
                try:
                    chunks = []
                    output = getattr(resp, "output", None) or []
                    for item in output:
                        msg = getattr(item, "content", None) or []
                        for c in msg:
                            if getattr(c, "type", None) == "output_text" and hasattr(c, "text"):
                                chunks.append(c.text)
                            elif getattr(c, "type", None) == "text" and hasattr(c, "text"):
                                chunks.append(c.text)
                    text = "\n".join(chunks)
                except Exception:
                    text = None
            return (text or "").strip()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "invalid api key" in msg or "invalid authentication" in msg:
                break
            # Some regions may not expose /responses; fall back to chat.completions
            if "404" in msg or "url.not_found" in msg or "/v1/responses" in msg:
                try:
                    msgs = []
                    if system:
                        msgs.append({"role": "system", "content": system})
                    msgs.append({"role": "user", "content": prompt})
                    cc = client.chat.completions.create(
                        model=model,
                        messages=msgs,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    )
                    content = cc.choices[0].message.content or ""
                    return content.strip()
                except Exception as e2:
                    last_err = e2
                    # Continue retry loop
                    continue
            continue
    raise LLMError(f"kimi error: {last_err}")
