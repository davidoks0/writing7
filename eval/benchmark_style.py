import json
import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm_clients import generate, LLMError
from .topics import DEFAULT_TOPICS


def _split_sentences_simple(text: str) -> List[str]:
    import re
    sentences = []
    pattern = r'[.!?]+["\'\)]*\s+(?=[A-Z])'
    parts = re.split(pattern, text)
    for part in parts:
        part = part.strip()
        if len(part) > 20:
            sentences.append(part)
    if not sentences and len(text.strip()) > 20:
        sentences.append(text.strip())
    return sentences


def _pick_excerpt(text: str, *, n_sentences: int = 15, rng: random.Random | None = None) -> tuple[str, Dict]:
    rng = rng or random
    sents = _split_sentences_simple(text)
    if not sents:
        excerpt = text.strip()
        meta = {"start_idx": 0, "sentences": 1, "total_sentences": 0}
        return excerpt, meta
    if len(sents) <= n_sentences:
        excerpt = " ".join(sents)
        meta = {"start_idx": 0, "sentences": len(sents), "total_sentences": len(sents)}
        return excerpt, meta
    start = rng.randint(0, len(sents) - n_sentences)
    excerpt = " ".join(sents[start:start + n_sentences]).strip()
    meta = {"start_idx": start, "sentences": n_sentences, "total_sentences": len(sents)}
    return excerpt, meta


def _default_system_prompt() -> str:
    return (
        "You are a creative writer. Adopt the narrative voice, pacing, cadence, and paragraphing style of the provided reference, "
        "but produce original text that does not copy phrases or named entities."
    )


def _make_prompt(excerpt: str, topic: str, *, target_words: tuple[int, int] = (600, 900)) -> str:
    lo, hi = target_words
    return (
        f"STYLE REFERENCE\n---\n{excerpt}\n---\n\n"
        f"Write an original short story about: {topic}.\n"
        f"Match the reference's voice, rhythm, and paragraph structure.\n"
        f"Avoid copying plot points, named entities, or distinctive phrases.\n"
        f"Target length: {lo}-{hi} words."
    )


def run_benchmark(
    *,
    model: str,
    book_path: str,
    topics: Optional[List[str]] = None,
    fixed_topic: Optional[str] = None,
    n_samples: int = 3,
    n_excerpts: int = 10,
    seed: int = 42,
    # Style similarity params
    model_dir: str = "/vol/models/book_matcher_contrastive/final",
    num_chunks: str | int = 'auto',
    chunk_size: int = 14,
    overlap: int = 4,
    aggregate: str = "mean",
    topk: int = 5,
    max_length: int = 512,
    # LLM params
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_tokens: int = 1200,
    concurrency: int = 3,
    stream_print: bool = True,
) -> Dict:
    """Run the style benchmark end-to-end and return results as a dict.

    Scoring runs locally (expects GPU environment) using the contrastive matcher.
    """
    # Read book
    if not os.path.exists(book_path):
        raise FileNotFoundError(f"Book not found: {book_path}")
    with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    rng = random.Random(seed)

    # Topics
    topics = topics or list(DEFAULT_TOPICS)

    # Prepare scorer (local, matching modal_app.py::style_similarity_remote_gpu implementation)
    from inference_contrastive import ContrastiveBookMatcherInference
    matcher = ContrastiveBookMatcherInference(model_dir)

    def score_pair(t1: str, t2: str) -> Dict:
        res = matcher.style_similarity(
            t1,
            t2,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            overlap=overlap,
            aggregate=aggregate,
            topk=topk,
            max_length=max_length,
        )
        cos = float(res.get("cosine", float("nan")))
        mapped = (cos + 1.0) / 2.0 if math.isfinite(cos) else float("nan")
        return {
            "cosine": cos,
            "score_0_1": mapped,
            "score_calibrated": (float(res.get("calibrated")) if res.get("calibrated") is not None else None),
            "aggregate": res.get("aggregate"),
            "pairs": int(res.get("pairs", 0)),
        }

    system = _default_system_prompt()
    samples: List[Dict] = []
    excerpts_info: List[Dict] = []
    n_excerpts = int(n_excerpts)
    n_samples = int(n_samples)
    concurrency = max(1, int(concurrency))

    # 1) Build all generation tasks (excerpt/topic/seed/prompt)
    gen_tasks: List[Dict] = []
    for ei in range(max(1, n_excerpts)):
        excerpt, meta = _pick_excerpt(raw, n_sentences=15, rng=rng)
        excerpts_info.append({"text": excerpt, **meta, "index": ei})
        for _ in range(n_samples):
            topic = fixed_topic or rng.choice(topics)
            prompt = _make_prompt(excerpt, topic)
            gen_tasks.append({
                "excerpt_index": ei,
                "excerpt": excerpt,
                "topic": topic,
                "prompt": prompt,
                "seed": rng.randint(0, 2**31 - 1),
            })

    # 2) Generate in parallel (thread pool)
    def _do_generate(task: Dict) -> Dict:
        try:
            out = generate(
                model=model,
                prompt=task["prompt"],
                system=system,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=task["seed"],
            )
            ok = True
        except LLMError as e:
            out = f"[LLM ERROR] {e}"
            ok = False
        return {**task, "output": out, "ok": ok}

    gen_results: List[Dict] = []
    if concurrency == 1:
        for t in gen_tasks:
            r = _do_generate(t)
            gen_results.append(r)
            if stream_print:
                try:
                    topic_i = r.get("topic")
                    out = r.get("output", "")
                    print(f"=== LLM OUTPUT {len(gen_results)}/{len(gen_tasks)} (topic={topic_i}) ===")
                    if isinstance(out, str):
                        print(out)
                    else:
                        print(str(out))
                    print(f"=== END LLM OUTPUT {len(gen_results)}/{len(gen_tasks)} ===")
                except Exception:
                    pass
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_do_generate, t): t for t in gen_tasks}
            done_count = 0
            total = len(gen_tasks)
            for fut in as_completed(futures):
                r = fut.result()
                gen_results.append(r)
                if stream_print:
                    done_count += 1
                    try:
                        topic_i = r.get("topic")
                        out = r.get("output", "")
                        print(f"=== LLM OUTPUT {done_count}/{total} (topic={topic_i}) ===")
                        if isinstance(out, str):
                            print(out)
                        else:
                            print(str(out))
                        print(f"=== END LLM OUTPUT {done_count}/{total} ===")
                    except Exception:
                        pass

    # 3) Score sequentially (GPU-bound)
    for r in gen_results:
        excerpt = r["excerpt"]
        if not r["ok"]:
            scores = {
                "cosine": None,
                "score_0_1": None,
                "score_calibrated": None,
                "aggregate": "error",
                "pairs": 0,
            }
        else:
            scores = score_pair(excerpt, r["output"])
        samples.append({
            "excerpt_index": r["excerpt_index"],
            "topic": r["topic"],
            "prompt": r["prompt"],
            "output": r["output"],
            "scores": scores,
        })

    # Aggregate
    def _collect(key: str):
        vals = []
        for s in samples:
            v = s["scores"].get(key)
            if v is None:
                continue
            try:
                v = float(v)
                if math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
        return vals

    cos = _collect("cosine")
    s01 = _collect("score_0_1")
    scal = _collect("score_calibrated")
    import statistics as stats
    agg = {
        "mean_cosine": (stats.fmean(cos) if cos else None),
        "mean_score_0_1": (stats.fmean(s01) if s01 else None),
        "mean_score_calibrated": (stats.fmean(scal) if scal else None),
        "median_cosine": (stats.median(cos) if cos else None),
        "median_score_0_1": (stats.median(s01) if s01 else None),
        "median_score_calibrated": (stats.median(scal) if scal else None),
        "n": len(samples),
    }

    return {
        "model": model,
        "book": book_path,
        "seed": seed,
        # Keep first excerpt for backward compatibility
        "excerpt": (excerpts_info[0] if excerpts_info else None),
        "excerpts": excerpts_info,
        "n_excerpts": n_excerpts,
        "n_samples_per_excerpt": n_samples,
        "params": {
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "aggregate": aggregate,
            "topk": topk,
            "max_length": max_length,
        },
        "samples": samples,
        "aggregate": agg,
    }
