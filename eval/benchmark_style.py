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
        "You are a creative writer. Adopt the style style of the provided reference text, "
        "but produce original text that does NOT copy phrases or named entities from the reference text."
    )


def _make_prompt(excerpt: str, topic: str, *, target_words: tuple[int, int] = (600, 900)) -> str:
    lo, hi = target_words
    return (
        f"STYLE REFERENCE\n---\n{excerpt}\n---\n\n"
        f"Write an original short story about: {topic}.\n"
        f"Match the reference's voice, rhythm, and paragraph structure.\n"
        f"Avoid copying plot points, named entities, or distinctive phrases. Avoid explicit violence, gore, and graphic harm. Keep content safe and non-graphic.\n"
        f"Target length: {lo}-{hi} words."
    )


def run_benchmark(
    *,
    model: str,
    book_path: str,
    topics: Optional[List[str]] = None,
    fixed_topic: Optional[str] = None,
    n_samples: int = 1,
    n_excerpts: int = 5,
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
    # Retry policy for empty/too-short generations
    retry_empty: bool = True,
    retry_attempts: int = 2,
    min_chars: int = 50,
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
        # Attempt generation, retrying on empty/too-short outputs if requested
        attempts = max(0, int(retry_attempts)) if retry_empty else 0
        rng_local = random.Random(task.get("seed", seed))
        last_err: Exception | None = None
        out: str = ""
        ok: bool = False
        for i in range(attempts + 1):
            try:
                s = task["seed"] if i == 0 else rng_local.randint(0, 2**31 - 1)
                out_try = generate(
                    model=model,
                    prompt=task["prompt"],
                    system=system,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=s,
                )
                txt = (out_try or "").strip()
                # Accept if long enough; else retry if allowed
                if txt and len(txt) >= int(min_chars):
                    out = txt
                    ok = True
                    break
                out = txt  # keep last attempt even if short
                ok = False
            except LLMError as e:
                last_err = e
                out = f"[LLM ERROR] {e}"
                ok = False
                # On API error, continue to next attempt (may recover)
                continue
        # If we exhausted attempts and still too short or error, return best-effort result
        if not ok and last_err is not None:
            out = f"[LLM ERROR after {attempts+1} attempt(s)] {last_err}"
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
    # Validity: count samples that produced usable outputs (scores not marked as error)
    total_n = len(samples)
    valid_n = 0
    for s in samples:
        try:
            if s.get("scores", {}).get("aggregate") != "error":
                valid_n += 1
        except Exception:
            pass

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
        "n": total_n,
        "valid": valid_n,
        "valid_rate": (float(valid_n) / float(total_n) if total_n else None),
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


if __name__ == "__main__":
    import argparse
    import json as _json
    ap = argparse.ArgumentParser(description="LLM style-match benchmark against a reference book excerpt")
    # Core
    ap.add_argument("--model", required=True, help="Provider-prefixed model, e.g. openai:gpt-4o-mini, anthropic:claude-3-5-sonnet, gemini:1.5-pro, kimi:moonshot-v1-8k")
    ap.add_argument("--book_path", default="eval/books/gatsby.txt", help="Path to reference book .txt file")
    ap.add_argument("--n_excerpts", type=int, default=5, help="How many independent excerpts to sample from the book")
    ap.add_argument("--n_samples", type=int, default=1, help="How many generations per excerpt")
    ap.add_argument("--seed", type=int, default=42, help="Global RNG seed for excerpt/topic selection")
    ap.add_argument("--fixed_topic", type=str, default=None, help="Force all prompts to use this topic instead of random topics")
    # Scorer (local contrastive matcher)
    ap.add_argument("--model_dir", default="/vol/models/book_matcher_contrastive/final", help="Path to contrastive matcher directory containing weights (final/)")
    ap.add_argument("--num_chunks", default="auto", help="Number of chunks per text or 'auto'")
    ap.add_argument("--chunk_size", type=int, default=14, help="Sentences per chunk for style similarity")
    ap.add_argument("--overlap", type=int, default=4, help="Sentence overlap between adjacent chunks")
    ap.add_argument("--aggregate", choices=["mean", "topk_mean"], default="mean", help="Aggregation over pairwise chunk cosines")
    ap.add_argument("--topk", type=int, default=5, help="Top-k for topk_mean aggregation")
    ap.add_argument("--max_length", type=int, default=512, help="Max tokens per chunk for encoder")
    # LLM sampling
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--concurrency", type=int, default=3, help="Parallel generation workers")
    ap.add_argument("--stream_print", type=str, default="true", help="Print generations as they complete (true/false)")
    ap.add_argument("--retry_empty", type=str, default="true", help="Retry empty/too-short outputs (true/false)")
    ap.add_argument("--retry_attempts", type=int, default=2, help="Extra attempts when retrying")
    ap.add_argument("--min_chars", type=int, default=50, help="Minimum characters required to accept a generation")
    # Output
    ap.add_argument("--out", type=str, default=None, help="Optional path to write full JSON results")

    args = ap.parse_args()

    def _to_bool(x: str) -> bool:
        return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}

    # Parse num_chunks (int or 'auto')
    try:
        nc = int(args.num_chunks)
    except Exception:
        nc = "auto"

    results = run_benchmark(
        model=args.model,
        book_path=args.book_path,
        topics=None,
        fixed_topic=args.fixed_topic,
        n_samples=args.n_samples,
        n_excerpts=args.n_excerpts,
        seed=args.seed,
        model_dir=args.model_dir,
        num_chunks=nc,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        aggregate=args.aggregate,
        topk=args.topk,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        stream_print=_to_bool(args.stream_print),
        retry_empty=_to_bool(args.retry_empty),
        retry_attempts=args.retry_attempts,
        min_chars=args.min_chars,
    )

    agg = results.get("aggregate", {})
    print("\n=== Aggregate ===")
    try:
        print(_json.dumps(agg, indent=2, sort_keys=True))
    except Exception:
        print(str(agg))
    print("\nModel:", results.get("model"))
    print("Book:", results.get("book"))
    print("Excerpts:", results.get("n_excerpts"), "Samples/Excerpt:", results.get("n_samples_per_excerpt"))

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                _json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nWrote results to {args.out}")
        except Exception as e:
            print(f"[warn] failed to write --out file: {e}")
