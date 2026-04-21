from __future__ import annotations

from pathlib import Path
from typing import Any

from corpus.layout import resolve_with_root
from corpus.manifests import load_corpus_config
from eval.benchmark_io import load_jsonl, write_jsonl
from eval.passage_sampling import extract_passage_candidates


def build_passage_pools(config_path_or_payload: str | Path | dict[str, Any]) -> dict[str, str]:
    config = load_corpus_config(config_path_or_payload)
    output_root = Path(config.get("output_root", "build/corpus"))
    books = load_jsonl(output_root / "meta" / "books_manifest_v1.jsonl")
    passage_policy = config.get("passage_policy", {})
    outputs: dict[str, str] = {}
    for book in books:
        if not book.get("eligible_book_track", False):
            continue
        clean_path = resolve_with_root(output_root, book["clean_path"])
        cleaned_text = clean_path.read_text(encoding="utf-8")
        passages = extract_passage_candidates(
            cleaned_text,
            book_id=book["book_id"],
            author_id=book["author_id"],
            min_words=passage_policy.get("min_words", 150),
            max_words=passage_policy.get("max_words", 300),
            min_sentences=passage_policy.get("min_sentences", 6),
            max_sentences=passage_policy.get("max_sentences", 18),
            region_buckets=passage_policy.get("region_buckets", 5),
        )
        book_slug = book["book_id"].replace(":", "_")
        destination = output_root / "meta" / "passage_pools_v1" / book["author_slug"] / f"{book_slug}.jsonl"
        write_jsonl(destination, passages)
        outputs[book["book_id"]] = destination.as_posix()
    return outputs
