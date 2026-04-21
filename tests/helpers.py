from __future__ import annotations

import json
from pathlib import Path


STYLE_WORDS = [
    "lantern",
    "vellum",
    "harbor",
    "orchard",
    "copper",
    "parlor",
    "quarry",
    "meadow",
    "galley",
    "cedar",
]


def _make_book_text(author_name: str, style_word: str, title: str, variant: int) -> str:
    sentences = []
    for index in range(28):
        scene = [
            f"{author_name} kept the {style_word} mood of {title} in mind when the afternoon began to lean toward evening.",
            f"The room held a patient hush, and the {style_word} detail on the table seemed to gather every wandering thought into one direction.",
            f"No one in the company spoke carelessly; even ordinary remarks were arranged as though they might later be examined for motive.",
            f"A visitor mentioned the day's errand, but the errand mattered less than the pauses that opened around it.",
            f"By the time the lamps were lit, the house had discovered a new uneasiness and carried it from doorway to doorway.",
            f"Someone laughed in a dutiful way, and that dutiful laugh made the true feeling in the room easier to detect.",
            f"The {style_word} token near the mantelpiece appeared insignificant, yet it altered the whole conversation once it had been noticed.",
        ]
        sentences.append(scene[(index + variant) % len(scene)])
    return " ".join(sentences) + "\n"


def create_synthetic_books_manifest(tmp_path: Path, *, author_count: int = 6, books_per_author: int = 3) -> Path:
    books_root = tmp_path / "books"
    books_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for author_index in range(author_count):
        author_name = f"Author {author_index:02d}"
        author_slug = f"author_{author_index:02d}"
        style_word = STYLE_WORDS[author_index % len(STYLE_WORDS)]
        period_bucket = "1800_1849" if author_index % 2 == 0 else "1850_1899"
        genre = "novel" if author_index % 2 == 0 else "adventure"
        for book_index in range(books_per_author):
            title = f"Book {book_index:02d} of {author_name}"
            title_slug = f"book_{book_index:02d}_of_author_{author_index:02d}"
            path = books_root / f"{author_slug}_{title_slug}.txt"
            path.write_text(_make_book_text(author_name, style_word, title, book_index), encoding="utf-8")
            rows.append(
                {
                    "book_id": f"book:{author_slug}:{title_slug}",
                    "author_id": f"author:{author_slug}",
                    "title": title,
                    "author": author_name,
                    "source_path": path.as_posix(),
                    "gutenberg_id": f"synthetic_{author_index:02d}_{book_index:02d}",
                    "language": "en",
                    "publication_year": 1840 + author_index + book_index,
                    "period_bucket": period_bucket,
                    "genre": genre,
                    "is_translation": False,
                }
            )
    manifest_path = tmp_path / "books_input.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return manifest_path

