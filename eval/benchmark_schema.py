from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


ASCII_ID_RE = re.compile(r"^[a-z0-9:_\.]+$")


def _validate_identifier(name: str, value: str, prefix: str | None = None) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if prefix and not value.startswith(prefix):
        raise ValueError(f"{name} must start with {prefix!r}")
    if not ASCII_ID_RE.fullmatch(value):
        raise ValueError(f"{name} must be lowercase ASCII and stable: {value!r}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass
class InputBookRecord:
    book_id: str
    author_id: str
    title: str
    author: str
    source_path: str
    gutenberg_id: str | None = None
    language: str | None = None
    publication_year: int | None = None
    period_bucket: str = "unknown"
    genre: str = "unknown"
    is_translation: bool = False
    subjects: list[str] = field(default_factory=list)
    bookshelves: list[str] = field(default_factory=list)
    source_type: str | None = None

    def validate(self) -> "InputBookRecord":
        _validate_identifier("book_id", self.book_id, "book:")
        _validate_identifier("author_id", self.author_id, "author:")
        _require(bool(self.title.strip()), "title must be non-empty")
        _require(bool(self.author.strip()), "author must be non-empty")
        _require(bool(self.source_path.strip()), "source_path must be non-empty")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InputBookRecord":
        return cls(**payload).validate()


@dataclass
class CleanBookRecord:
    book_id: str
    author_id: str
    title: str
    author: str
    source_path: str
    clean_path: str
    gutenberg_id: str | None = None
    language: str | None = None
    publication_year: int | None = None
    period_bucket: str = "unknown"
    genre: str = "unknown"
    is_translation: bool = False
    subjects: list[str] = field(default_factory=list)
    bookshelves: list[str] = field(default_factory=list)
    source_type: str | None = None
    clean_word_count: int = 0
    clean_char_count: int = 0
    clean_sentence_count: int = 0
    eligible_author_track: bool = False
    eligible_book_track: bool = False
    exclusion_reasons: list[str] = field(default_factory=list)

    def validate(self) -> "CleanBookRecord":
        _validate_identifier("book_id", self.book_id, "book:")
        _validate_identifier("author_id", self.author_id, "author:")
        _require(bool(self.clean_path.strip()), "clean_path must be non-empty")
        _require(self.clean_word_count >= 0, "clean_word_count must be non-negative")
        _require(self.clean_char_count >= 0, "clean_char_count must be non-negative")
        _require(self.clean_sentence_count >= 0, "clean_sentence_count must be non-negative")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CleanBookRecord":
        return cls(**payload).validate()


@dataclass
class PassageRecord:
    passage_id: str
    book_id: str
    author_id: str
    track: str | None = None
    role_pool: str | None = None
    text: str = ""
    start_sentence: int = 0
    end_sentence: int = 0
    start_char: int = 0
    end_char: int = 0
    word_count: int = 0
    char_count: int = 0
    region_bucket: int = 0
    text_sha1: str | None = None

    def validate(self) -> "PassageRecord":
        _validate_identifier("passage_id", self.passage_id, "passage:")
        _validate_identifier("book_id", self.book_id, "book:")
        _validate_identifier("author_id", self.author_id, "author:")
        _require(self.end_sentence > self.start_sentence, "end_sentence must be greater than start_sentence")
        _require(self.end_char >= self.start_char, "end_char must be >= start_char")
        _require(bool(self.text.strip()), "text must be non-empty")
        _require(self.region_bucket >= 0, "region_bucket must be non-negative")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PassageRecord":
        return cls(**payload).validate()


@dataclass
class PromptRecord:
    prompt_id: str
    family: str
    text: str
    required_keywords: list[str] = field(default_factory=list)
    preferred_pov: str | None = None
    dialogue_expected: bool | None = None
    target_word_range: list[int] | None = None

    def validate(self) -> "PromptRecord":
        _validate_identifier("prompt_id", self.prompt_id, "prompt:")
        _require(bool(self.family.strip()), "family must be non-empty")
        _require(bool(self.text.strip()), "text must be non-empty")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptRecord":
        return cls(**payload).validate()


@dataclass
class BenchmarkTarget:
    target_id: str
    track: str
    author_id: str
    conditioning_book_ids: list[str] | None = None
    evaluation_book_id: str | None = None
    book_id: str | None = None

    def validate(self) -> "BenchmarkTarget":
        _validate_identifier("target_id", self.target_id)
        _validate_identifier("author_id", self.author_id, "author:")
        _require(self.track in {"author", "book"}, "track must be 'author' or 'book'")
        if self.track == "author":
            _validate_identifier("target_id", self.target_id, "author:")
            _require(bool(self.conditioning_book_ids), "author targets require conditioning_book_ids")
            _require(len(self.conditioning_book_ids or []) >= 2, "author targets need at least 2 conditioning books")
            _require(self.evaluation_book_id is not None, "author targets require evaluation_book_id")
        else:
            _validate_identifier("target_id", self.target_id, "book:")
            _require(self.book_id is not None, "book targets require book_id")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkTarget":
        return cls(**payload).validate()


@dataclass
class BenchmarkCase:
    case_id: str
    benchmark_version: str
    track: str
    split: str
    target_id: str
    prompt_id: str
    conditioning_passage_ids: list[str]
    evaluation_passage_ids: list[str]
    distractor_target_ids: list[str]
    distractor_passage_ids_by_target: dict[str, list[str]]
    generation_profile_id: str
    sample_seeds: list[int]

    def validate(self) -> "BenchmarkCase":
        _validate_identifier("case_id", self.case_id, "case:")
        _validate_identifier("target_id", self.target_id)
        _validate_identifier("prompt_id", self.prompt_id, "prompt:")
        _require(self.track in {"author", "book"}, "track must be 'author' or 'book'")
        _require(self.split in {"dev", "test"}, "split must be 'dev' or 'test'")
        _require(bool(self.benchmark_version.strip()), "benchmark_version must be non-empty")
        _require(len(self.conditioning_passage_ids) == 3, "conditioning_passage_ids must contain exactly 3 passages")
        _require(len(self.evaluation_passage_ids) == 4, "evaluation_passage_ids must contain exactly 4 passages")
        _require(bool(self.distractor_target_ids), "distractor_target_ids must be non-empty")
        _require(bool(self.sample_seeds), "sample_seeds must be non-empty")
        for passage_id in self.conditioning_passage_ids + self.evaluation_passage_ids:
            _validate_identifier("passage_id", passage_id, "passage:")
        _require(
            set(self.distractor_target_ids) == set(self.distractor_passage_ids_by_target),
            "distractor_passage_ids_by_target must match distractor_target_ids exactly",
        )
        for distractor_id in self.distractor_target_ids:
            _validate_identifier("distractor_target_id", distractor_id)
            distractor_passage_ids = self.distractor_passage_ids_by_target.get(distractor_id, [])
            _require(
                len(distractor_passage_ids) == 4,
                f"distractor target {distractor_id!r} must contain exactly 4 evaluation passages",
            )
            for passage_id in distractor_passage_ids:
                _validate_identifier("passage_id", passage_id, "passage:")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkCase":
        return cls(**payload).validate()


def validate_case_payload(payload: dict[str, Any]) -> BenchmarkCase:
    return BenchmarkCase.from_dict(payload)
