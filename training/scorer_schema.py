from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScorerPairRow:
    pair_id: str
    split: str
    passage1_id: str
    passage2_id: str
    text1: str
    text2: str
    style_text1: str
    style_text2: str
    label: int
    pair_role: str
    neg_type: str | None
    book1: str
    book2: str
    author1: str
    author2: str
    content_cluster1: int | None
    content_cluster2: int | None
    same_author: bool
    same_book: bool
    same_content_cluster: bool | None
