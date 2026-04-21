from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from eval.benchmark_io import load_json
from training.style_text import (
    RAW_TEXT_VIEW,
    STYLE_MASKED_TEXT_VIEW,
    apply_text_view,
    normalize_blend_weights,
    normalize_score_text_views,
    normalize_text_view,
)
from training.transformer_style_model import load_transformer_artifact, require_transformer_dependencies, torch


TOKEN_RE = re.compile(r"[a-z0-9']+")
WORD_RE = re.compile(r"\S+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if not first or not second:
        return 0.0
    dot = sum(left * right for left, right in zip(first, second))
    norm_first = math.sqrt(sum(value * value for value in first))
    norm_second = math.sqrt(sum(value * value for value in second))
    if norm_first == 0.0 or norm_second == 0.0:
        return 0.0
    return dot / (norm_first * norm_second)


def mean_pool(vectors: Iterable[list[float]]) -> list[float]:
    collected = list(vectors)
    if not collected:
        return []
    width = len(collected[0])
    pooled = [0.0] * width
    for vector in collected:
        for index, value in enumerate(vector):
            pooled[index] += value
    pooled = [value / len(collected) for value in pooled]
    norm = math.sqrt(sum(value * value for value in pooled))
    if norm == 0.0:
        return pooled
    return [value / norm for value in pooled]


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def percentile_of(score: float, distribution: dict[str, Any] | None) -> float:
    if not distribution:
        return 0.5
    raw_values = distribution.get("raw_values")
    if raw_values:
        count = sum(1 for value in raw_values if value <= score)
        return count / max(1, len(raw_values))
    quantiles = [
        ("p1", 0.01),
        ("p5", 0.05),
        ("p10", 0.10),
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p95", 0.95),
        ("p99", 0.99),
    ]
    previous_score = distribution.get("p1", distribution.get("mean", score))
    previous_pct = 0.01
    if score <= previous_score:
        return previous_pct
    for name, pct in quantiles[1:]:
        current_score = distribution.get(name, previous_score)
        if score <= current_score:
            span = max(1e-9, current_score - previous_score)
            position = (score - previous_score) / span
            return previous_pct + ((pct - previous_pct) * position)
        previous_score = current_score
        previous_pct = pct
    return 1.0


@dataclass
class StyleScorerConfig:
    model_type: str
    vocabulary: list[str] | None = None
    idf: list[float] | None = None
    hashing_dim: int = 512
    max_features: int = 4096
    base_encoder: str | None = None
    pooling: str | None = None
    use_projection: bool | None = None
    projection_dim: int | None = None
    max_length: int | None = None
    text_view: str = RAW_TEXT_VIEW
    score_text_views: list[str] | None = None
    blend_weights: dict[str, float] | None = None
    chunk_size_words: int | None = None
    chunk_overlap_words: int | None = None
    chunk_aggregation: str = "mean"
    chunk_top_k: int | None = None


class StyleScorer:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        config_payload = load_json(self.model_dir / "config.json")
        self.manifest = load_json(self.model_dir / "scorer_manifest.json") if (self.model_dir / "scorer_manifest.json").exists() else {}
        chunk_manifest = self.manifest.get("chunking", {})
        default_text_view = normalize_text_view(config_payload.get("text_view"))
        score_text_views = normalize_score_text_views(
            config_payload.get("score_text_views") or self.manifest.get("score_text_views"),
            default=default_text_view,
        )
        blend_weights = normalize_blend_weights(
            config_payload.get("blend_weights") or self.manifest.get("blend_weights"),
            score_text_views=score_text_views,
        )
        self.config = StyleScorerConfig(
            model_type=config_payload.get("model_type", "bag_of_words_style_scorer_v1"),
            vocabulary=config_payload.get("vocabulary"),
            idf=config_payload.get("idf"),
            hashing_dim=config_payload.get("hashing_dim", 512),
            max_features=config_payload.get("max_features", 4096),
            base_encoder=config_payload.get("base_encoder"),
            pooling=config_payload.get("pooling"),
            use_projection=config_payload.get("use_projection"),
            projection_dim=config_payload.get("projection_dim"),
            max_length=config_payload.get("max_length"),
            text_view=default_text_view,
            score_text_views=score_text_views,
            blend_weights=blend_weights,
            chunk_size_words=config_payload.get("chunk_size_words", chunk_manifest.get("chunk_size_words")),
            chunk_overlap_words=config_payload.get("chunk_overlap_words", chunk_manifest.get("chunk_overlap_words")),
            chunk_aggregation=config_payload.get("chunk_aggregation", chunk_manifest.get("chunk_aggregation", "mean")),
            chunk_top_k=config_payload.get("chunk_top_k", chunk_manifest.get("chunk_top_k")),
        )
        calibration_path = self._resolve_calibration_path()
        if calibration_path.exists():
            calibration_payload = load_json(calibration_path)
            style_calibration = calibration_payload.get("style_calibration", {})
            selection = calibration_payload.get("selection", {})
            chosen_method = selection.get("chosen") or style_calibration.get("method") or "logistic"
            if chosen_method == "identity":
                self.calibration_coef = None
                self.calibration_intercept = None
            else:
                self.calibration_coef = float(style_calibration.get("coef", 1.0))
                self.calibration_intercept = float(style_calibration.get("intercept", 0.0))
        else:
            self.calibration_coef = None
            self.calibration_intercept = None
        self._transformer_model = None
        self._transformer_tokenizer = None
        self._transformer_device = None
        self.vocab_index = {token: index for index, token in enumerate(self.config.vocabulary or [])}
        self.idf = self.config.idf or [1.0] * len(self.vocab_index)

    def _resolve_calibration_path(self) -> Path:
        local_path = self.model_dir / "style_calibration_v1.json"
        if local_path.exists():
            return local_path
        return self.model_dir.parent / "style_calibration_v1.json"

    def _ensure_transformer(self) -> None:
        if self._transformer_model is not None:
            return
        require_transformer_dependencies()
        model, tokenizer, _ = load_transformer_artifact(self.model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        self._transformer_model = model
        self._transformer_tokenizer = tokenizer
        self._transformer_device = device

    def _embed_bow(self, text: str, *, text_view: str) -> list[float]:
        rendered = apply_text_view(text, text_view)
        tokens = tokenize(rendered)
        if self.vocab_index:
            vector = [0.0] * len(self.vocab_index)
            token_counts: dict[str, int] = {}
            for token in tokens:
                if token in self.vocab_index:
                    token_counts[token] = token_counts.get(token, 0) + 1
            for token, count in token_counts.items():
                index = self.vocab_index[token]
                vector[index] = count * self.idf[index]
        else:
            vector = [0.0] * self.config.hashing_dim
            for token in tokens:
                bucket = hash(token) % self.config.hashing_dim
                vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def _embed_transformer(self, text: str, *, text_view: str) -> list[float]:
        rendered = apply_text_view(text, text_view)
        self._ensure_transformer()
        tokenizer = self._transformer_tokenizer
        model = self._transformer_model
        device = self._transformer_device
        batch = tokenizer(
            [rendered],
            padding=True,
            truncation=True,
            max_length=int(self.config.max_length or 384),
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            embedding = model.encode(batch["input_ids"], batch["attention_mask"])[0]
        return embedding.detach().cpu().tolist()

    def embed(self, text: str, text_view: str | None = None) -> list[float]:
        view = normalize_text_view(text_view or self.config.text_view)
        if self.config.model_type == "hf_transformer_contrastive_v1":
            return self._embed_transformer(text, text_view=view)
        return self._embed_bow(text, text_view=view)

    def _chunk_text(self, text: str) -> list[str]:
        chunk_size = int(self.config.chunk_size_words or 0)
        if chunk_size <= 0:
            return [text]
        words = WORD_RE.findall(text)
        if len(words) <= chunk_size:
            return [text]
        overlap = int(self.config.chunk_overlap_words or 0)
        overlap = max(0, min(overlap, chunk_size - 1))
        step = max(1, chunk_size - overlap)
        chunks = []
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            if not window:
                continue
            chunks.append(" ".join(window))
            if start + chunk_size >= len(words):
                break
        return chunks or [text]

    def primary_score(self, cosine: float) -> float:
        score_0_1 = (cosine + 1.0) / 2.0
        if self.calibration_coef is None or self.calibration_intercept is None:
            return score_0_1
        return sigmoid((self.calibration_coef * score_0_1) + self.calibration_intercept)

    def _aggregate_chunk_scores(self, values: list[float]) -> tuple[float, str]:
        if not values:
            return 0.0, "mean"
        aggregation = (self.config.chunk_aggregation or "mean").lower()
        if aggregation in {"mean", "average"}:
            return sum(values) / len(values), "mean"
        if aggregation in {"median"}:
            ordered = sorted(values)
            middle = len(ordered) // 2
            if len(ordered) % 2 == 1:
                return ordered[middle], "median"
            return (ordered[middle - 1] + ordered[middle]) / 2.0, "median"
        if aggregation in {"trimmed_mean", "trim_mean"}:
            ordered = sorted(values)
            trim = int(len(ordered) * 0.1)
            trimmed = ordered[trim : len(ordered) - trim] if len(ordered) > 2 * trim else ordered
            return sum(trimmed) / max(1, len(trimmed)), "trimmed_mean"
        if aggregation in {"topk_mean", "top_k_mean"}:
            top_k = int(self.config.chunk_top_k or max(1, math.ceil(len(values) / 2)))
            selected = sorted(values, reverse=True)[: max(1, min(top_k, len(values)))]
            return sum(selected) / len(selected), f"top{len(selected)}_mean"
        return sum(values) / len(values), aggregation

    def _single_view_score(self, text1: str, text2: str, *, text_view: str) -> dict[str, float | int | str]:
        hypothesis_chunks = self._chunk_text(text1)
        reference_chunks = self._chunk_text(text2)
        hypothesis_embeddings = [self.embed(chunk, text_view=text_view) for chunk in hypothesis_chunks]
        reference_embeddings = [self.embed(chunk, text_view=text_view) for chunk in reference_chunks]
        if not hypothesis_embeddings or not reference_embeddings:
            cosine = 0.0
            aggregate = "mean"
        elif len(hypothesis_embeddings) == 1 and len(reference_embeddings) == 1:
            cosine = cosine_similarity(hypothesis_embeddings[0], reference_embeddings[0])
            aggregate = "mean"
        else:
            best_matches = [
                max(cosine_similarity(hypothesis_embedding, reference_embedding) for reference_embedding in reference_embeddings)
                for hypothesis_embedding in hypothesis_embeddings
            ]
            cosine, aggregate = self._aggregate_chunk_scores(best_matches)
        score_0_1 = (cosine + 1.0) / 2.0
        payload: dict[str, float | int | str] = {
            "cosine": round(cosine, 6),
            "score_0_1": round(score_0_1, 6),
            "pairs": len(hypothesis_embeddings),
            "reference_chunks": len(reference_embeddings),
            "aggregate": aggregate,
        }
        if self.calibration_coef is not None and self.calibration_intercept is not None:
            payload["calibrated"] = round(self.primary_score(cosine), 6)
        return payload

    def score_pair(self, text1: str, text2: str) -> dict[str, float | int | str | dict[str, Any]]:
        score_text_views = self.config.score_text_views or [self.config.text_view]
        blend_weights = self.config.blend_weights or {self.config.text_view: 1.0}
        view_scores = {
            view: self._single_view_score(text1, text2, text_view=view)
            for view in score_text_views
        }
        blended_cosine = sum(float(view_scores[view]["cosine"]) * blend_weights.get(view, 0.0) for view in score_text_views)
        score_0_1 = (blended_cosine + 1.0) / 2.0
        pairs = max((int(view_scores[view].get("pairs", 1)) for view in score_text_views), default=1)
        payload: dict[str, float | int | str | dict[str, Any]] = {
            "cosine": round(blended_cosine, 6),
            "score_0_1": round(score_0_1, 6),
            "pairs": pairs,
            "aggregate": "blended_multi_view" if len(score_text_views) > 1 else str(view_scores[score_text_views[0]]["aggregate"]),
            "blended_similarity": round(score_0_1, 6),
            "view_scores": {
                view: {
                    key: value
                    for key, value in score_payload.items()
                    if key != "calibrated"
                }
                for view, score_payload in view_scores.items()
            },
        }
        if RAW_TEXT_VIEW in view_scores:
            payload["raw_similarity"] = float(view_scores[RAW_TEXT_VIEW]["score_0_1"])
        if STYLE_MASKED_TEXT_VIEW in view_scores:
            payload["masked_similarity"] = float(view_scores[STYLE_MASKED_TEXT_VIEW]["score_0_1"])
        if self.calibration_coef is not None and self.calibration_intercept is not None:
            payload["calibrated"] = round(self.primary_score(blended_cosine), 6)
        return payload

    def score_many(self, hypothesis: str, references: list[str]) -> list[dict[str, float | int | str | dict[str, Any]]]:
        return [self.score_pair(hypothesis, reference) for reference in references]


def group_similarity_details(scorer: StyleScorer, hypothesis: str, references: list[str]) -> dict[str, float]:
    if not references:
        return {"value": 0.0}
    scores = scorer.score_many(hypothesis, references)
    values = [float(row.get("calibrated", row["score_0_1"])) for row in scores]
    details = {"value": sum(values) / max(1, len(values))}
    if all("raw_similarity" in row for row in scores):
        details["raw_similarity"] = sum(float(row["raw_similarity"]) for row in scores) / max(1, len(scores))
    if all("masked_similarity" in row for row in scores):
        details["masked_similarity"] = sum(float(row["masked_similarity"]) for row in scores) / max(1, len(scores))
    if all("blended_similarity" in row for row in scores):
        details["blended_similarity"] = sum(float(row["blended_similarity"]) for row in scores) / max(1, len(scores))
    return details


def group_similarity(scorer: StyleScorer, hypothesis: str, references: list[str]) -> float:
    return group_similarity_details(scorer, hypothesis, references)["value"]


def compute_style_metrics(
    scorer: StyleScorer,
    output_text: str,
    target_references: list[str],
    distractor_references: dict[str, list[str]],
    reference_distribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not distractor_references:
        raise ValueError("style scoring requires at least one distractor target")
    target_details = group_similarity_details(scorer, output_text, target_references)
    target_similarity = float(target_details["value"])
    distractor_details = {
        target_id: group_similarity_details(scorer, output_text, references)
        for target_id, references in distractor_references.items()
    }
    distractor_similarity_means = {
        target_id: float(details["value"])
        for target_id, details in distractor_details.items()
    }
    distractor_values = list(distractor_similarity_means.values())
    if distractor_values:
        wins = [
            1.0 if target_similarity > value else 0.5 if target_similarity == value else 0.0
            for value in distractor_values
        ]
        style_win_rate = sum(wins) / len(wins)
        mean_distractor = sum(distractor_values) / len(distractor_values)
    else:
        style_win_rate = 1.0
        mean_distractor = 0.0
    rank = 1 + sum(1 for value in distractor_values if value > target_similarity)
    top1 = 1 if rank == 1 else 0
    percentile = percentile_of(target_similarity, (reference_distribution or {}).get("global", {}).get("target_similarity"))
    payload = {
        "target_similarity_mean": round(target_similarity, 6),
        "distractor_similarity_means": {key: round(value, 6) for key, value in distractor_similarity_means.items()},
        "style_win_rate_case": round(style_win_rate, 6),
        "style_margin_case": round(target_similarity - mean_distractor, 6),
        "top1_target_case": top1,
        "rank_target_case": rank,
        "mrr_case": round(1.0 / rank, 6),
        "style_percentile_case": round(percentile, 6),
    }
    for metric_name in ("raw_similarity", "masked_similarity", "blended_similarity"):
        if metric_name in target_details:
            payload[f"target_{metric_name}_mean"] = round(float(target_details[metric_name]), 6)
    return payload
