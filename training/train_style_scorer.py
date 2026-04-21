from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.benchmark_io import load_json, load_jsonl, write_json
from eval.style_scoring import StyleScorer
from training.diagnostics import build_style_diagnostics
from training.transformer_style_model import (
    AutoTokenizer,
    TransformerStyleEncoder,
    build_model_config,
    require_transformer_dependencies,
    save_transformer_artifact,
    torch,
)
from training.style_text import (
    RAW_TEXT_VIEW,
    STYLE_MASKED_TEXT_VIEW,
    apply_text_view,
    normalize_blend_weights,
    normalize_score_text_views,
    normalize_text_view,
)


def _load_config(path_or_payload: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_payload, dict):
        return dict(path_or_payload)
    return load_json(path_or_payload)


def _utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _select_pair_text(row: dict[str, Any], field_index: int, text_view: str) -> str:
    normalized_text_view = normalize_text_view(text_view)
    raw_key = f"text{field_index}"
    if normalized_text_view == RAW_TEXT_VIEW:
        return row[raw_key]
    style_key = f"style_text{field_index}"
    if style_key in row and row[style_key]:
        return row[style_key]
    return apply_text_view(row[raw_key], normalized_text_view)


def _resolve_scoring_config(config: dict[str, Any]) -> tuple[list[str], dict[str, float], str]:
    training_text_view = normalize_text_view(config.get("text_view"))
    score_text_views = normalize_score_text_views(config.get("score_text_views"), default=training_text_view)
    blend_weights = normalize_blend_weights(config.get("blend_weights"), score_text_views=score_text_views)
    return score_text_views, blend_weights, training_text_view


def _train_bow_scorer(config: dict[str, Any]) -> dict[str, str]:
    import math
    import re
    from collections import Counter

    TOKEN_RE = re.compile(r"[a-z0-9']+")

    def _tokenize(text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    artifacts_root = Path(config.get("artifacts_root", "build/artifacts"))
    dataset_root = artifacts_root / "scorer" / "datasets"
    final_root = artifacts_root / "scorer" / "final"
    final_root.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(dataset_root / "train_pairs_v1.jsonl")
    test_rows = load_jsonl(dataset_root / "test_pairs_v1.jsonl")
    score_text_views, blend_weights, text_view = _resolve_scoring_config(config)

    document_frequency: Counter[str] = Counter()
    total_documents = 0
    for row in train_rows:
        for field_index in (1, 2):
            tokens = set(_tokenize(_select_pair_text(row, field_index, text_view)))
            if not tokens:
                continue
            document_frequency.update(tokens)
            total_documents += 1

    max_features = int(config.get("max_features", 2048))
    vocabulary = [token for token, _ in document_frequency.most_common(max_features)]
    idf = [math.log((1 + total_documents) / (1 + document_frequency[token])) + 1.0 for token in vocabulary]
    config_payload = {
        "model_type": "bag_of_words_style_scorer_v1",
        "vocabulary": vocabulary,
        "idf": idf,
        "hashing_dim": int(config.get("hashing_dim", 512)),
        "max_features": max_features,
        "text_view": text_view,
        "score_text_views": score_text_views,
        "blend_weights": blend_weights,
        "chunk_size_words": config.get("chunk_size_words"),
        "chunk_overlap_words": config.get("chunk_overlap_words"),
        "chunk_aggregation": config.get("chunk_aggregation", "topk_mean"),
        "chunk_top_k": config.get("chunk_top_k"),
    }
    write_json(final_root / "config.json", config_payload)
    write_json(final_root / "tokenizer.json", {"type": "whitespace_regex", "pattern": TOKEN_RE.pattern})
    write_json(final_root / "tokenizer_config.json", {"lowercase": True})
    write_json(final_root / "special_tokens_map.json", {})
    write_json(final_root / "vocab.json", {token: index for index, token in enumerate(vocabulary)})
    write_json(final_root / "merges.txt", [])
    (final_root / "pytorch_model.bin").write_text("local smoke scorer artifact\n", encoding="utf-8")

    manifest = {
        "artifact_type": "style_scorer",
        "artifact_version": "style_scorer_v1",
        "model_name": "bag_of_words_local",
        "pooling": "mean",
        "use_projection": False,
        "contrastive_mode": "pairwise_similarity",
        "use_topic_adversary": False,
        "semantic_adversary_model": None,
        "max_length": int(config.get("max_length", 384)),
        "primary_score": "calibrated_or_score_0_1",
        "train_pairs_relpath": "../datasets/train_pairs_v1.jsonl",
        "validation_pairs_relpath": "../datasets/validation_pairs_v1.jsonl",
        "test_pairs_relpath": "../datasets/test_pairs_v1.jsonl",
        "system_spec_version": "2026-04-16",
        "benchmark_spec_version": "2026-04-16",
        "canonical_architecture_requested": config.get("base_encoder", "roberta-large"),
        "smoke_mode": True,
        "training_backend": "bow",
        "text_view": text_view,
        "score_text_views": score_text_views,
        "blend_weights": blend_weights,
        "chunking": {
            "chunk_size_words": config.get("chunk_size_words"),
            "chunk_overlap_words": config.get("chunk_overlap_words"),
            "chunk_aggregation": config.get("chunk_aggregation", "topk_mean"),
            "chunk_top_k": config.get("chunk_top_k"),
        },
    }
    write_json(final_root / "scorer_manifest.json", manifest)
    write_json(final_root / "train_config.json", config)

    scorer = StyleScorer(final_root)
    scores = []
    for row in test_rows:
        result = scorer.score_pair(row["text1"], row["text2"])
        score = float(result["score_0_1"])
        prediction = 1 if score >= 0.5 else 0
        scores.append({"label": row["label"], "score": score, "prediction": prediction})
    accuracy = sum(int(item["label"] == item["prediction"]) for item in scores) / max(1, len(scores))
    positive_scores = [item["score"] for item in scores if item["label"] == 1]
    negative_scores = [item["score"] for item in scores if item["label"] == 0]
    metrics = {
        "accuracy": round(accuracy, 6),
        "positive_score_mean": round(sum(positive_scores) / max(1, len(positive_scores)), 6),
        "negative_score_mean": round(sum(negative_scores) / max(1, len(negative_scores)), 6),
        "test_pair_count": len(scores),
    }
    write_json(final_root / "test_metrics.json", metrics)
    build_style_diagnostics(
        model_dir=final_root,
        dataset_root=dataset_root,
        out_path=final_root / "diagnostics_v1.json",
    )
    return {"model_dir": final_root.as_posix(), "metrics": (final_root / "test_metrics.json").as_posix()}


def _supervised_contrastive_loss(embeddings, labels, temperature: float):
    import torch.nn.functional as F

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    logits = torch.matmul(embeddings, embeddings.T) / max(temperature, 1e-6)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-9))
    positive_counts = mask.sum(dim=1)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
    valid = positive_counts > 0
    if not valid.any():
        return torch.tensor(0.0, device=embeddings.device)
    return -mean_log_prob_pos[valid].mean()


def _build_batch_author_ids(rows: list[dict[str, Any]]) -> list[str]:
    author_ids_a = [row["author1"] for row in rows]
    author_ids_b = [row["author2"] for row in rows]
    return author_ids_a + author_ids_b


def _build_batch_content_ids(rows: list[dict[str, Any]]) -> list[int] | None:
    values_a = [row.get("content_cluster1") for row in rows]
    values_b = [row.get("content_cluster2") for row in rows]
    if not any(value is not None for value in values_a + values_b):
        return None
    return [int(value or 0) for value in values_a + values_b]


def _label_encoder(values: list[str]) -> list[int]:
    mapping: dict[str, int] = {}
    encoded: list[int] = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        encoded.append(mapping[value])
    return encoded


def _collate_rows(tokenizer, rows: list[dict[str, Any]], max_length: int, *, text_view: str):
    texts_a = [_select_pair_text(row, 1, text_view) for row in rows]
    texts_b = [_select_pair_text(row, 2, text_view) for row in rows]
    batch_a = tokenizer(texts_a, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    batch_b = tokenizer(texts_b, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = torch.tensor([float(row["label"]) for row in rows], dtype=torch.float32)
    author_labels = torch.tensor(_label_encoder(_build_batch_author_ids(rows)), dtype=torch.long)
    content_ids = _build_batch_content_ids(rows)
    content_labels = torch.tensor(content_ids, dtype=torch.long) if content_ids is not None else None
    return batch_a, batch_b, labels, author_labels, content_labels


def _batched(rows: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _pairwise_logits(embeddings_a, embeddings_b, temperature: float):
    import torch.nn.functional as F

    return F.cosine_similarity(embeddings_a, embeddings_b) / max(temperature, 1e-6)


def _evaluate_transformer_model(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    device,
    batch_size: int,
    max_length: int,
    temperature: float,
    text_view: str,
    adv_lambda: float,
):
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_pairs = 0
    total_correct = 0
    positives = []
    negatives = []
    total_adv_loss = 0.0
    total_adv_correct = 0
    total_adv_examples = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    adversary_criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_rows in _batched(rows, batch_size):
            batch_a, batch_b, labels, author_labels, content_labels = _collate_rows(
                tokenizer,
                batch_rows,
                max_length,
                text_view=text_view,
            )
            batch_a = {key: value.to(device) for key, value in batch_a.items()}
            batch_b = {key: value.to(device) for key, value in batch_b.items()}
            labels = labels.to(device)
            author_labels = author_labels.to(device)
            if content_labels is not None:
                content_labels = content_labels.to(device)
            features_a = model.encode_features(batch_a["input_ids"], batch_a["attention_mask"])
            features_b = model.encode_features(batch_b["input_ids"], batch_b["attention_mask"])
            embeddings_a = features_a["normalized"]
            embeddings_b = features_b["normalized"]
            pair_logits = _pairwise_logits(embeddings_a, embeddings_b, temperature)
            pair_loss = criterion(pair_logits, labels)
            contrastive_loss = _supervised_contrastive_loss(torch.cat([embeddings_a, embeddings_b], dim=0), author_labels, temperature)
            adv_loss = torch.tensor(0.0, device=device)
            if model.topic_adversary is not None and content_labels is not None:
                topic_logits = model.topic_logits(
                    torch.cat([features_a["projected"], features_b["projected"]], dim=0),
                    reversal_scale=0.0,
                )
                adv_loss = adversary_criterion(topic_logits, content_labels)
                predictions = topic_logits.argmax(dim=1)
                total_adv_correct += int((predictions == content_labels).sum().item())
                total_adv_examples += int(content_labels.numel())
                total_adv_loss += float(adv_loss.item()) * len(batch_rows)
            loss = pair_loss + contrastive_loss + (adv_lambda * adv_loss)
            total_loss += float(loss.item()) * len(batch_rows)
            total_pairs += len(batch_rows)
            predictions = (torch.sigmoid(pair_logits) >= 0.5).long()
            total_correct += int((predictions == labels.long()).sum().item())
            scores = ((F.cosine_similarity(embeddings_a, embeddings_b) + 1.0) / 2.0).tolist()
            for row, score in zip(batch_rows, scores):
                if row["label"] == 1:
                    positives.append(score)
                else:
                    negatives.append(score)
    return {
        "loss": total_loss / max(1, total_pairs),
        "accuracy": total_correct / max(1, total_pairs),
        "positive_score_mean": sum(positives) / max(1, len(positives)),
        "negative_score_mean": sum(negatives) / max(1, len(negatives)),
        "pair_count": total_pairs,
        "adv_loss": total_adv_loss / max(1, total_pairs),
        "adv_accuracy": total_adv_correct / max(1, total_adv_examples),
        "adv_examples": total_adv_examples,
    }


def _train_transformer_scorer(config: dict[str, Any]) -> dict[str, str]:
    require_transformer_dependencies()
    from transformers import get_linear_schedule_with_warmup

    artifacts_root = Path(config.get("artifacts_root", "build/artifacts"))
    dataset_root = artifacts_root / "scorer" / "datasets"
    runs_root = artifacts_root / "scorer" / "runs"
    final_root = artifacts_root / "scorer" / "final"
    run_root = runs_root / f"run_{_utc_timestamp_slug()}"
    run_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(dataset_root / "train_pairs_v1.jsonl")
    validation_rows = load_jsonl(dataset_root / "validation_pairs_v1.jsonl")
    test_rows = load_jsonl(dataset_root / "test_pairs_v1.jsonl")
    if not train_rows:
        raise ValueError("training dataset is empty")

    base_encoder = config.get("base_encoder", "roberta-large")
    tokenizer = AutoTokenizer.from_pretrained(base_encoder, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token
    pooling = config.get("pooling", "attn")
    use_projection = bool(config.get("use_projection", True))
    projection_dim = config.get("projection_dim")
    max_length = int(config.get("max_length", 384))
    score_text_views, blend_weights, text_view = _resolve_scoring_config(config)
    use_topic_adversary = bool(config.get("use_topic_adversary", False))
    topic_adversary_num_labels = (
        max(int(row.get("content_cluster1", 0) or 0), int(row.get("content_cluster2", 0) or 0))
        for row in train_rows + validation_rows + test_rows
        if row.get("content_cluster1") is not None or row.get("content_cluster2") is not None
    )
    max_content_cluster = max(topic_adversary_num_labels, default=-1)
    adversary_label_count = max_content_cluster + 1 if use_topic_adversary and max_content_cluster >= 0 else 0
    model = TransformerStyleEncoder(
        base_encoder=base_encoder,
        pooling=pooling,
        use_projection=use_projection,
        projection_dim=projection_dim,
        topic_adversary_num_labels=adversary_label_count,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = int(config.get("batch_size_per_device", 16))
    epochs = int(config.get("epochs", 6))
    learning_rate = float(config.get("learning_rate", 2e-5))
    warmup_steps = int(config.get("warmup_steps", 1000))
    patience = int(config.get("early_stopping_patience", 3))
    temperature = float(config.get("supcon_temperature", 0.07))
    gradient_accumulation = int(config.get("gradient_accumulation", 1))
    adv_lambda = float(config.get("adv_lambda", 0.7))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_train_steps = max(1, math.ceil(len(train_rows) / batch_size / gradient_accumulation) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(warmup_steps, total_train_steps // 2),
        num_training_steps=total_train_steps,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    stalled_epochs = 0
    epoch_summaries = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_pairs = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch_rows in enumerate(_batched(train_rows, batch_size), start=1):
            batch_a, batch_b, labels, author_labels, content_labels = _collate_rows(
                tokenizer,
                batch_rows,
                max_length,
                text_view=text_view,
            )
            batch_a = {key: value.to(device) for key, value in batch_a.items()}
            batch_b = {key: value.to(device) for key, value in batch_b.items()}
            labels = labels.to(device)
            author_labels = author_labels.to(device)
            if content_labels is not None:
                content_labels = content_labels.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                features_a = model.encode_features(batch_a["input_ids"], batch_a["attention_mask"])
                features_b = model.encode_features(batch_b["input_ids"], batch_b["attention_mask"])
                embeddings_a = features_a["normalized"]
                embeddings_b = features_b["normalized"]
                pair_logits = _pairwise_logits(embeddings_a, embeddings_b, temperature)
                pair_loss = criterion(pair_logits, labels)
                contrastive_loss = _supervised_contrastive_loss(torch.cat([embeddings_a, embeddings_b], dim=0), author_labels, temperature)
                adv_loss = torch.tensor(0.0, device=device)
                if model.topic_adversary is not None and content_labels is not None:
                    topic_logits = model.topic_logits(
                        torch.cat([features_a["projected"], features_b["projected"]], dim=0),
                        reversal_scale=1.0,
                    )
                    adv_loss = torch.nn.functional.cross_entropy(topic_logits, content_labels)
                loss = pair_loss + contrastive_loss + (adv_lambda * adv_loss)
                scaled_loss = loss / max(1, gradient_accumulation)
            scaler.scale(scaled_loss).backward()
            if step % gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running_loss += float(loss.item()) * len(batch_rows)
            seen_pairs += len(batch_rows)
        if seen_pairs and (math.ceil(seen_pairs / batch_size) % gradient_accumulation) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        validation_metrics = _evaluate_transformer_model(
            model,
            tokenizer,
            validation_rows or train_rows[: min(len(train_rows), batch_size * 4)],
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            temperature=temperature,
            text_view=text_view,
            adv_lambda=adv_lambda,
        )
        epoch_summary = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, seen_pairs), 6),
            "validation_loss": round(validation_metrics["loss"], 6),
            "validation_accuracy": round(validation_metrics["accuracy"], 6),
            "validation_adv_loss": round(validation_metrics["adv_loss"], 6),
            "validation_adv_accuracy": round(validation_metrics["adv_accuracy"], 6),
        }
        epoch_summaries.append(epoch_summary)
        write_json(run_root / f"epoch_{epoch:02d}.metrics.json", epoch_summary)

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = float(validation_metrics["loss"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stalled_epochs = 0
        else:
            stalled_epochs += 1
            if stalled_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    config_payload = build_model_config(
        base_encoder=base_encoder,
        pooling=pooling,
        use_projection=use_projection,
        projection_dim=projection_dim,
        max_length=max_length,
        text_view=text_view,
        score_text_views=score_text_views,
        blend_weights=blend_weights,
        chunk_size_words=config.get("chunk_size_words"),
        chunk_overlap_words=config.get("chunk_overlap_words"),
        chunk_aggregation=config.get("chunk_aggregation", "topk_mean"),
        chunk_top_k=config.get("chunk_top_k"),
        use_topic_adversary=use_topic_adversary,
        semantic_adversary_model=config.get("semantic_adversary_model"),
        adv_lambda=adv_lambda,
    )
    save_transformer_artifact(model, tokenizer, final_root, config_payload)
    write_json(run_root / "train_history.json", epoch_summaries)
    write_json(run_root / "train_config.json", config)

    manifest = {
        "artifact_type": "style_scorer",
        "artifact_version": "style_scorer_v1",
        "model_name": base_encoder,
        "pooling": pooling,
        "use_projection": use_projection,
        "contrastive_mode": config.get("contrastive_mode", "supcon"),
        "use_topic_adversary": bool(config.get("use_topic_adversary", False)),
        "semantic_adversary_model": config.get("semantic_adversary_model"),
        "adv_lambda": adv_lambda,
        "max_length": max_length,
        "primary_score": "calibrated_or_score_0_1",
        "train_pairs_relpath": "../datasets/train_pairs_v1.jsonl",
        "validation_pairs_relpath": "../datasets/validation_pairs_v1.jsonl",
        "test_pairs_relpath": "../datasets/test_pairs_v1.jsonl",
        "system_spec_version": "2026-04-16",
        "benchmark_spec_version": "2026-04-16",
        "smoke_mode": False,
        "training_backend": "transformer",
        "text_view": text_view,
        "score_text_views": score_text_views,
        "blend_weights": blend_weights,
        "chunking": {
            "chunk_size_words": config.get("chunk_size_words"),
            "chunk_overlap_words": config.get("chunk_overlap_words"),
            "chunk_aggregation": config.get("chunk_aggregation", "topk_mean"),
            "chunk_top_k": config.get("chunk_top_k"),
        },
        "content_cluster_count": adversary_label_count,
        "best_epoch": best_epoch,
        "run_relpath": f"../runs/{run_root.name}",
    }
    write_json(final_root / "scorer_manifest.json", manifest)
    write_json(final_root / "train_config.json", config)

    test_metrics = _evaluate_transformer_model(
        model,
        tokenizer,
        test_rows or validation_rows or train_rows[: min(len(train_rows), batch_size * 4)],
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        temperature=temperature,
        text_view=text_view,
        adv_lambda=adv_lambda,
    )
    output_metrics = {
        "loss": round(test_metrics["loss"], 6),
        "accuracy": round(test_metrics["accuracy"], 6),
        "positive_score_mean": round(test_metrics["positive_score_mean"], 6),
        "negative_score_mean": round(test_metrics["negative_score_mean"], 6),
        "test_pair_count": int(test_metrics["pair_count"]),
        "best_epoch": best_epoch,
        "adv_loss": round(test_metrics["adv_loss"], 6),
        "adv_accuracy": round(test_metrics["adv_accuracy"], 6),
        "adv_examples": int(test_metrics["adv_examples"]),
    }
    write_json(final_root / "test_metrics.json", output_metrics)
    build_style_diagnostics(
        model_dir=final_root,
        dataset_root=dataset_root,
        out_path=final_root / "diagnostics_v1.json",
    )
    return {"model_dir": final_root.as_posix(), "metrics": (final_root / "test_metrics.json").as_posix()}


def train_style_scorer(config_path_or_payload: str | Path | dict[str, Any]) -> dict[str, str]:
    config = _load_config(config_path_or_payload)
    backend = config.get("training_backend")
    inferred_backend = backend is None
    if backend is None:
        backend = "transformer" if AutoTokenizer is not None and torch is not None else "bow"
    if backend == "bow":
        return _train_bow_scorer(config)
    if not inferred_backend:
        return _train_transformer_scorer(config)
    try:
        return _train_transformer_scorer(config)
    except (ImportError, OSError) as exc:
        fallback_config = dict(config)
        fallback_config["training_backend"] = "bow"
        fallback_config["transformer_fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return _train_bow_scorer(fallback_config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_style_scorer(args.config)


if __name__ == "__main__":
    main()
