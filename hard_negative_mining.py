"""
Experimental hard-negative mining utilities.

This module adds ANN-based mining of near-duplicate but non-matching pairs
to toughen training. It uses the current contrastive model's encoder to embed
single chunks, builds a nearest-neighbor index, and proposes negatives that are
close in embedding space but the model predicts as negatives.

NOTE: This is an experimental feature we might later remove.
It is designed to be optional and conservative to avoid adding false negatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer

# Reuse model architecture and pooling from the contrastive pipeline
from train_contrastive import ContrastiveBookMatcher


@dataclass
class MiningConfig:
    # ANN mining parameters
    k_neighbors: int = 20
    sim_threshold: float = 0.55  # min cosine similarity to consider as "hard"
    prob_max: float = 0.20       # rerun model; keep only pairs model deems negative with prob <= prob_max
    anchors_per_book: int = 120  # sample this many chunks per book as anchors
    pool_samples_per_book: int = 200  # cap candidate pool per book for memory
    batch_size: int = 32
    max_negatives_per_book: int = 100
    max_total_negatives: Optional[int] = None  # optional global cap


class ContrastiveChunkEmbedder:
    """Embeds single text chunks using the trained contrastive model.

    We load the full model and reuse its encoder + pooling (+ projection) to produce
    per-text embeddings compatible with the training representation.
    """

    def __init__(self, model_dir: str, device: Optional[str] = None) -> None:
        import os
        from safetensors.torch import load_file as safe_load_file

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load state dict (pt or safetensors)
        pt_path = os.path.join(model_dir, "pytorch_model.bin")
        st_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location="cpu")
        elif os.path.exists(st_path):
            state_dict = safe_load_file(st_path)
        else:
            raise FileNotFoundError(f"No model weights found in {model_dir}")

        # Infer base encoder hidden dim to choose pretrained name
        hidden_size = None
        try:
            emb = state_dict.get("encoder.embeddings.word_embeddings.weight")
            if emb is not None:
                hidden_size = int(emb.shape[1])
        except Exception:
            pass
        base_model_map = {768: "roberta-base", 1024: "roberta-large"}
        detected_model_name = base_model_map.get(hidden_size, "roberta-base")

        # Detect head config
        use_style_features = False
        use_symmetric_features = False
        use_projection = False
        pooling = "mean"
        classifier_type = "mlp"

        try:
            if any(k.startswith("arcface.weight") for k in state_dict.keys()):
                classifier_type = "arcface"
        except Exception:
            pass

        head_in_features = None
        try:
            if classifier_type == "arcface":
                w0 = state_dict.get("feat_head.0.weight")
                if w0 is not None:
                    head_in_features = int(w0.shape[1])
            else:
                w0 = state_dict.get("classifier.0.weight")
                if w0 is not None:
                    head_in_features = int(w0.shape[1])
        except Exception:
            head_in_features = None

        if hidden_size is not None and head_in_features is not None:
            base = hidden_size * 2
            residual = head_in_features - base
            if residual >= (hidden_size * 2):
                use_symmetric_features = True
                residual -= hidden_size * 2
            if residual > 0:
                use_style_features = True

        try:
            if any(k.startswith("proj.0.weight") for k in state_dict.keys()):
                use_projection = True
        except Exception:
            pass

        try:
            if any(k.startswith("attn_mlp.0.weight") for k in state_dict.keys()):
                pooling = "attn"
        except Exception:
            pass

        # Build model and load weights
        self.model = ContrastiveBookMatcher(
            model_name=detected_model_name,
            use_style_features=use_style_features,
            use_symmetric_features=use_symmetric_features,
            pooling=pooling,
            use_projection=use_projection,
            classifier=classifier_type,
            use_topic_adversary=False,
            n_topics=5,
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str], batch_size: int = 32, max_length: int = 512) -> np.ndarray:
        """Return L2-normalized pooled embeddings for given texts."""
        vecs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                list(batch),
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model.encoder(enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            pooled = self.model._pool_embeddings(out, enc["attention_mask"])  # type: ignore[attr-defined]
            if getattr(self.model, "proj", None) is not None:
                pooled = self.model.proj(pooled)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs.append(pooled.detach().cpu().numpy())
        return np.vstack(vecs) if vecs else np.zeros((0, 1), dtype=np.float32)

    @torch.no_grad()
    def pair_probs(self, pairs: Sequence[Tuple[str, str]], batch_size: int = 16, max_length: int = 512) -> np.ndarray:
        """Compute positive-class probabilities for text pairs using the contrastive model head."""
        from train_contrastive import extract_style_features_batch

        probs: List[np.ndarray] = []
        for i in range(0, len(pairs), batch_size):
            sub = pairs[i : i + batch_size]
            t1 = [a for a, _ in sub]
            t2 = [b for _, b in sub]
            enc1 = self.tokenizer(t1, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            enc2 = self.tokenizer(t2, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            s1 = torch.tensor(extract_style_features_batch(t1), dtype=torch.float32)
            s2 = torch.tensor(extract_style_features_batch(t2), dtype=torch.float32)
            batch_in = {
                "input_ids_1": enc1["input_ids"].to(self.device),
                "attention_mask_1": enc1["attention_mask"].to(self.device),
                "input_ids_2": enc2["input_ids"].to(self.device),
                "attention_mask_2": enc2["attention_mask"].to(self.device),
                "style_features_1": s1.to(self.device),
                "style_features_2": s2.to(self.device),
            }
            logits = self.model(**batch_in)["logits"]
            p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.append(p)
        return np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)


def _sample_per_book(book_chunks: Dict[str, List[str]], limit: int) -> Tuple[List[str], List[Tuple[int, str]]]:
    texts: List[str] = []
    owners: List[Tuple[int, str]] = []  # (index, book_id)
    for bid, chunks in book_chunks.items():
        if not chunks:
            continue
        if limit is not None and limit > 0 and len(chunks) > limit:
            idxs = np.random.choice(len(chunks), size=limit, replace=False)
            sel = [chunks[i] for i in idxs]
        else:
            sel = list(chunks)
        start = len(texts)
        texts.extend(sel)
        owners.extend([(start + j, bid) for j in range(len(sel))])
    return texts, owners


def mine_ann_negatives(
    book_chunks: Dict[str, List[str]],
    embedder: ContrastiveChunkEmbedder,
    cfg: MiningConfig = MiningConfig(),
) -> List[Dict[str, object]]:
    """Mine hard negatives using ANN over chunk embeddings.

    Returns a list of dicts with keys: {'text1','text2','label'} where label=0.
    """
    if not book_chunks:
        return []

    # Sample anchors and candidate pool
    anchors_text, anchors_owner = _sample_per_book(book_chunks, cfg.anchors_per_book)
    pool_text, pool_owner = _sample_per_book(book_chunks, cfg.pool_samples_per_book)
    if not anchors_text or not pool_text:
        return []

    # Embed
    A = embedder.embed_texts(anchors_text, batch_size=cfg.batch_size)
    P = embedder.embed_texts(pool_text, batch_size=cfg.batch_size)

    # Build ANN on pool
    nn = NearestNeighbors(n_neighbors=min(cfg.k_neighbors + 1, len(pool_text)), metric="cosine", algorithm="brute")
    nn.fit(P)
    dist, idx = nn.kneighbors(A, return_distance=True)
    # cosine distance -> similarity
    sims = 1.0 - dist

    mined: List[Dict[str, object]] = []
    neg_count_by_book: Dict[str, int] = {}

    # Optional safety: batch rescore with model probs
    candidate_pairs: List[Tuple[str, str]] = []
    candidate_meta: List[Tuple[str, int]] = []  # (anchor_book, neighbor_index)

    for i, (sim_row, idx_row) in enumerate(zip(sims, idx)):
        a_text = anchors_text[i]
        a_book = anchors_owner[i][1]
        taken = 0
        for sim, j in zip(sim_row, idx_row):
            if j < 0:
                continue
            if sim < cfg.sim_threshold:
                continue
            nb_text = pool_text[j]
            nb_book = pool_owner[j][1]
            if nb_book == a_book:
                continue  # skip same-book
            candidate_pairs.append((a_text, nb_text))
            candidate_meta.append((a_book, j))
            taken += 1
            if taken >= cfg.k_neighbors:
                break

    if not candidate_pairs:
        return []

    probs = embedder.pair_probs(candidate_pairs, batch_size=max(8, cfg.batch_size // 2))

    for (a_book, j), (t1, t2), p in zip(candidate_meta, candidate_pairs, probs):
        if float(p) <= cfg.prob_max:
            # book-level cap
            n = neg_count_by_book.get(a_book, 0)
            if n >= cfg.max_negatives_per_book:
                continue
            mined.append({"text1": t1, "text2": t2, "label": 0})
            neg_count_by_book[a_book] = n + 1
            # global cap
            if cfg.max_total_negatives is not None and len(mined) >= cfg.max_total_negatives:
                break

    return mined


