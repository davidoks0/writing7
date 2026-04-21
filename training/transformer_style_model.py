from __future__ import annotations

from pathlib import Path
from typing import Any


try:  # pragma: no cover - optional heavyweight dependency path
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None
    F = None
    nn = object  # type: ignore[assignment]
    AutoModel = None
    AutoTokenizer = None


def require_transformer_dependencies() -> None:
    if torch is None or AutoModel is None or AutoTokenizer is None or F is None:  # pragma: no cover
        raise ImportError(
            "Transformer scorer training requires torch and transformers. "
            "Install requirements.lock.txt or use the Modal image."
        )


def masked_mean(hidden_states, attention_mask):
    weights = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1e-9)
    return summed / denom


if torch is not None:  # pragma: no branch

    class _GradientReversalFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs, scale):
            ctx.scale = scale
            return inputs.view_as(inputs)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.scale, None


    class GradientReversal(nn.Module):
        def forward(self, inputs, scale: float = 1.0):
            return _GradientReversalFn.apply(inputs, float(scale))


    class TopicAdversaryHead(nn.Module):
        def __init__(self, input_dim: int, num_labels: int, hidden_dim: int | None = None) -> None:
            super().__init__()
            if num_labels < 1:
                raise ValueError("num_labels must be at least 1 when topic adversary is enabled")
            hidden = hidden_dim or max(128, min(1024, input_dim))
            self.gradient_reversal = GradientReversal()
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, num_labels),
            )

        def forward(self, embeddings, *, reversal_scale: float = 1.0):
            reversed_embeddings = self.gradient_reversal(embeddings, reversal_scale)
            return self.classifier(reversed_embeddings)

    class TransformerStyleEncoder(nn.Module):
        def __init__(
            self,
            *,
            base_encoder: str,
            pooling: str = "attn",
            use_projection: bool = True,
            projection_dim: int | None = None,
            topic_adversary_num_labels: int = 0,
            topic_adversary_hidden_dim: int | None = None,
        ) -> None:
            super().__init__()
            require_transformer_dependencies()
            self.base_encoder_name = base_encoder
            self.pooling = pooling
            self.use_projection = use_projection
            self.encoder = AutoModel.from_pretrained(base_encoder)
            hidden_size = int(self.encoder.config.hidden_size)
            self.hidden_size = hidden_size
            self.output_dim = projection_dim or hidden_size
            self.topic_adversary_num_labels = int(topic_adversary_num_labels)
            self.attention_pool = nn.Linear(hidden_size, 1) if pooling == "attn" else None
            if use_projection:
                projection_hidden = max(hidden_size, self.output_dim)
                self.projection = nn.Sequential(
                    nn.Linear(hidden_size, projection_hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(projection_hidden, self.output_dim),
                )
            else:
                self.projection = None
            self.topic_adversary = (
                TopicAdversaryHead(self.output_dim, self.topic_adversary_num_labels, hidden_dim=topic_adversary_hidden_dim)
                if self.topic_adversary_num_labels > 0
                else None
            )

        def _pool(self, hidden_states, attention_mask):
            if self.pooling == "mean" or self.attention_pool is None:
                return masked_mean(hidden_states, attention_mask)
            logits = self.attention_pool(hidden_states).squeeze(-1)
            logits = logits.masked_fill(attention_mask == 0, -1e9)
            weights = torch.softmax(logits, dim=1).unsqueeze(-1)
            return (hidden_states * weights).sum(dim=1)

        def encode_features(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self._pool(outputs.last_hidden_state, attention_mask)
            projected = self.projection(pooled) if self.projection is not None else pooled
            normalized = F.normalize(projected, dim=-1)
            return {
                "pooled": pooled,
                "projected": projected,
                "normalized": normalized,
            }

        def encode(self, input_ids, attention_mask):
            return self.encode_features(input_ids, attention_mask)["normalized"]

        def topic_logits(self, embeddings, *, reversal_scale: float = 1.0):
            if self.topic_adversary is None:
                return None
            return self.topic_adversary(embeddings, reversal_scale=reversal_scale)

        def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
            features_a = self.encode_features(input_ids_a, attention_mask_a)
            features_b = self.encode_features(input_ids_b, attention_mask_b)
            return features_a["normalized"], features_b["normalized"]


else:

    class TransformerStyleEncoder:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            require_transformer_dependencies()


def build_model_config(
    *,
    base_encoder: str,
    pooling: str,
    use_projection: bool,
    projection_dim: int | None,
    max_length: int,
    text_view: str = "raw",
    score_text_views: list[str] | None = None,
    blend_weights: dict[str, float] | None = None,
    chunk_size_words: int | None = None,
    chunk_overlap_words: int | None = None,
    chunk_aggregation: str = "mean",
    chunk_top_k: int | None = None,
    use_topic_adversary: bool = False,
    semantic_adversary_model: str | None = None,
    adv_lambda: float | None = None,
) -> dict[str, Any]:
    return {
        "model_type": "hf_transformer_contrastive_v1",
        "base_encoder": base_encoder,
        "pooling": pooling,
        "use_projection": use_projection,
        "projection_dim": projection_dim,
        "max_length": max_length,
        "text_view": text_view,
        "score_text_views": score_text_views or [text_view],
        "blend_weights": blend_weights or {text_view: 1.0},
        "chunk_size_words": chunk_size_words,
        "chunk_overlap_words": chunk_overlap_words,
        "chunk_aggregation": chunk_aggregation,
        "chunk_top_k": chunk_top_k,
        "use_topic_adversary": use_topic_adversary,
        "semantic_adversary_model": semantic_adversary_model,
        "adv_lambda": adv_lambda,
        "state_dict_path": "pytorch_model.bin",
    }


def save_transformer_artifact(model: "TransformerStyleEncoder", tokenizer, output_dir: str | Path, config_payload: dict[str, Any]) -> None:
    require_transformer_dependencies()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("topic_adversary.")
    }
    torch.save(state_dict, target / "pytorch_model.bin")
    tokenizer.save_pretrained(target)
    (target / "config.json").write_text(__import__("json").dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_transformer_artifact(model_dir: str | Path):
    require_transformer_dependencies()
    source = Path(model_dir)
    config_payload = __import__("json").loads((source / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = TransformerStyleEncoder(
        base_encoder=config_payload["base_encoder"],
        pooling=config_payload.get("pooling", "attn"),
        use_projection=bool(config_payload.get("use_projection", True)),
        projection_dim=config_payload.get("projection_dim"),
    )
    state_dict = torch.load(source / config_payload.get("state_dict_path", "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model, tokenizer, config_payload
