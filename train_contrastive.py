"""
Train a contrastive RoBERTa-based model for pure style embedding learning.

Implements:
- Siamese architecture with shared encoder
- SupCon / InfoNCE contrastive loss for style similarity
- Optional topic adversary (GRL) for style-content disentanglement
- Temperature curriculum for contrastive learning
"""
import os
import time
import contextlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, fbeta_score, balanced_accuracy_score, average_precision_score
import numpy as np

# Prefer the new SDPA kernel context manager when available; avoid deprecated fallbacks.
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel_ctx, SDPBackend
    _FAST_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
except Exception:
    _sdpa_kernel_ctx = None
    _FAST_BACKENDS = None

def _fast_sdpa_ctx():
    """Return a context manager that prefers Flash/MemEff SDPA kernels.

    Uses torch.nn.attention.sdpa_kernel when available (Torch >= 2.1). If not
    available, returns a no-op context instead of calling the deprecated
    torch.backends.cuda.sdp_kernel to avoid deprecation warnings.
    """
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    if _sdpa_kernel_ctx is not None and _FAST_BACKENDS is not None:
        try:
            return _sdpa_kernel_ctx(backends=_FAST_BACKENDS)
        except Exception:
            pass
    # Fallback: no special kernel context
    return contextlib.nullcontext()


class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training.

    Forward: identity. Backward: multiply incoming gradient by -lambda.
    Use with GradReverse.apply(x, lambda).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class ContrastiveBookMatcher(nn.Module):
    """
    Siamese model for pure contrastive style embedding learning.

    Architecture:
    - Shared RoBERTa encoder
    - Attention / mean / CLS pooling
    - Optional projection head
    - SupCon or InfoNCE contrastive loss
    - Optional topic adversary (GRL) for style-content disentanglement
    """

    def __init__(
        self,
        model_name: str = 'roberta-base',
        pooling: str = 'attn',
        use_projection: bool = True,
        contrastive_mode: str = 'supcon',  # 'supcon' or 'infonce'
        # Topic adversary (default ON)
        use_topic_adversary: bool = True,
        adv_lambda: float = 0.4,
        n_topics: int = 5,
        # Semantic adversary config
        semantic_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        semantic_loss: str = 'cosine',  # 'cosine' or 'mse'
        semantic_max_length: int = 256,
        # Enhancements
        multi_head_adversary: bool = False,
        supcon_temperature: float = 0.07,  # Literature suggests 0.07-0.1 works best
        # Online hard negative mining: weight hard negatives more in the loss
        hard_negative_weight: float = 1.0,  # >1 to emphasize hard negatives
        # Legacy kwargs accepted for backward compatibility with old checkpoints/callers
        **_kwargs,
    ):
        super().__init__()
        self.pooling_type = pooling
        self.use_projection = bool(use_projection)
        self.contrastive_mode = contrastive_mode
        self.use_topic_adversary = bool(use_topic_adversary)
        self.multi_head_adversary = bool(multi_head_adversary)
        self.adv_lambda = float(adv_lambda)
        self.n_topics = int(n_topics)
        # Always contrastive-only; kept as attribute so inference code that checks it still works
        self.contrastive_only = True
        # Online hard negative mining weight (>1 emphasizes hard negatives)
        self.hard_negative_weight = float(hard_negative_weight)
        # Semantic adversary settings
        self.semantic_model_name = str(semantic_model_name)
        self.semantic_loss = str(semantic_loss)
        self.semantic_max_length = int(semantic_max_length)
        # GRL scale is runtime-scheduled; start at 0 (no adversary) and ramp up via callback
        self.grl_scale: float = 0.0

        # Load base model
        config = AutoConfig.from_pretrained(model_name)
        # Load without pooling layer (we pool manually)
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        # Prefer FlashAttention v2 when available; fall back to SDPA. Quality is unchanged, kernels are faster.
        try:
            if torch.cuda.is_available():
                _fa_ok = False
                try:
                    import flash_attn  # noqa: F401
                    _fa_ok = True
                except Exception:
                    _fa_ok = False
                if _fa_ok and hasattr(self.encoder, 'config') and hasattr(self.encoder.config, 'attn_implementation'):
                    # Transformers will validate head size/dtypes; if unsupported it will raise and we'll ignore.
                    try:
                        setattr(self.encoder.config, 'attn_implementation', 'flash_attention_2')
                    except Exception:
                        # Fall back to SDPA below
                        setattr(self.encoder.config, 'attn_implementation', 'sdpa')
                elif hasattr(self.encoder, 'config') and hasattr(self.encoder.config, 'attn_implementation'):
                    setattr(self.encoder.config, 'attn_implementation', 'sdpa')
        except Exception:
            pass
        
        hidden_dim = config.hidden_size

        # Attention pooling (optional)
        if self.pooling_type == 'attn':
            self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.attn_mlp = None

        # Optional projection head (keeps same dim)
        if self.use_projection:
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
            )
        else:
            self.proj = None

        # Frozen semantic encoder and regression adversary head(s)
        self.semantic_tokenizer = None
        self.semantic_encoder = None
        self.semantic_dim: int | None = None
        if self.use_topic_adversary:
            try:
                self.semantic_tokenizer = AutoTokenizer.from_pretrained(self.semantic_model_name)
                self.semantic_encoder = AutoModel.from_pretrained(self.semantic_model_name, add_pooling_layer=False)
                for p in self.semantic_encoder.parameters():
                    p.requires_grad = False
                self.semantic_encoder.eval()
                try:
                    self.semantic_dim = int(getattr(self.semantic_encoder.config, 'hidden_size'))
                except Exception:
                    self.semantic_dim = hidden_dim
            except Exception as _e:
                print(f"Warning: failed to load semantic adversary model '{self.semantic_model_name}': {_e}. Disabling adversary.")
                self.use_topic_adversary = False
        if self.use_topic_adversary and self.semantic_dim is not None:
            # Backward-compat: allow overriding final output dim to match checkpoints
            _topic_out_kwarg = _kwargs.get('topic_out_dim', None)
            _topic_out = int(_topic_out_kwarg) if (_topic_out_kwarg is not None and int(_topic_out_kwarg) > 0) else int(self.semantic_dim)
            self.topic_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, _topic_out),
            )
            self.topic_head_pre = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, _topic_out),
            ) if self.multi_head_adversary else None
        else:
            self.topic_head = None
            self.topic_head_pre = None
        
        # Temperature for contrastive learning
        # Fixed temperature (not learnable) - literature shows 0.07-0.1 works best
        # Using register_buffer so it's saved with the model but not updated by optimizer
        self.register_buffer('temperature', torch.tensor([float(supcon_temperature)]))
        
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                labels=None,
                book_ids_1: Optional[torch.Tensor] = None,
                book_ids_2: Optional[torch.Tensor] = None,
                author_ids_1: Optional[torch.Tensor] = None,
                author_ids_2: Optional[torch.Tensor] = None,
                text1: Optional[List[str]] = None,
                text2: Optional[List[str]] = None,
                # Legacy kwargs accepted so callers that pass style_features/topic_labels don't break
                **_fwd_kwargs):
        """
        Forward pass for Siamese architecture.

        Args:
            input_ids_1, input_ids_2: Tokenized text pairs
            attention_mask_1, attention_mask_2: Attention masks
            labels: Ground truth labels

        Returns:
            dict with loss, logits (None), emb1, emb2
        """
        # Encode both texts. Use SDPA preferences if supported to favor flash/mem-efficient kernels on H100/H200.
        with _fast_sdpa_ctx():
            emb1 = self.encoder(input_ids_1, attention_mask=attention_mask_1).last_hidden_state
            emb2 = self.encoder(input_ids_2, attention_mask=attention_mask_2).last_hidden_state

        # Pool embeddings
        emb1_pooled = self._pool_embeddings(emb1, attention_mask_1)
        emb2_pooled = self._pool_embeddings(emb2, attention_mask_2)

        # Optional projection (retain pre-projection for adversary)
        emb1_pre = emb1_pooled
        emb2_pre = emb2_pooled
        if self.proj is not None:
            emb1_pooled = self.proj(emb1_pooled)
            emb2_pooled = self.proj(emb2_pooled)

        # Semantic adversarial predictions (regression to frozen embeddings)
        topic_logits_1 = None
        topic_logits_2 = None
        topic_logits_1_pre = None
        topic_logits_2_pre = None
        targets_sem_1 = None
        targets_sem_2 = None
        if self.use_topic_adversary and self.topic_head is not None:
            # Compute frozen semantic targets when raw texts provided
            if (text1 is not None) and (text2 is not None) and (self.semantic_tokenizer is not None) and (self.semantic_encoder is not None):
                try:
                    dev = emb1_pooled.device
                    self.semantic_encoder.to(dev)
                except Exception:
                    dev = emb1_pooled.device
                with torch.no_grad():
                    batch1 = self.semantic_tokenizer(text1, truncation=True, padding=True, max_length=int(self.semantic_max_length), return_tensors='pt')
                    batch2 = self.semantic_tokenizer(text2, truncation=True, padding=True, max_length=int(self.semantic_max_length), return_tensors='pt')
                    batch1 = {k: v.to(dev) for k, v in batch1.items()}
                    batch2 = {k: v.to(dev) for k, v in batch2.items()}
                    out1 = self.semantic_encoder(**batch1).last_hidden_state
                    out2 = self.semantic_encoder(**batch2).last_hidden_state
                    def _mean_pool(h, mask):
                        mask = mask.unsqueeze(-1).float()
                        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                    targets_sem_1 = _mean_pool(out1, batch1.get('attention_mask', torch.ones_like(out1[:, :, 0])))
                    targets_sem_2 = _mean_pool(out2, batch2.get('attention_mask', torch.ones_like(out2[:, :, 0])))
            # Apply GRL to pooled embeddings before regression heads
            def _grl(x):
                return GradReverse.apply(x, float(self.grl_scale))
            z1 = _grl(emb1_pooled) if self.training else emb1_pooled
            z2 = _grl(emb2_pooled) if self.training else emb2_pooled
            topic_logits_1 = self.topic_head(z1)
            topic_logits_2 = self.topic_head(z2)
            if self.multi_head_adversary and self.topic_head_pre is not None:
                z1p = _grl(emb1_pre) if self.training else emb1_pre
                z2p = _grl(emb2_pre) if self.training else emb2_pre
                topic_logits_1_pre = self.topic_head_pre(z1p)
                topic_logits_2_pre = self.topic_head_pre(z2p)
        
        # Compute loss
        loss = None
        if labels is not None:
            # Normalize embeddings for cosine similarity
            emb1_norm = F.normalize(emb1_pooled, p=2, dim=1)
            emb2_norm = F.normalize(emb2_pooled, p=2, dim=1)
            device = emb1_pooled.device

            # Compute contrastive loss (SupCon or InfoNCE)
            # Note: We initialize to None and ensure all code paths set a proper loss
            # that's connected to the computation graph (has grad_fn)
            info_nce_loss = None

            # Contrastive loss: use ALL pairs (both positive and negative)
            # Positive pairs should be pulled together, negative pairs pushed apart
            if self.contrastive_mode == 'supcon':
                # Build a 2B x 2B similarity matrix
                Z = torch.cat([emb1_norm, emb2_norm], dim=0)
                sim = Z @ Z.T / self.temperature
                eye = torch.eye(sim.size(0), device=device, dtype=torch.bool)
                sim = sim.masked_fill(eye, -1e9)
                B = emb1_norm.size(0)

                # Prefer author-level grouping (learns style), fall back to book-level
                # Author-level: same author = positive (even if different books)
                # This teaches the model to recognize author style, not book identity
                group_ids_1 = author_ids_1 if author_ids_1 is not None else book_ids_1
                group_ids_2 = author_ids_2 if author_ids_2 is not None else book_ids_2

                if group_ids_1 is not None and group_ids_2 is not None:
                    # Multi-positive SupCon: same group_id = positive
                    groups = torch.cat([group_ids_1, group_ids_2], dim=0)
                    P = (groups.unsqueeze(0) == groups.unsqueeze(1)) & (~eye)

                    # Online hard negative mining: weight negatives by their difficulty
                    # Hard negatives (high similarity but different group) get higher weight
                    if self.hard_negative_weight > 1.0:
                        # Create negative mask
                        N = (~P) & (~eye)
                        # Weight = sim^(hnw-1) for negatives, 1 for positives
                        # This emphasizes high-similarity negatives
                        with torch.no_grad():
                            # Detach to avoid gradient through weights
                            neg_sims = sim.masked_fill(~N, 0.0).detach()
                            # Normalize to [0,1] range and apply power weighting
                            neg_sims_norm = (neg_sims - neg_sims.min()) / (neg_sims.max() - neg_sims.min() + 1e-6)
                            neg_weights = neg_sims_norm.pow(self.hard_negative_weight - 1.0)
                            # Scale weights for negatives, 1 for positives/self
                            weights = torch.where(N, neg_weights, torch.ones_like(neg_weights))
                        # Apply weights to similarities
                        weighted_sim = sim + torch.log(weights.clamp(min=1e-6))
                    else:
                        weighted_sim = sim

                    pos_logits = weighted_sim.masked_fill(~P, -1e9)
                    num = torch.logsumexp(pos_logits, dim=1)
                    den = torch.logsumexp(weighted_sim, dim=1)
                    info_nce_loss = (den - num).mean()
                else:
                    # Fallback: use labels - positive pairs have matching indices
                    pos_idx = (labels == 1).nonzero(as_tuple=False).squeeze(-1)
                    if pos_idx.numel() > 0:
                        anchors = torch.cat([pos_idx, pos_idx + B], dim=0)
                        positives = torch.cat([pos_idx + B, pos_idx], dim=0)
                        sim_rows = sim[anchors]
                        info_nce_loss = F.cross_entropy(sim_rows, positives)
                    else:
                        # No positive pairs - create dummy loss connected to computation graph
                        info_nce_loss = (emb1_norm.sum() * 0.0) + (emb2_norm.sum() * 0.0)
            else:
                # InfoNCE: diagonal matching
                sim12 = emb1_norm @ emb2_norm.T / self.temperature
                sim21 = emb2_norm @ emb1_norm.T / self.temperature
                targets = torch.arange(sim12.size(0), device=device)
                # Only positive pairs contribute; negatives are in-batch
                pos_idx = (labels == 1).nonzero(as_tuple=False).squeeze(-1)
                if pos_idx.numel() > 0:
                    loss12 = F.cross_entropy(sim12[pos_idx], targets[pos_idx])
                    loss21 = F.cross_entropy(sim21[pos_idx], targets[pos_idx])
                    info_nce_loss = 0.5 * (loss12 + loss21)
                else:
                    # No positive pairs - create dummy loss connected to computation graph
                    info_nce_loss = (emb1_norm.sum() * 0.0) + (emb2_norm.sum() * 0.0)

            # Contrastive loss + optional adversary for disentanglement
            # Adversary loss pushes content/topic information OUT of embeddings
            adv_loss = torch.tensor(0.0, device=device)
            if self.use_topic_adversary and topic_logits_1 is not None and topic_logits_2 is not None and targets_sem_1 is not None and targets_sem_2 is not None:
                if self.semantic_loss.lower() == 'mse':
                    p1 = F.normalize(topic_logits_1, p=2, dim=1)
                    p2 = F.normalize(topic_logits_2, p=2, dim=1)
                    t1 = F.normalize(targets_sem_1.detach(), p=2, dim=1)
                    t2 = F.normalize(targets_sem_2.detach(), p=2, dim=1)
                    adv_loss = 0.5 * (F.mse_loss(p1, t1) + F.mse_loss(p2, t2))
                else:
                    y = torch.ones(targets_sem_1.size(0), device=device)
                    adv_loss = 0.5 * (
                        F.cosine_embedding_loss(topic_logits_1, targets_sem_1.detach(), y) +
                        F.cosine_embedding_loss(topic_logits_2, targets_sem_2.detach(), y)
                    )

            loss = info_nce_loss + self.adv_lambda * adv_loss

        return {
            'loss': loss,
            'logits': None,
            'emb1': emb1_pooled,
            'emb2': emb2_pooled,
            'topic_logits_1': topic_logits_1,
            'topic_logits_2': topic_logits_2,
        }
    
    def _pool_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool embeddings with attention mask (mean, cls, or attention)."""
        if self.pooling_type == 'cls':
            return embeddings[:, 0]
        if self.pooling_type == 'attn' and self.attn_mlp is not None:
            # Compute attention weights
            attn_scores = self.attn_mlp(embeddings).squeeze(-1)  # (B, L)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, L, 1)
            pooled = torch.sum(embeddings * attn_weights, dim=1)
            return pooled
        # Fallback: mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class ContrastiveTrainer(Trainer):
    """Custom Trainer for contrastive style embedding learning."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute contrastive loss.

        Accepts `num_items_in_batch` for compatibility with recent Transformers versions.
        """
        labels = inputs.pop('labels')
        # Keep raw text fields for semantic adversary targets
        texts1 = inputs.pop('text1', None)
        texts2 = inputs.pop('text2', None)
        # Optional hashed book/author ids for multi-positive SupCon
        b1 = inputs.pop('book1', None)
        b2 = inputs.pop('book2', None)
        a1 = inputs.pop('author1', None)
        a2 = inputs.pop('author2', None)

        def _hash_ids(xs):
            if xs is None:
                return None
            try:
                import hashlib
                ints = []
                for s in xs:
                    s = '' if s is None else str(s)
                    h = int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16)
                    ints.append(h)
                return torch.tensor(ints, device=labels.device, dtype=torch.long)
            except Exception:
                return None

        # Only needed for SupCon multi-positive semantics; skip hashing otherwise for a tiny CPU win
        book_ids_1, book_ids_2 = None, None
        author_ids_1, author_ids_2 = None, None
        if getattr(model, 'contrastive_mode', '') == 'supcon':
            if b1 is not None and b2 is not None:
                book_ids_1 = _hash_ids(b1)
                book_ids_2 = _hash_ids(b2)
            if a1 is not None and a2 is not None:
                author_ids_1 = _hash_ids(a1)
                author_ids_2 = _hash_ids(a2)

        outputs = model(
            **inputs, labels=labels,
            book_ids_1=book_ids_1, book_ids_2=book_ids_2,
            author_ids_1=author_ids_1, author_ids_2=author_ids_2,
            text1=texts1, text2=texts2
        )
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss


class AdversarySchedulerCallback(TrainerCallback):
    """Linearly ramp adversarial strength (adv_lambda) and GRL scale during training.

    - Warmup: adversary off for a fraction of total steps.
    - Ramp: linearly increase to max over a fraction of total steps.
    - Hold: keep at max for the remainder.
    """

    def __init__(
        self,
        model: ContrastiveBookMatcher,
        warmup_ratio: float = 0.1,
        ramp_ratio: float = 0.3,
        max_adv_lambda: float = 0.4,
        max_grl_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.warmup_ratio = max(0.0, float(warmup_ratio))
        self.ramp_ratio = max(0.0, float(ramp_ratio))
        self.max_adv_lambda = max(0.0, float(max_adv_lambda))
        self.max_grl_scale = max(0.0, float(max_grl_scale))
        self._max_steps: int | None = None

    def _scale(self, step: int) -> float:
        if not self._max_steps or self._max_steps <= 0:
            return 1.0
        p = step / float(self._max_steps)
        if p < self.warmup_ratio:
            return 0.0
        ramp_end = self.warmup_ratio + self.ramp_ratio
        if p < ramp_end and self.ramp_ratio > 0:
            return (p - self.warmup_ratio) / self.ramp_ratio
        return 1.0

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            self._max_steps = int(state.max_steps)
        except Exception:
            self._max_steps = None
        # Initialize at zero
        if getattr(self.model, 'use_topic_adversary', False):
            self.model.grl_scale = 0.0
            # Keep current adv_lambda value as the target max; do not overwrite user choice
            self._user_max_lambda = float(getattr(self.model, 'adv_lambda', self.max_adv_lambda))
            self.model.adv_lambda = 0.0

    def on_step_begin(self, args, state, control, **kwargs):
        if not getattr(self.model, 'use_topic_adversary', False):
            return
        s = self._scale(state.global_step)
        # Respect user-provided adv_lambda cap if set; else fallback to callback default
        max_lambda = getattr(self, '_user_max_lambda', self.max_adv_lambda)
        self.model.adv_lambda = float(max_lambda) * s
        self.model.grl_scale = float(self.max_grl_scale) * s


class TemperatureSchedulerCallback(TrainerCallback):
    """Curriculum learning via temperature annealing for contrastive loss.

    - Start with higher temperature (easier: broader positive associations)
    - Anneal to target temperature (harder: more precise discrimination)
    - Warmup: keep at start temperature for initial steps
    - Anneal: linearly decrease from start to target temperature

    Literature suggests starting at 0.1-0.2 and annealing to 0.05-0.07.
    """

    def __init__(
        self,
        model: ContrastiveBookMatcher,
        start_temperature: float = 0.15,
        target_temperature: float = 0.07,
        warmup_ratio: float = 0.1,
        anneal_ratio: float = 0.6,
    ) -> None:
        super().__init__()
        self.model = model
        self.start_temperature = max(0.01, float(start_temperature))
        self.target_temperature = max(0.01, float(target_temperature))
        self.warmup_ratio = max(0.0, float(warmup_ratio))
        self.anneal_ratio = max(0.0, float(anneal_ratio))
        self._max_steps: int | None = None

    def _get_temperature(self, step: int) -> float:
        if not self._max_steps or self._max_steps <= 0:
            return self.target_temperature
        p = step / float(self._max_steps)
        # Warmup: stay at start temperature
        if p < self.warmup_ratio:
            return self.start_temperature
        # Anneal: linearly decrease from start to target
        anneal_end = self.warmup_ratio + self.anneal_ratio
        if p < anneal_end and self.anneal_ratio > 0:
            anneal_progress = (p - self.warmup_ratio) / self.anneal_ratio
            return self.start_temperature - anneal_progress * (self.start_temperature - self.target_temperature)
        # Hold: stay at target temperature
        return self.target_temperature

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            self._max_steps = int(state.max_steps)
        except Exception:
            self._max_steps = None
        # Initialize at start temperature
        self.model.supcon_temperature = self.start_temperature

    def on_step_begin(self, args, state, control, **kwargs):
        temp = self._get_temperature(state.global_step)
        self.model.supcon_temperature = temp


class PerfLoggerCallback(TrainerCallback):
    """Lightweight per-step performance logger.

    - Estimates data loading time as time between last step end and current step begin.
    - Measures step compute time as time between step begin and end (forward+backward+optimizer).
    - Logs smoothed averages every N steps on rank 0 only to avoid spam.
    """

    def __init__(self, print_every: int = 50) -> None:
        super().__init__()
        self.print_every = max(1, int(print_every))
        self._last_end_t: float | None = None
        self._step_start_t: float | None = None
        self._ema_data: float | None = None
        self._ema_step: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Hardware + run summary (rank 0)
        is_w0 = getattr(state, "is_world_process_zero", False)
        if is_w0:
            try:
                ws = int(os.environ.get("WORLD_SIZE", "1"))
            except Exception:
                ws = 1
            try:
                rk = int(os.environ.get("RANK", "0"))
            except Exception:
                rk = 0
            gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            devs = []
            for i in range(gpus):
                try:
                    devs.append(torch.cuda.get_device_name(i))
                except Exception:
                    devs.append("cuda")
            print({
                "world_size": ws,
                "rank": rk,
                "gpus": gpus,
                "devices": devs,
                "bf16": bool(getattr(args, "bf16", False)),
                "fp16": bool(getattr(args, "fp16", False)),
                "dl_workers": int(getattr(args, "dataloader_num_workers", 0)),
                "batch_per_device": int(getattr(args, "per_device_train_batch_size", 0)),
                "grad_accum": int(getattr(args, "gradient_accumulation_steps", 1)),
                "max_steps": (int(getattr(state, "max_steps", 0)) if getattr(state, "max_steps", None) is not None else None),
            })
        self._last_end_t = time.perf_counter()

    def on_step_begin(self, args, state, control, **kwargs):
        now = time.perf_counter()
        if self._last_end_t is not None:
            data_t = now - self._last_end_t
            # EMA smoothing
            if self._ema_data is None:
                self._ema_data = data_t
            else:
                self._ema_data = 0.9 * self._ema_data + 0.1 * data_t
        self._step_start_t = now

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        if self._step_start_t is not None:
            step_t = now - self._step_start_t
            if self._ema_step is None:
                self._ema_step = step_t
            else:
                self._ema_step = 0.9 * self._ema_step + 0.1 * step_t
        self._last_end_t = now
        # Rank 0 logging
        if getattr(state, "is_world_process_zero", False):
            gs = getattr(state, "global_step", 0)
            if int(gs) % self.print_every == 0 and self._ema_step is not None:
                data_ms = (self._ema_data or 0.0) * 1000.0
                step_ms = self._ema_step * 1000.0
                total_ms = data_ms + step_ms
                it_s = 1000.0 / total_ms if total_ms > 0 else 0.0
                # Optional CUDA memory snapshot for GPU 0
                gpu_mem_gb = None
                if torch.cuda.is_available():
                    try:
                        gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
                    except Exception:
                        gpu_mem_gb = None
                print({
                    "step": int(gs),
                    "it_per_s": round(it_s, 3),
                    "data_ms": round(data_ms, 1),
                    "compute_ms": round(step_ms, 1),
                    "gpu0_mem_GB": (round(gpu_mem_gb, 2) if gpu_mem_gb is not None else None),
                })


def tokenize_pair(examples, tokenizer, max_length: int = 512, dynamic_padding: bool = True):
    """Tokenize pairs of texts separately for Siamese architecture."""
    # Tokenize both texts
    encoded1 = tokenizer(
        examples['text1'],
        truncation=True,
        padding=(False if dynamic_padding else 'max_length'),
        max_length=max_length
    )
    
    encoded2 = tokenizer(
        examples['text2'],
        truncation=True,
        padding=(False if dynamic_padding else 'max_length'),
        max_length=max_length
    )
    
    return {
        'input_ids_1': encoded1['input_ids'],
        'attention_mask_1': encoded1['attention_mask'],
        'input_ids_2': encoded2['input_ids'],
        'attention_mask_2': encoded2['attention_mask'],
        # Optional: book ids
        'book1': (examples['book1'] if 'book1' in examples else [''] * len(examples['text1'])),
        'book2': (examples['book2'] if 'book2' in examples else [''] * len(examples['text2'])),
        # Keep raw text for semantic adversary targets
        'text1': examples['text1'],
        'text2': examples['text2'],
    }


def _extract_logits_np(preds_obj) -> np.ndarray:
    """Robustly extract a (N, C) logits ndarray from possibly nested predictions.

    Handles cases where HF aggregates multiple outputs (e.g., logits + auxiliary heads)
    and passes them as a list/tuple or object-dtype array.
    """
    import numpy as _np

    def _try_array(x):
        try:
            arr = _np.asarray(x)
            return arr
        except Exception:
            return None

    # Direct ndarray
    arr = _try_array(preds_obj)
    if arr is not None and arr.dtype != object and arr.ndim >= 2:
        return arr

    # Tuple/list of arrays: select the one with last dim == 2 (binary logits)
    if isinstance(preds_obj, (list, tuple)):
        for x in preds_obj:
            ax = _try_array(x)
            if ax is not None and ax.ndim >= 2 and ax.shape[-1] == 2:
                return ax
        # Fallback to first
        ax = _try_array(preds_obj[0])
        if ax is not None:
            return ax

    # Object-dtype array (ragged). Try elements.
    if arr is not None and arr.dtype == object:
        for x in arr:
            ax = _try_array(x)
            if ax is not None and ax.ndim >= 2 and ax.shape[-1] == 2:
                return ax
        # Fallback to first element
        ax = _try_array(arr[0])
        if ax is not None:
            return ax

    # Last resort: convert to numpy and hope it's 2D
    return _np.asarray(preds_obj)


def compute_metrics(eval_pred):
    """Compute accuracy, F1, precision, recall, PR AUC, and ROC AUC.

    Handles cases where `label_ids` is a tuple/list/dict (e.g., multiple label-like
    fields present in the dataset). We select the primary binary labels.
    """
    logits = _extract_logits_np(eval_pred.predictions)
    # Ensure (N, C)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 2)

    N = logits.shape[0]

    def _to_np(arr):
        try:
            x = np.asarray(arr)
            return x
        except Exception:
            return None

    labels_obj = eval_pred.label_ids

    # Extract the primary label vector from possibly nested structures
    labels_np = None
    if isinstance(labels_obj, (list, tuple)):
        # Prefer a binary vector with matching length
        candidates = []
        for x in labels_obj:
            ax = _to_np(x)
            if ax is None:
                continue
            ax = ax.reshape(-1)
            if ax.shape[0] == N:
                candidates.append(ax)
        # Choose first binary candidate, else first matching-length candidate
        for c in candidates:
            uniq = np.unique(c)
            if set(uniq.tolist()).issubset({0, 1}):
                labels_np = c
                break
        if labels_np is None and candidates:
            labels_np = candidates[0]
    elif isinstance(labels_obj, dict):
        # If dict-like, try common keys
        for key in ['labels', 'label']:
            if key in labels_obj:
                ax = _to_np(labels_obj[key])
                if ax is not None:
                    labels_np = ax.reshape(-1)
                    break
    else:
        ax = _to_np(labels_obj)
        if ax is not None:
            labels_np = ax.reshape(-1)

    if labels_np is None:
        # Fallback: attempt converting directly
        labels_np = _to_np(labels_obj)
        if labels_np is None:
            raise ValueError("Unable to extract labels for compute_metrics.")
        labels_np = labels_np.reshape(-1)

    # Align lengths if something went awry (avoid crashes; keep data consistent)
    if labels_np.shape[0] != N:
        M = min(N, labels_np.shape[0])
        labels_np = labels_np[:M]
        logits = logits[:M]

    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels_np, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels_np, preds)
    bal_acc = balanced_accuracy_score(labels_np, preds)

    # Probabilities for positive class
    probs_pos = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
    # Some splits may have a single class; guard AUC metrics
    try:
        auc = roc_auc_score(labels_np, probs_pos)
    except Exception:
        auc = float('nan')
    try:
        pr_auc = average_precision_score(labels_np, probs_pos)
    except Exception:
        pr_auc = float('nan')

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'pr_auc': pr_auc,
    }


class SiameseDataCollator:
    """Custom collator for siamese inputs that are already padded to max_length.

    Converts lists to tensors and forwards labels.
    """

    def __call__(self, features: List[Dict]):
        import torch
        batch = {}

        # Text tensors (long)
        for key in [
            'input_ids_1', 'attention_mask_1',
            'input_ids_2', 'attention_mask_2',
        ]:
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)

        # Raw text for semantic adversary targets (kept as list[str])
        if 'text1' in features[0] and 'text2' in features[0]:
            batch['text1'] = [f['text1'] for f in features]
            batch['text2'] = [f['text2'] for f in features]

        # Optional book ids as lists of strings
        if 'book1' in features[0] and 'book2' in features[0]:
            batch['book1'] = [f.get('book1', '') for f in features]
            batch['book2'] = [f.get('book2', '') for f in features]

        # Optional author ids as lists of strings (for author-level grouping)
        if 'author1' in features[0] and 'author2' in features[0]:
            batch['author1'] = [f.get('author1', '') for f in features]
            batch['author2'] = [f.get('author2', '') for f in features]

        # Labels (long)
        if 'labels' in features[0]:
            batch['labels'] = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        elif 'label' in features[0]:
            batch['labels'] = torch.tensor([f['label'] for f in features], dtype=torch.long)

        return batch


class SiameseDynamicPaddingCollator:
    """Collator that pads each side of the siamese pair to the longest length in the batch.

    Uses the provided tokenizer's .pad to build two padded tensors: one for side 1 and one for side 2.
    Keeps auxiliary fields (style/topic/text/book/labels) consistent with SiameseDataCollator.
    """

    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict]):
        import torch
        # Prepare lists for each side
        to_pad_1 = [
            {
                'input_ids': f['input_ids_1'],
                'attention_mask': f['attention_mask_1'],
            }
            for f in features
        ]
        to_pad_2 = [
            {
                'input_ids': f['input_ids_2'],
                'attention_mask': f['attention_mask_2'],
            }
            for f in features
        ]
        # Pad with tokenizer (longest in batch)
        pad_kwargs = dict(padding=True, return_tensors='pt')
        if self.pad_to_multiple_of is not None:
            pad_kwargs['pad_to_multiple_of'] = int(self.pad_to_multiple_of)
        b1 = self.tokenizer.pad(to_pad_1, **pad_kwargs)
        b2 = self.tokenizer.pad(to_pad_2, **pad_kwargs)

        batch = {
            'input_ids_1': b1['input_ids'],
            'attention_mask_1': b1['attention_mask'],
            'input_ids_2': b2['input_ids'],
            'attention_mask_2': b2['attention_mask'],
        }

        # Raw text for semantic adversary targets (list[str])
        if 'text1' in features[0] and 'text2' in features[0]:
            batch['text1'] = [f['text1'] for f in features]
            batch['text2'] = [f['text2'] for f in features]

        # Optional book ids as lists of strings
        if 'book1' in features[0] and 'book2' in features[0]:
            batch['book1'] = [f.get('book1', '') for f in features]
            batch['book2'] = [f.get('book2', '') for f in features]

        # Optional author ids as lists of strings (for author-level grouping)
        if 'author1' in features[0] and 'author2' in features[0]:
            batch['author1'] = [f.get('author1', '') for f in features]
            batch['author2'] = [f.get('author2', '') for f in features]

        # Labels
        if 'labels' in features[0]:
            batch['labels'] = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        elif 'label' in features[0]:
            batch['labels'] = torch.tensor([f['label'] for f in features], dtype=torch.long)

        return batch


def train_contrastive(
    model_name: str = 'roberta-large',
    output_dir: str = 'models/book_matcher_contrastive',
    data_dir: str = 'data/processed',
    num_epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    pooling: str = 'attn',
    use_projection: bool = True,
    grad_accum_steps: int = 2,
    weight_decay: float = 0.01,
    max_length: int = 384,
    grad_checkpointing: Optional[bool] = None,
    contrastive_mode: str = 'supcon',
    supcon_temperature: float = 0.07,  # Literature suggests 0.07-0.1 works best
    # Temperature curriculum: start easy (higher temp), anneal to target
    use_temperature_curriculum: bool = True,
    temperature_start: float = 0.15,  # Easier learning at start
    temperature_warmup_ratio: float = 0.1,  # Stay at start temp for first 10%
    temperature_anneal_ratio: float = 0.6,  # Anneal over next 60%
    # Fine-tuning: initialize from a previous checkpoint
    init_from: Optional[str] = None,  # Path to checkpoint directory to load encoder weights from
    # Topic adversary defaults
    use_topic_adversary: bool = True,
    adv_lambda: float = 0.7,  # Increased from 0.3 for stronger style disentanglement
    n_topics: int = 5,
    semantic_adversary_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    semantic_adversary_loss: str = 'cosine',
    semantic_max_length: int = 256,
    # Adversary scheduling
    adv_warmup_ratio: float = 0.1,
    adv_ramp_ratio: float = 0.3,
    grl_max_scale: float = 1.0,
    # Enhancements
    multi_head_adversary: bool = False,
    # Tokenization parallelism for HF datasets.map (None -> use all available cores)
    tokenize_workers: int | None = None,
    # Training/eval control
    eval_strategy: str = 'epoch',
    eval_steps: int = 500,
    save_strategy: str = 'epoch',
    save_steps: int = 500,
    logging_steps: int = 100,
    # Dynamic padding per-batch to reduce wasted compute
    dynamic_padding: bool = True,
    # Optional on-disk tokenized dataset cache directory
    tokenized_cache_dir: Optional[str] = None,
    # Larger eval batch for faster evaluations (multiplier of train batch)
    eval_batch_multiplier: int = 2,
    # Optional: torch.compile for extra speed (can increase startup time)
    compile_model: bool = False,
    # Checkpoint resumption for long training runs (Modal timeout handling)
    resume_from_checkpoint: bool | str | None = None,  # True=auto-detect, str=specific path
    # Legacy kwargs accepted for backward compatibility with callers
    **_kwargs,
):
    """Train the contrastive style embedding model.

    Trains a pure contrastive model (SupCon/InfoNCE) for style embedding learning.
    The model learns embeddings directly usable for cosine similarity scoring.

    Args:
        resume_from_checkpoint: If True, auto-detect latest checkpoint in output_dir.
            If str, use that specific checkpoint path. If None/False, start fresh.
    """
    # Distributed defaults; avoid over-constraining CUDA connection settings
    # Use newer TORCH_NCCL_* names (PyTorch 2.4+), fall back to old names for compatibility
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')  # Legacy fallback
    os.environ.setdefault('NCCL_DEBUG', os.environ.get('NCCL_DEBUG', 'WARN'))
    print(f"Loading datasets from {data_dir}...")
    from datasets import load_from_disk
    datasets = load_from_disk(data_dir)
    # Ensure label column name matches Trainer expectations
    if 'label' in datasets['train'].column_names:
        datasets = datasets.rename_column('label', 'labels')
    
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize datasets (DDP-aware, cached to disk to avoid duplicate work across ranks)
    import os as _os
    from datasets import load_from_disk as _load_from_disk
    # Determine DDP context
    try:
        world_size = int(_os.environ.get('WORLD_SIZE', '1'))
        rank = int(_os.environ.get('RANK', '0'))
    except Exception:
        world_size, rank = 1, 0
    # Resolve workers
    try:
        auto_workers = max(1, int(_os.cpu_count() or 1))
    except Exception:
        auto_workers = 1
    # Allow env override
    env_tok = _os.environ.get('TOKENIZE_WORKERS')
    num_workers = auto_workers if (tokenize_workers is None or int(tokenize_workers) <= 0) else int(tokenize_workers)
    if env_tok is not None:
        try:
            num_workers = max(1, int(env_tok))
        except Exception:
            pass
    # Cache path
    if tokenized_cache_dir:
        cache_dir = tokenized_cache_dir
    else:
        base = '/vol/data/tokenized' if _os.path.exists('/vol') else 'data/tokenized'
        _os.makedirs(base, exist_ok=True)
        model_slug = str(model_name).replace('/', '-').replace(':', '-')
        cache_dir = _os.path.join(base, f'{model_slug}-len{int(max_length)}-dynpad{int(bool(dynamic_padding))}')
    sentinel = _os.path.join(cache_dir, '_SUCCESS')
    # Attempt to load from cache
    tokenized_datasets = None
    if _os.path.exists(cache_dir):
        try:
            tokenized_datasets = _load_from_disk(cache_dir)
            print(f"Loaded tokenized datasets from cache: {cache_dir}")
        except Exception:
            tokenized_datasets = None
    if tokenized_datasets is None:
        if rank == 0:
            print("Tokenizing datasets...")
            print(f"Tokenization workers: {num_workers} (auto={auto_workers})")
            tokenized_datasets = datasets.map(
                lambda x: tokenize_pair(x, tokenizer, max_length=max_length, dynamic_padding=bool(dynamic_padding)),
                batched=True,
                remove_columns=[],
                num_proc=max(1, int(num_workers))
            )
            # Save to cache for other ranks/future runs
            try:
                tokenized_datasets.save_to_disk(cache_dir)
                # Write sentinel
                try:
                    with open(sentinel, 'w') as _f:
                        _f.write('ok')
                except Exception:
                    pass
                print(f"Saved tokenized datasets to {cache_dir}")
            except Exception as _e:
                print(f"Warning: failed to save tokenized datasets cache: {_e}")
        else:
            # Wait for rank 0 to finish
            import time as _time
            print(f"Rank {rank} waiting for tokenized cache at {cache_dir} ...")
            waited = 0
            while waited < 60 * 60 * 6:  # up to 6 hours
                if _os.path.exists(sentinel) or _os.path.exists(_os.path.join(cache_dir, 'dataset_info.json')):
                    break
                _time.sleep(5)
                waited += 5
            tokenized_datasets = _load_from_disk(cache_dir)
    # Leave features as lists; collator will convert to tensors
    
    # Initialize model
    model = ContrastiveBookMatcher(
        model_name,
        pooling=pooling,
        use_projection=use_projection,
        contrastive_mode=contrastive_mode,
        use_topic_adversary=use_topic_adversary,
        adv_lambda=adv_lambda,
        n_topics=n_topics,
        semantic_model_name=semantic_adversary_model,
        semantic_loss=semantic_adversary_loss,
        semantic_max_length=semantic_max_length,
        multi_head_adversary=multi_head_adversary,
        supcon_temperature=supcon_temperature,
    )
    print("Training pure contrastive style embedding model.")

    # Load weights from previous checkpoint if specified (for fine-tuning)
    if init_from and os.path.exists(init_from):
        print(f"Loading weights from checkpoint: {init_from}")
        try:
            weight_file = os.path.join(init_from, 'model.safetensors')
            if not os.path.exists(weight_file):
                weight_file = os.path.join(init_from, 'pytorch_model.bin')
            if os.path.exists(weight_file):
                if weight_file.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(weight_file)
                else:
                    state_dict = torch.load(weight_file, map_location='cpu')
                # Load with strict=False to allow new components (e.g., topic_head) to be randomly initialized
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"  Loaded checkpoint. Missing keys (will be initialized): {len(missing)}, Unexpected keys: {len(unexpected)}")
                if missing:
                    print(f"  Missing keys (first 10): {missing[:10]}")
            else:
                print(f"  Warning: No weight file found in {init_from}")
        except Exception as e:
            print(f"  Warning: Failed to load checkpoint: {e}")
    # Reduce memory footprint where possible
    # Prefer non-reentrant gradient checkpointing on multi-GPU (safe with shared encoder)
    try:
        _ws_env = int(os.environ.get('WORLD_SIZE', '1'))
    except Exception:
        _ws_env = 1
    if grad_checkpointing is None:
        # Default: off for multi-GPU (throughput), on for single-GPU (memory saver)
        grad_ckpt_effective = (_ws_env <= 1)
    else:
        grad_ckpt_effective = bool(grad_checkpointing)

    try:
        # Disable cache for training
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'config'):
            setattr(model.encoder.config, 'use_cache', False)
            # Prefer FlashAttention v2 when available; else fall back to SDPA for speed
            try:
                # If flash-attn is importable and we're on CUDA, explicitly select FA2
                if torch.cuda.is_available():
                    try:
                        import flash_attn  # type: ignore # noqa: F401
                        if hasattr(model.encoder.config, 'attn_implementation'):
                            setattr(model.encoder.config, 'attn_implementation', 'flash_attention_2')
                    except Exception:
                        # Otherwise, ensure SDPA is used over eager
                        if hasattr(model.encoder.config, 'attn_implementation'):
                            cur = getattr(model.encoder.config, 'attn_implementation', None)
                            if cur not in ('flash_attention_2', 'sdpa'):
                                setattr(model.encoder.config, 'attn_implementation', 'sdpa')
                else:
                    if hasattr(model.encoder.config, 'attn_implementation'):
                        cur = getattr(model.encoder.config, 'attn_implementation', None)
                        if cur not in ('sdpa',):
                            setattr(model.encoder.config, 'attn_implementation', 'sdpa')
            except Exception:
                pass
        # Enable gradient checkpointing on the encoder (non-reentrant when supported)
        if grad_ckpt_effective and hasattr(model, 'encoder') and hasattr(model.encoder, 'gradient_checkpointing_enable'):
            try:
                # Available in recent transformers/torch
                model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                if _ws_env > 1:
                    print(f"Enabled non-reentrant gradient checkpointing (WORLD_SIZE={_ws_env}).")
            except TypeError:
                # Older versions: fall back. Reentrant can break under DDP with a shared module.
                if _ws_env > 1:
                    model.encoder.gradient_checkpointing_disable()
                    grad_ckpt_effective = False
                    print(f"Disabled gradient checkpointing for multi-GPU DDP (WORLD_SIZE={_ws_env}); non-reentrant not supported.")
                else:
                    model.encoder.gradient_checkpointing_enable()
            except Exception as _e:
                if _ws_env > 1:
                    model.encoder.gradient_checkpointing_disable()
                    grad_ckpt_effective = False
                    print(f"Gradient checkpointing setup failed under DDP: {_e}. Disabled for safety.")
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'gradient_checkpointing_disable'):
            model.encoder.gradient_checkpointing_disable()
    except Exception:
        pass
    
    # Speed settings on Ampere/Hopper GPUs
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # Optional compilation (can improve throughput 5-20%; increases warmup)
    if compile_model and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Enabled torch.compile(mode='reduce-overhead')")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    # Training arguments
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # Important: set gradient_checkpointing=False here because our model is a custom nn.Module
    # and HF Trainer will try to call model.gradient_checkpointing_enable() if this is True.
    # We already handle enabling checkpointing directly on the encoder above when requested.
    # Adapt dataloader workers to world size to avoid CPU oversubscription in DDP
    try:
        cpu_ct = max(1, int(_os.cpu_count() or 1))
    except Exception:
        cpu_ct = 8
    dl_workers = max(2, cpu_ct // max(1, world_size))
    scaled_lr = float(learning_rate)

    # Ensure HF Trainer picks up torchrun distributed launch only when we truly run DDP
    try:
        _local_rank_env = int(os.environ.get('LOCAL_RANK', '-1'))
    except Exception:
        _local_rank_env = -1
    try:
        _ws_env_for_ddp = int(os.environ.get('WORLD_SIZE', '1'))
    except Exception:
        _ws_env_for_ddp = 1
    _ddp_enabled = (_ws_env_for_ddp > 1) or (_local_rank_env >= 0)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(batch_size * max(1, int(eval_batch_multiplier)), 32),
        learning_rate=scaled_lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
        logging_dir=f'{output_dir}/logs',
        logging_steps=int(logging_steps),
        eval_strategy=str(eval_strategy),
        eval_steps=int(eval_steps),
        save_strategy=str(save_strategy),
        save_steps=int(save_steps),
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        prediction_loss_only=True,
        report_to='none',
        fp16=(torch.cuda.is_available() and not bf16_ok),
        bf16=bf16_ok,
        dataloader_num_workers=int(dl_workers),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=False,
        # Distributed config (disable unless launched via torchrun)
        local_rank=(_local_rank_env if _ddp_enabled else -1),
        # DDP settings: no unused params detected in forward; disable extra autograd traversal.
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        ddp_backend=("nccl" if _ddp_enabled else None),
        ddp_bucket_cap_mb=100,
        optim=("adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"),
    )
    
    # Create trainer
    eval_ds = tokenized_datasets['validation']

    # Prefer padding to multiples of 16 on GPU for better kernel efficiency
    _pad_mul = 16 if (dynamic_padding and torch.cuda.is_available()) else 8
    collator = SiameseDynamicPaddingCollator(tokenizer, pad_to_multiple_of=_pad_mul) if dynamic_padding else SiameseDataCollator()
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            AdversarySchedulerCallback(
                model,
                warmup_ratio=adv_warmup_ratio,
                ramp_ratio=adv_ramp_ratio,
                max_adv_lambda=adv_lambda,
                max_grl_scale=grl_max_scale,
            ),
            PerfLoggerCallback(print_every=max(10, int(logging_steps))),
        ] + ([TemperatureSchedulerCallback(
            model,
            start_temperature=temperature_start,
            target_temperature=supcon_temperature,
            warmup_ratio=temperature_warmup_ratio,
            anneal_ratio=temperature_anneal_ratio,
        )] if use_temperature_curriculum else []),
    )
    # Ensure only the primary 'labels' are used for metrics/label_ids
    try:
        setattr(trainer, 'label_names', ['labels'])
    except Exception:
        pass
    
    # Resolve checkpoint for resumption
    _resume_ckpt = None
    if resume_from_checkpoint is True:
        # Auto-detect latest checkpoint in output_dir
        import glob as _glob
        ckpt_dirs = sorted(_glob.glob(os.path.join(output_dir, 'checkpoint-*')), key=os.path.getmtime)
        if ckpt_dirs:
            _resume_ckpt = ckpt_dirs[-1]
            print(f"Auto-detected checkpoint for resumption: {_resume_ckpt}")
        else:
            print("No checkpoint found in output_dir, starting fresh.")
    elif isinstance(resume_from_checkpoint, str) and resume_from_checkpoint:
        _resume_ckpt = resume_from_checkpoint
        print(f"Resuming from specified checkpoint: {_resume_ckpt}")

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=_resume_ckpt)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets['test'])
    print(f"Test results: {test_results}")

    # Save final model
    trainer.save_model(f'{output_dir}/final')
    tokenizer.save_pretrained(f'{output_dir}/final')
    print(f"\nModel saved to {output_dir}/final")
    
    return trainer, test_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train contrastive style embedding model')
    parser.add_argument('--model', type=str, default='roberta-large', help='Base model name')
    parser.add_argument('--output', type=str, default='models/book_matcher_contrastive',
                       help='Output directory')
    parser.add_argument('--data', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--epochs', type=int, default=6, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--pooling', type=str, default='attn', choices=['mean','cls','attn'], help='Pooling strategy for sequence embeddings')
    parser.add_argument('--no-projection', action='store_true', help='Disable projection head')
    parser.add_argument('--grad-accum', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--grad-ckpt', action='store_true', help='Enable gradient checkpointing (save memory, slower)')
    parser.add_argument('--no-grad-ckpt', action='store_true', help='Disable gradient checkpointing (maximize tokens/s)')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='AdamW weight decay')
    parser.add_argument('--contrastive-mode', type=str, default='supcon', choices=['supcon','infonce'], help='Type of contrastive objective')
    parser.add_argument('--supcon-temperature', type=float, default=0.07, help='Temperature for SupCon/InfoNCE (0.07-0.1 recommended)')
    # Temperature curriculum
    parser.add_argument('--no-temperature-curriculum', action='store_true', help='Disable temperature annealing curriculum')
    parser.add_argument('--temperature-start', type=float, default=0.15, help='Starting temperature for curriculum (higher=easier)')
    parser.add_argument('--temperature-warmup-ratio', type=float, default=0.1, help='Fraction of training to stay at start temperature')
    parser.add_argument('--temperature-anneal-ratio', type=float, default=0.6, help='Fraction of training to anneal to target temperature')
    parser.add_argument('--init-from', type=str, default=None, help='Path to checkpoint directory to initialize weights from (for fine-tuning)')
    # Topic adversary & scheduler
    parser.add_argument('--no-topic-adversary', action='store_true', help='Disable topic adversary (GRL + topic head)')
    parser.add_argument('--adv-lambda', type=float, default=0.4, help='Max weight for adversarial topic loss (will be scheduled)')
    parser.add_argument('--adv-warmup-ratio', type=float, default=0.1, help='Fraction of total steps with adversary off (0->no warmup)')
    parser.add_argument('--adv-ramp-ratio', type=float, default=0.3, help='Fraction of total steps to ramp adversary to max')
    parser.add_argument('--grl-max-scale', type=float, default=1.0, help='Maximum gradient reversal scale applied to embeddings')
    parser.add_argument('--multi-head-adversary', action='store_true', help='Add a second topic head on pre-projection pooled embeddings')
    parser.add_argument('--semantic-adversary-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Frozen encoder used for adversarial regression targets')
    parser.add_argument('--semantic-adversary-loss', type=str, default='cosine', choices=['cosine','mse'], help='Adversarial regression loss: cosine or MSE')
    parser.add_argument('--semantic-max-length', type=int, default=256, help='Max tokens for semantic adversary tokenizer')
    # Efficiency knobs (safe): torch.compile and max length
    parser.add_argument('--compile', dest='compile_model', action='store_true', help='Enable torch.compile for faster training (no quality change)')
    parser.add_argument('--max-length', type=int, default=384, help='Max tokens per side (affects speed but not model capacity)')
    # Eval throughput control
    parser.add_argument('--eval-batch-multiplier', type=int, default=2, help='Multiplier of train batch for per-device eval batch size')
    # Checkpoint resumption for long training / Modal timeout handling
    parser.add_argument('--resume', action='store_true',
                       help='Auto-detect and resume from latest checkpoint in output_dir. '
                            'Use this with Modal retries for training runs that exceed 24h.')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from a specific checkpoint path.')

    args = parser.parse_args()

    train_contrastive(
        model_name=args.model,
        output_dir=args.output,
        data_dir=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        pooling=args.pooling,
        use_projection=(not args.no_projection),
        grad_accum_steps=args.grad_accum,
        grad_checkpointing=(True if args.grad_ckpt else (False if args.no_grad_ckpt else None)),
        weight_decay=args.weight_decay,
        contrastive_mode=args.contrastive_mode,
        supcon_temperature=args.supcon_temperature,
        use_temperature_curriculum=(not args.no_temperature_curriculum),
        temperature_start=args.temperature_start,
        temperature_warmup_ratio=args.temperature_warmup_ratio,
        temperature_anneal_ratio=args.temperature_anneal_ratio,
        init_from=getattr(args, 'init_from', None),
        use_topic_adversary=(not args.no_topic_adversary),
        adv_lambda=args.adv_lambda,
        semantic_adversary_model=args.semantic_adversary_model,
        semantic_adversary_loss=args.semantic_adversary_loss,
        semantic_max_length=args.semantic_max_length,
        adv_warmup_ratio=args.adv_warmup_ratio,
        adv_ramp_ratio=args.adv_ramp_ratio,
        grl_max_scale=args.grl_max_scale,
        multi_head_adversary=args.multi_head_adversary,
        compile_model=getattr(args, 'compile_model', False),
        eval_batch_multiplier=int(getattr(args, 'eval_batch_multiplier', 2)),
        resume_from_checkpoint=(args.resume_from if args.resume_from else args.resume),
    )
