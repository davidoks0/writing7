"""
Train a contrastive RoBERTa-based model for book-text matching.

Implements:
- Siamese architecture with twin encoders
- Contrastive loss for style similarity learning
- Style feature engineering (type-token ratio, sentence length, punctuation patterns)
- Longer context support (512 tokens + hierarchical pooling if needed)
"""
import os
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, fbeta_score, balanced_accuracy_score, average_precision_score
import numpy as np


class StyleFeatures(nn.Module):
    """Extract style features from text."""
    
    def __init__(self, vocab_size: int = 50265):
        super().__init__()
        self.vocab_size = vocab_size
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract style features from tokenized text.
        
        Features:
        - Type-token ratio proxy (unique tokens / total tokens)
        - Average sentence length
        - Punctuation frequency
        - Word formality indicators
        """
        batch_size = input_ids.size(0)
        
        # Placeholder tensors for features
        features = []
        
        for i in range(batch_size):
            ids = input_ids[i]
            mask = attention_mask[i]
            
            # Count unique tokens (type-token ratio proxy)
            unique = torch.unique(ids[mask.bool()])
            type_token_ratio = len(unique) / mask.sum().item()
            
            # Count punctuation tokens (somewhat proxy for punctuation frequency)
            # Punctuation token IDs are typically > 50000 in BPE vocabularies
            punct_mask = ids > 50000
            punct_ratio = punct_mask.sum().item() / mask.sum().item()
            
            # Average token sequence length (proxy for sentence complexity)
            avg_length = mask.sum().item() / batch_size if batch_size > 0 else 0
            
            features.append([type_token_ratio, punct_ratio, avg_length])
        
        return torch.tensor(features, device=input_ids.device, dtype=torch.float32)


def extract_style_features_batch(texts: List[str]) -> np.ndarray:
    """Extract compact style features from raw text.

    Returns 3 features per text: [type_token_ratio, punct_ratio, avg_sentence_len].
    """
    features = []

    for text in texts:
        # Type-token ratio
        words = text.lower().split()
        unique_words = len(set(words))
        type_token_ratio = unique_words / len(words) if words else 0.0

        # Punctuation frequency
        punct_chars = sum(1 for c in text if c in '.,;:!?"()-[]{}')
        punct_ratio = (punct_chars / len(text)) if text else 0.0

        # Average sentence length (in words)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_sentence_len = float(np.mean(sentence_lengths))
        else:
            avg_sentence_len = 0.0

        features.append([type_token_ratio, punct_ratio, avg_sentence_len])

    return np.array(features)


# ---------------------- Topic heuristics (weak labels) ----------------------

_TOPIC_VOCAB = [
    'religious',
    'historical',
    'adventure',
    'romance',
    'general',
]

_TOPIC_KEYWORDS = {
    'religious': set(['god', 'church', 'prayer', 'scripture', 'lord', 'faith', 'holy', 'sacred', 'divine']),
    'historical': set(['prince', 'king', 'castle', 'knight', 'lord', 'duke', 'empire', 'queen', 'court']),
    'adventure': set(['captain', 'ship', 'sea', 'voyage', 'island', 'expedition', 'jungle', 'desert', 'treasure']),
    'romance': set(['love', 'heart', 'soul', 'passion', 'beloved', 'darling', 'kiss', 'romance', 'marriage']),
}


def _label_topic(text: str) -> int:
    """Predict a coarse topic for a chunk via simple keyword counts.

    Returns an integer ID in [0, n_topics-1] matching _TOPIC_VOCAB order.
    Defaults to 'general' when no topic dominates.
    """
    tl = text.lower()
    scores = {k: 0 for k in _TOPIC_KEYWORDS.keys()}
    for topic, keys in _TOPIC_KEYWORDS.items():
        # Count keyword occurrences (very rough)
        scores[topic] = sum(tl.count(w) for w in keys)
    # Choose best non-zero; else general
    if scores and max(scores.values()) > 0:
        best = max(scores.items(), key=lambda kv: kv[1])[0]
    else:
        best = 'general'
    return _TOPIC_VOCAB.index(best)


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


class ArcMarginProduct(nn.Module):
    """ArcFace margin product for classification on normalized features."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.2, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cos = F.linear(x, W)
        if labels is None:
            return cos * self.s
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sin = torch.sqrt(torch.clamp(1.0 - torch.pow(cos, 2), min=1e-9))
        phi = cos * self.cos_m - sin * self.sin_m
        if not self.easy_margin:
            phi = torch.where(cos > self.th, phi, cos - self.mm)
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cos
        output *= self.s
        return output


class ContrastiveBookMatcher(nn.Module):
    """
    Siamese model for book-text matching with style features.
    
    Architecture:
    - Twin RoBERTa encoders (shared weights)
    - Style feature extraction
    - Concatenation of embeddings + style features
    - Classification head
    """
    
    def __init__(
        self,
        model_name: str = 'roberta-base',
        use_style_features: bool = True,
        use_symmetric_features: bool = True,
        contrastive_weight: float = 0.1,
        class_weights: torch.Tensor | None = None,
        pooling: str = 'attn',
        use_projection: bool = True,
        label_smoothing: float = 0.05,
        classifier: str = 'arcface',  # 'mlp' or 'arcface'
        arcface_margin: float = 0.2,
        arcface_scale: float = 30.0,
        contrastive_mode: str = 'supcon',  # 'supcon' or 'infonce'
        # Topic adversary (default ON)
        use_topic_adversary: bool = True,
        adv_lambda: float = 0.2,
        n_topics: int = 5,
        # Enhancements
        multi_head_adversary: bool = False,
        use_independence_penalty: bool = False,
        independence_weight: float = 0.0,
        indep_kernel: str = 'rbf',
        supcon_temperature: float = 1.0,
    ):
        super().__init__()
        self.use_style_features = use_style_features
        self.use_symmetric_features = use_symmetric_features
        self.contrastive_weight = float(contrastive_weight)
        self.pooling_type = pooling
        self.use_projection = bool(use_projection)
        self.label_smoothing = float(label_smoothing)
        self.classifier_type = classifier
        self.arcface_margin = float(arcface_margin)
        self.arcface_scale = float(arcface_scale)
        self.contrastive_mode = contrastive_mode
        self.use_topic_adversary = bool(use_topic_adversary)
        self.multi_head_adversary = bool(multi_head_adversary)
        self.use_independence_penalty = bool(use_independence_penalty)
        self.independence_weight = float(independence_weight)
        self.indep_kernel = indep_kernel
        self.adv_lambda = float(adv_lambda)
        self.n_topics = int(n_topics)
        # GRL scale is runtime-scheduled; start at 0 (no adversary) and ramp up via callback
        self.grl_scale: float = 0.0

        # Load base model
        config = AutoConfig.from_pretrained(model_name)
        # Load without pooling layer (we pool manually)
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        
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

        # Style feature extractor
        if use_style_features:
            self.style_extractor = StyleFeatures()
            self.style_dim = 3  # per-side features: type-token, punct, avg_length
        else:
            self.style_dim = 0
        
        # Classification head
        # Input: [h1, h2, |h1-h2|, h1*h2] (+ style features) if symmetric; else [h1, h2] (+ style)
        base_dim = hidden_dim * 2
        sym_dim = hidden_dim * 2 if use_symmetric_features else 0
        input_dim = base_dim + sym_dim + (self.style_dim * 2 if self.style_dim else 0)
        if self.classifier_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 2)
            )
            self.feat_head = None
            self.arcface = None
        else:
            self.feat_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
            )
            self.arcface = ArcMarginProduct(hidden_dim, 2, s=self.arcface_scale, m=self.arcface_margin)
            self.classifier = None

        # Topic adversarial head (shared for both sides)
        if self.use_topic_adversary:
            self.topic_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, self.n_topics),
            )
            self.topic_head_pre = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, self.n_topics),
            ) if self.multi_head_adversary else None
        else:
            self.topic_head = None
            self.topic_head_pre = None
        
        # Temperature for contrastive learning
        self.temperature = nn.Parameter(torch.tensor([float(supcon_temperature)]))
        
        # Optional class weights for CE loss
        if class_weights is not None:
            # register as buffer to move with .to(device)
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None
        
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, 
                style_features_1=None, style_features_2=None, labels=None,
                topic_labels_1=None, topic_labels_2=None,
                book_ids_1: Optional[torch.Tensor] = None,
                book_ids_2: Optional[torch.Tensor] = None):
        """
        Forward pass for Siamese architecture.
        
        Args:
            input_ids_1, input_ids_2: Tokenized text pairs
            attention_mask_1, attention_mask_2: Attention masks
            style_features_1, style_features_2: Optional style features
            labels: Ground truth labels
        
        Returns:
            loss, logits
        """
        # Encode both texts
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
        
        # Concatenate embeddings (with symmetric features if enabled)
        pieces = [emb1_pooled, emb2_pooled]
        if self.use_symmetric_features:
            diff = torch.abs(emb1_pooled - emb2_pooled)
            prod = emb1_pooled * emb2_pooled
            pieces.extend([diff, prod])
        combined = torch.cat(pieces, dim=1)
        
        # Add style features if using them
        if self.use_style_features and style_features_1 is not None:
            style_concat = torch.cat([style_features_1, style_features_2], dim=1)
            combined = torch.cat([combined, style_concat], dim=1)
        
        # Classify
        if self.classifier_type == 'mlp':
            logits = self.classifier(combined)
        else:
            feat = self.feat_head(combined)
            # Apply ArcFace margin only during training; at eval/inference use plain cosine logits
            logits = self.arcface(feat, labels if self.training else None)

        # Topic adversarial logits (computed even at eval, but loss only in training)
        topic_logits_1 = None
        topic_logits_2 = None
        topic_logits_1_pre = None
        topic_logits_2_pre = None
        if self.use_topic_adversary and self.topic_head is not None:
            # Gradient reversal only matters in training; autograd disabled at eval
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
            # Classification loss
            cw = self.class_weights.to(logits.device) if isinstance(self.class_weights, torch.Tensor) else None
            ce_loss = F.cross_entropy(logits, labels, weight=cw, label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0)
            
            # Supervised contrastive (SupCon) or InfoNCE with in-batch negatives (only for positive pairs)
            # Normalize embeddings for cosine similarity
            emb1_norm = F.normalize(emb1_pooled, p=2, dim=1)
            emb2_norm = F.normalize(emb2_pooled, p=2, dim=1)

            info_nce_loss = torch.tensor(0.0, device=logits.device)
            pos_idx = (labels == 1).nonzero(as_tuple=False).squeeze(-1)
            if pos_idx.numel() > 0:
                if self.contrastive_mode == 'supcon':
                    # Build a 2B x 2B similarity matrix; allow multi-positive via book_ids
                    Z = torch.cat([emb1_norm, emb2_norm], dim=0)
                    sim = Z @ Z.T / self.temperature
                    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
                    sim = sim.masked_fill(eye, -1e9)
                    B = emb1_norm.size(0)
                    if book_ids_1 is not None and book_ids_2 is not None:
                        books = torch.cat([book_ids_1, book_ids_2], dim=0)
                        logits_all = sim
                        P = (books.unsqueeze(0) == books.unsqueeze(1)) & (~eye)
                        pos_logits = logits_all.masked_fill(~P, -1e9)
                        num = torch.logsumexp(pos_logits, dim=1)
                        den = torch.logsumexp(logits_all, dim=1)
                        info_nce_loss = (den - num).mean()
                    else:
                        anchors = torch.cat([pos_idx, pos_idx + B], dim=0)
                        positives = torch.cat([pos_idx + B, pos_idx], dim=0)
                        sim_rows = sim[anchors]
                        targets = positives
                        info_nce_loss = F.cross_entropy(sim_rows, targets)
                else:
                    # Similarity matrices (B x B)
                    sim12 = emb1_norm @ emb2_norm.T / self.temperature
                    sim21 = emb2_norm @ emb1_norm.T / self.temperature
                    targets = torch.arange(sim12.size(1), device=logits.device)
                    loss12 = F.cross_entropy(sim12[pos_idx], targets[pos_idx])
                    loss21 = F.cross_entropy(sim21[pos_idx], targets[pos_idx])
                    info_nce_loss = 0.5 * (loss12 + loss21)

            # Topic adversarial loss (encourages topic-invariance via GRL)
            adv_loss = torch.tensor(0.0, device=logits.device)
            if self.use_topic_adversary and topic_logits_1 is not None and topic_logits_2 is not None and topic_labels_1 is not None and topic_labels_2 is not None:
                adv_terms = [
                    0.5 * (
                        F.cross_entropy(topic_logits_1, topic_labels_1) +
                        F.cross_entropy(topic_logits_2, topic_labels_2)
                    )
                ]
                if self.multi_head_adversary and topic_logits_1_pre is not None and topic_logits_2_pre is not None:
                    adv_terms.append(
                        0.5 * (
                            F.cross_entropy(topic_logits_1_pre, topic_labels_1) +
                            F.cross_entropy(topic_logits_2_pre, topic_labels_2)
                        )
                    )
                adv_loss = torch.stack(adv_terms).mean()

            # Independence penalty (HSIC)
            indep_loss = torch.tensor(0.0, device=logits.device)
            if self.use_independence_penalty and topic_labels_1 is not None and topic_labels_2 is not None:
                indep_loss = 0.5 * (self._hsic_penalty(emb1_norm, topic_labels_1) + self._hsic_penalty(emb2_norm, topic_labels_2))

            # Combine losses
            loss = ce_loss + self.contrastive_weight * info_nce_loss + self.adv_lambda * adv_loss + self.independence_weight * indep_loss

        return {'loss': loss, 'logits': logits, 'topic_logits_1': topic_logits_1, 'topic_logits_2': topic_logits_2}

    def _pairwise_sq_dists(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise squared distances for rows of X (B,D) -> (B,B)."""
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        x2 = (X * X).sum(dim=1, keepdim=True)
        dist2 = x2 + x2.T - 2.0 * (X @ X.T)
        dist2 = torch.clamp(dist2, min=0.0)
        return dist2

    def _rbf_kernel(self, X: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
        """RBF kernel with median heuristic if gamma is None."""
        d2 = self._pairwise_sq_dists(X).detach() if not X.requires_grad else self._pairwise_sq_dists(X)
        if gamma is None:
            # Median heuristic on detached distances to stabilize
            with torch.no_grad():
                vals = d2.flatten()
                med = torch.median(vals[vals > 0]) if (vals > 0).any() else torch.tensor(1.0, device=X.device)
                # gamma = 1 / (2 sigma^2); use sigma^2 = median
                gamma = float(1.0 / (2.0 * (med.item() + 1e-6)))
        K = torch.exp(-float(gamma) * d2)
        return K

    def _delta_kernel_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Simple label kernel: 1 if same class else 0."""
        y = y.view(-1, 1)
        return (y == y.T).float()

    def _center_kernel(self, K: torch.Tensor) -> torch.Tensor:
        n = K.size(0)
        one = torch.ones((n, n), device=K.device, dtype=K.dtype) / n
        return K - one @ K - K @ one + one @ K @ one

    def _hsic_penalty(self, Z: torch.Tensor, topic_labels: torch.Tensor) -> torch.Tensor:
        """Biased HSIC between embeddings Z (B,D) and topic labels (B,) using RBF on Z and delta on labels.

        Returns a non-negative scalar; larger means more dependence. Differentiable w.r.t. Z.
        """
        if Z.dim() != 2 or topic_labels.dim() != 1 or Z.size(0) != topic_labels.size(0):
            return torch.tensor(0.0, device=Z.device)
        # Kernels
        if self.indep_kernel == 'rbf':
            K = self._rbf_kernel(Z)
        else:
            # Linear kernel fallback
            K = Z @ Z.T
        L = self._delta_kernel_labels(topic_labels.detach())
        # Center
        Kc = self._center_kernel(K)
        Lc = self._center_kernel(L)
        # HSIC (biased): (1/n^2) trace(Kc Lc)
        n = Z.size(0)
        hsic = torch.trace(Kc @ Lc) / (n * n + 1e-6)
        return hsic
    
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
    """Custom Trainer for contrastive learning with style features."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with optional contrastive component.

        Accepts `num_items_in_batch` for compatibility with recent Transformers versions.
        """
        labels = inputs.pop('labels')
        # Drop any raw text fields if present (not used in base trainer)
        inputs.pop('text1', None)
        inputs.pop('text2', None)
        # Optional hashed book ids for multi-positive SupCon
        b1 = inputs.pop('book1', None)
        b2 = inputs.pop('book2', None)
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
        book_ids_1 = _hash_ids(b1)
        book_ids_2 = _hash_ids(b2)
        outputs = model(**inputs, labels=labels, book_ids_1=book_ids_1, book_ids_2=book_ids_2)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss


class DistillContrastiveTrainer(Trainer):
    """Trainer that adds cross-encoder distillation to the contrastive loss."""

    def __init__(self, *args, teacher_model=None, teacher_tokenizer=None, distill_weight: float = 0.5, distill_temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.distill_weight = float(distill_weight)
        self.distill_temperature = float(distill_temperature)
        if self.teacher_model is not None:
            self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        texts1 = inputs.pop('text1', None)
        texts2 = inputs.pop('text2', None)
        # Optional hashed book ids for multi-positive SupCon
        b1 = inputs.pop('book1', None)
        b2 = inputs.pop('book2', None)
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
        book_ids_1 = _hash_ids(b1)
        book_ids_2 = _hash_ids(b2)
        outputs = model(**inputs, labels=labels, book_ids_1=book_ids_1, book_ids_2=book_ids_2)
        loss = outputs['loss']

        if self.teacher_model is not None and texts1 is not None and texts2 is not None:
            with torch.no_grad():
                batch = self.teacher_tokenizer(
                    texts1,
                    texts2,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                # Run teacher on its own device (often CPU) to avoid extra GPU memory
                t_device = next(self.teacher_model.parameters()).device
                batch_t = {k: v.to(t_device) for k, v in batch.items()}
                # Use autocast on GPU to speed up teacher inference
                if t_device.type == 'cuda':
                    # Infer dtype from student logits after computing outputs
                    s_logits_peek = model(**{**inputs, 'labels': labels})['logits'] if False else None
                    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        t_logits = self.teacher_model(**batch_t).logits
                else:
                    t_logits = self.teacher_model(**batch_t).logits
            T = self.distill_temperature
            s_logits = outputs['logits']
            # Ensure both tensors share device and dtype for KL-div
            t_logits = t_logits.to(device=s_logits.device, dtype=s_logits.dtype)
            distill_loss = F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction='batchmean'
            ) * (T * T)
            loss = loss + self.distill_weight * distill_loss

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
        max_adv_lambda: float = 0.2,
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


class ContrastiveWeightSchedulerCallback(TrainerCallback):
    """Linearly ramp the contrastive loss weight from its initial value to a final target.

    This approximates a curriculum where in-batch negatives become increasingly emphasized.
    """
    def __init__(self, model: ContrastiveBookMatcher, final_weight: float = 0.5, warmup_ratio: float = 0.0, ramp_ratio: float = 0.3) -> None:
        super().__init__()
        self.model = model
        self.final = float(final_weight)
        self.warmup_ratio = max(0.0, float(warmup_ratio))
        self.ramp_ratio = max(0.0, float(ramp_ratio))
        self._max_steps: int | None = None
        self._init_weight: float | None = None

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
        self._init_weight = float(getattr(self.model, 'contrastive_weight', 0.0))

    def on_step_begin(self, args, state, control, **kwargs):
        if self._init_weight is None:
            return
        s = self._scale(state.global_step)
        target = self._init_weight + (self.final - self._init_weight) * s
        self.model.contrastive_weight = float(target)


def tokenize_pair(examples, tokenizer, max_length: int = 512):
    """Tokenize pairs of texts separately for Siamese architecture."""
    # Tokenize both texts
    encoded1 = tokenizer(
        examples['text1'],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    
    encoded2 = tokenizer(
        examples['text2'],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    
    # Extract style features from raw text
    style1 = extract_style_features_batch(examples['text1'])
    style2 = extract_style_features_batch(examples['text2'])
    
    # Weak topic labels per chunk (int IDs)
    topics1 = [_label_topic(t) for t in examples['text1']]
    topics2 = [_label_topic(t) for t in examples['text2']]
    
    return {
        'input_ids_1': encoded1['input_ids'],
        'attention_mask_1': encoded1['attention_mask'],
        'input_ids_2': encoded2['input_ids'],
        'attention_mask_2': encoded2['attention_mask'],
        'style_features_1': style1.tolist(),
        'style_features_2': style2.tolist(),
        'topic1': topics1,
        'topic2': topics2,
        # Optional: book ids
        'book1': (examples['book1'] if 'book1' in examples else [''] * len(examples['text1'])),
        'book2': (examples['book2'] if 'book2' in examples else [''] * len(examples['text2'])),
        # Keep raw text for distillation with a cross-encoder teacher
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


def _optimize_temperature(logits: np.ndarray, labels: np.ndarray, init_temp: float = 1.0) -> float:
    """Fit a single temperature on validation logits to minimize NLL."""
    import torch
    t = torch.tensor([init_temp], requires_grad=True)
    optimizer = torch.optim.LBFGS([t], lr=0.01, max_iter=50)
    x = torch.from_numpy(logits)
    y = torch.from_numpy(labels).long()

    def closure():
        optimizer.zero_grad()
        scaled = x / t.clamp(min=1e-3)
        loss = F.cross_entropy(scaled, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(t.clamp(min=1e-3).item())


def _search_best_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float = 1.0,
    metric: str = 'f1',
    target_acc: float | None = None,
    target_recall: float | None = None,
) -> tuple[float, dict]:
    """Grid-search the probability threshold that maximizes a target metric on validation data.

    metric: one of {'f1','accuracy','balanced_accuracy','f0.5','f2'}
    Returns (best_threshold, best_row_metrics)
    """
    import numpy as np
    scaled = logits / max(temperature, 1e-6)
    probs = torch.softmax(torch.from_numpy(scaled), dim=1).numpy()[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        acc = accuracy_score(labels, preds)
        bal_acc = balanced_accuracy_score(labels, preds)
        f05 = fbeta_score(labels, preds, beta=0.5, zero_division=0)
        f2 = fbeta_score(labels, preds, beta=2.0, zero_division=0)
        rows.append({
            'threshold': float(thr),
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'f0.5': float(f05),
            'f2': float(f2),
        })
    def score_row(r: dict) -> float:
        if metric == 'accuracy':
            return r['accuracy']
        if metric == 'balanced_accuracy':
            return r['balanced_accuracy']
        if metric == 'f0.5':
            return r['f0.5']
        if metric == 'f2':
            return r['f2']
        return r['f1']
    feasible = [r for r in rows if (target_acc is None or r['accuracy'] >= target_acc) and (target_recall is None or r['recall'] >= target_recall)]
    candidates = feasible if feasible else rows
    best = max(candidates, key=score_row)
    return best['threshold'], best


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

        # Style features (float)
        for key in ['style_features_1', 'style_features_2']:
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.float32)

        # Topic labels (long), if present
        if 'topic1' in features[0] and 'topic2' in features[0]:
            batch['topic_labels_1'] = torch.tensor([f['topic1'] for f in features], dtype=torch.long)
            batch['topic_labels_2'] = torch.tensor([f['topic2'] for f in features], dtype=torch.long)

        # Raw text for distillation (kept as list[str])
        if 'text1' in features[0] and 'text2' in features[0]:
            batch['text1'] = [f['text1'] for f in features]
            batch['text2'] = [f['text2'] for f in features]

        # Optional book ids as lists of strings
        if 'book1' in features[0] and 'book2' in features[0]:
            batch['book1'] = [f.get('book1', '') for f in features]
            batch['book2'] = [f.get('book2', '') for f in features]

        # Labels (long)
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
    use_style_features: bool = True,
    use_symmetric_features: bool = True,
    contrastive_weight: float = 0.3,
    teacher_model_dir: Optional[str] = None,
    distill_weight: float = 0.5,
    distill_temperature: float = 3.0,
    calibrate_for: str = 'accuracy',
    target_acc: Optional[float] = None,
    target_recall: Optional[float] = 0.85,
    pooling: str = 'attn',
    use_projection: bool = True,
    label_smoothing: float = 0.03,
    grad_accum_steps: int = 2,
    weight_decay: float = 0.01,
    select_metric: str = 'balanced_accuracy',
    max_length: int = 512,
    grad_checkpointing: bool = True,
    teacher_on_gpu: bool | None = None,
    classifier: str = 'arcface',
    arcface_margin: float = 0.25,
    arcface_scale: float = 30.0,
    contrastive_mode: str = 'supcon',
    supcon_temperature: float = 1.0,
    # Topic adversary defaults
    use_topic_adversary: bool = True,
    adv_lambda: float = 0.2,
    n_topics: int = 5,
    # Adversary scheduling
    adv_warmup_ratio: float = 0.1,
    adv_ramp_ratio: float = 0.3,
    grl_max_scale: float = 1.0,
    # Enhancements
    multi_head_adversary: bool = False,
    use_independence_penalty: bool = False,
    independence_weight: float = 0.0,
):
    """Train the contrastive model."""
    print(f"Loading datasets from {data_dir}...")
    from datasets import load_from_disk
    datasets = load_from_disk(data_dir)
    # Ensure label column name matches Trainer expectations
    if 'label' in datasets['train'].column_names:
        datasets = datasets.rename_column('label', 'labels')
    
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = datasets.map(
        lambda x: tokenize_pair(x, tokenizer, max_length=max_length),
        batched=True,
        remove_columns=[]
    )
    # Leave features as lists; collator will convert to tensors
    
    # Compute class weights from training labels if dataset is imbalanced
    try:
        train_labels = tokenized_datasets['train']['labels']
        pos = sum(1 for y in train_labels if int(y) == 1)
        neg = sum(1 for y in train_labels if int(y) == 0)
        if pos > 0 and neg > 0:
            # weight for classes [neg, pos] so minority gets larger weight
            w_neg = pos / (pos + neg)
            w_pos = neg / (pos + neg)
            class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)
        else:
            class_weights = None
    except Exception:
        class_weights = None
    
    # Initialize model after computing weights
    model = ContrastiveBookMatcher(
        model_name,
        use_style_features=use_style_features,
        use_symmetric_features=use_symmetric_features,
        contrastive_weight=contrastive_weight,
        class_weights=class_weights,
        pooling=pooling,
        use_projection=use_projection,
        label_smoothing=label_smoothing,
        classifier=classifier,
        arcface_margin=arcface_margin,
        arcface_scale=arcface_scale,
        contrastive_mode=contrastive_mode,
        use_topic_adversary=use_topic_adversary,
        adv_lambda=adv_lambda,
        n_topics=n_topics,
        multi_head_adversary=multi_head_adversary,
        use_independence_penalty=use_independence_penalty,
        independence_weight=independence_weight,
        supcon_temperature=supcon_temperature,
    )
    # Reduce memory footprint where possible
    try:
        # Disable cache for training
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'config'):
            setattr(model.encoder.config, 'use_cache', False)
        # Enable gradient checkpointing on the encoder if available
        if grad_checkpointing and hasattr(model, 'encoder') and hasattr(model.encoder, 'gradient_checkpointing_enable'):
            model.encoder.gradient_checkpointing_enable()
        elif (not grad_checkpointing) and hasattr(model, 'encoder') and hasattr(model.encoder, 'gradient_checkpointing_disable'):
            model.encoder.gradient_checkpointing_disable()
    except Exception:
        pass
    
    # Speed settings on Ampere+ GPUs
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # Training arguments
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # Important: set gradient_checkpointing=False here because our model is a custom nn.Module
    # and HF Trainer will try to call model.gradient_checkpointing_enable() if this is True.
    # We already handle enabling checkpointing directly on the encoder above when requested.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model=select_metric,
        greater_is_better=True,
        report_to='none',
        fp16=(torch.cuda.is_available() and not bf16_ok),
        bf16=bf16_ok,
        dataloader_num_workers=8,
        remove_unused_columns=False,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=False,
    )
    
    # Create trainer
    # Optionally load cross-encoder teacher for distillation
    teacher_model = None
    teacher_tokenizer = None
    if teacher_model_dir and os.path.exists(teacher_model_dir):
        try:
            from transformers import AutoModelForSequenceClassification
            teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_dir)
            teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_dir)
            # Optionally place teacher on GPU for faster distillation
            if teacher_on_gpu is True or (teacher_on_gpu is None and torch.cuda.is_available()):
                teacher_model = teacher_model.to('cuda')
        except Exception as e:
            print(f"Warning: failed to load teacher from {teacher_model_dir}: {e}")
            teacher_model = None
            teacher_tokenizer = None

    if teacher_model is not None:
        trainer = DistillContrastiveTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            data_collator=SiameseDataCollator(),
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
                ContrastiveWeightSchedulerCallback(
                    model,
                    final_weight=contrastive_weight,
                    warmup_ratio=max(0.0, adv_warmup_ratio / 2),
                    ramp_ratio=adv_ramp_ratio,
                ),
            ],
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            distill_weight=distill_weight,
            distill_temperature=distill_temperature,
        )
    else:
        trainer = ContrastiveTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            data_collator=SiameseDataCollator(),
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
                ContrastiveWeightSchedulerCallback(
                    model,
                    final_weight=contrastive_weight,
                    warmup_ratio=max(0.0, adv_warmup_ratio / 2),
                    ramp_ratio=adv_ramp_ratio,
                ),
            ],
        )
    # Ensure only the primary 'labels' are used for metrics/label_ids
    try:
        setattr(trainer, 'label_names', ['labels'])
    except Exception:
        pass
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets['test'])
    print(f"Test results: {test_results}")

    # Calibration: temperature scaling and threshold selection on validation logits
    print("\nCalibrating on validation set...")
    val_pred = trainer.predict(tokenized_datasets['validation'])
    val_logits = _extract_logits_np(val_pred.predictions)
    val_labels = val_pred.label_ids
    temperature = _optimize_temperature(val_logits, val_labels)
    thr, row = _search_best_threshold(
        val_logits,
        val_labels,
        temperature,
        metric=calibrate_for,
        target_acc=target_acc,
        target_recall=target_recall,
    )
    calib = {"temperature": float(temperature), "threshold": float(thr), "opt_metric": calibrate_for, "selection_metrics": row}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'calibration.json'), 'w') as f:
        import json
        json.dump(calib, f)
    print(f"Saved calibration to {output_dir}/calibration.json: {calib}")
    
    # Save final model
    trainer.save_model(f'{output_dir}/final')
    tokenizer.save_pretrained(f'{output_dir}/final')
    print(f"\nModel saved to {output_dir}/final")
    
    return trainer, test_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train contrastive book-text matching model')
    parser.add_argument('--model', type=str, default='roberta-large', help='Base model name')
    parser.add_argument('--output', type=str, default='models/book_matcher_contrastive', 
                       help='Output directory')
    parser.add_argument('--data', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--epochs', type=int, default=6, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--contrastive-weight', type=float, default=0.3, help='Weight for InfoNCE/SupCon loss term')
    parser.add_argument('--symmetric-head/--no-symmetric-head', dest='symmetric_head', default=True, action='store_true', help='Use symmetric head features [h1,h2,|h1-h2|,h1*h2]')
    parser.add_argument('--teacher', type=str, default=None, help='Path to cross-encoder teacher model directory')
    parser.add_argument('--distill-weight', type=float, default=0.5, help='Weight for distillation loss')
    parser.add_argument('--distill-temperature', type=float, default=3.0, help='Temperature for distillation')
    parser.add_argument('--calibrate-for', type=str, default='accuracy', choices=['accuracy','balanced_accuracy','f1','f0.5','f2'], help='Metric to optimize threshold on')
    parser.add_argument('--target-acc', type=float, default=None, help='Minimum accuracy constraint for threshold selection')
    parser.add_argument('--target-recall', type=float, default=0.85, help='Minimum recall constraint for threshold selection')
    parser.add_argument('--pooling', type=str, default='attn', choices=['mean','cls','attn'], help='Pooling strategy for sequence embeddings')
    parser.add_argument('--no-projection', action='store_true', help='Disable projection head before classifier')
    parser.add_argument('--label-smoothing', type=float, default=0.03, help='Label smoothing in CE loss')
    parser.add_argument('--grad-accum', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='AdamW weight decay')
    parser.add_argument('--select-metric', type=str, default='balanced_accuracy', choices=['accuracy','balanced_accuracy','f1','precision','recall','auc','pr_auc'], help='Metric used to select best checkpoint')
    parser.add_argument('--classifier', type=str, default='arcface', choices=['mlp','arcface'], help='Classifier type for final decision')
    parser.add_argument('--arcface-margin', type=float, default=0.25, help='ArcFace additive angular margin')
    parser.add_argument('--arcface-scale', type=float, default=30.0, help='ArcFace scale for logits')
    parser.add_argument('--contrastive-mode', type=str, default='supcon', choices=['supcon','infonce'], help='Type of contrastive objective')
    parser.add_argument('--supcon-temperature', type=float, default=1.0, help='Initial temperature for SupCon/InfoNCE')
    parser.add_argument('--no-style-features', action='store_true', 
                       help='Disable style feature engineering')
    # Topic adversary & scheduler
    parser.add_argument('--no-topic-adversary', action='store_true', help='Disable topic adversary (GRL + topic head)')
    parser.add_argument('--adv-lambda', type=float, default=0.2, help='Max weight for adversarial topic loss (will be scheduled)')
    parser.add_argument('--adv-warmup-ratio', type=float, default=0.1, help='Fraction of total steps with adversary off (0→no warmup)')
    parser.add_argument('--adv-ramp-ratio', type=float, default=0.3, help='Fraction of total steps to ramp adversary to max')
    parser.add_argument('--grl-max-scale', type=float, default=1.0, help='Maximum gradient reversal scale applied to embeddings')
    parser.add_argument('--multi-head-adversary', action='store_true', help='Add a second topic head on pre-projection pooled embeddings')
    # Independence penalty
    parser.add_argument('--independence-penalty', action='store_true', help='Add HSIC penalty to reduce dependence on topic labels')
    parser.add_argument('--independence-weight', type=float, default=0.0, help='Weight for HSIC independence penalty')
    
    args = parser.parse_args()
    
    train_contrastive(
        model_name=args.model,
        output_dir=args.output,
        data_dir=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_style_features=not args.no_style_features,
        use_symmetric_features=args.symmetric_head,
        contrastive_weight=args.contrastive_weight,
        teacher_model_dir=args.teacher,
        distill_weight=args.distill_weight,
        distill_temperature=args.distill_temperature,
        calibrate_for=args.calibrate_for,
        target_acc=args.target_acc,
        target_recall=args.target_recall,
        pooling=args.pooling,
        use_projection=(not args.no_projection),
        label_smoothing=args.label_smoothing,
        grad_accum_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        select_metric=args.select_metric,
        classifier=args.classifier,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        contrastive_mode=args.contrastive_mode,
        supcon_temperature=args.supcon_temperature,
        use_topic_adversary=(not args.no_topic_adversary),
        adv_lambda=args.adv_lambda,
        adv_warmup_ratio=args.adv_warmup_ratio,
        adv_ramp_ratio=args.adv_ramp_ratio,
        grl_max_scale=args.grl_max_scale,
        multi_head_adversary=args.multi_head_adversary,
        use_independence_penalty=args.independence_penalty,
        independence_weight=args.independence_weight,
    )
