"""
Inference script for contrastive book-text matching classifier.

Given two text snippets, predict whether they belong to the same book.
"""
import torch
import numpy as np
import re
from transformers import AutoTokenizer
from train_contrastive import ContrastiveBookMatcher


class ContrastiveBookMatcherInference:
    """Class for making predictions with contrastive model."""

    def __init__(self, model_path: str, threshold: float | None = None, calibration_path: str | None = None, style_calibration_path: str | None = None, device: str | None = None, verbose: bool = True):
        """Load model and tokenizer, auto-detecting base encoder size (base vs large).

        device: Optional override ('cpu', 'cuda', 'mps', or 'auto'/None for automatic detection)
        """
        import os
        # Device selection with MPS support
        def _auto_device() -> torch.device:
            try:
                if torch.cuda.is_available():
                    return torch.device('cuda')
            except Exception:
                pass
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return torch.device('mps')
            except Exception:
                pass
            return torch.device('cpu')

        if device and device != 'auto':
            self.device = torch.device(device)
        else:
            self.device = _auto_device()
        # Load weights from either PyTorch or safetensors format (prefer local path)
        pt_path = os.path.join(model_path, 'pytorch_model.bin')
        st_path = os.path.join(model_path, 'model.safetensors')

        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location='cpu')
        elif os.path.exists(st_path):
            try:
                from safetensors.torch import load_file as safe_load_file
            except Exception as e:
                raise RuntimeError(
                    "model.safetensors found but safetensors is not installed."
                ) from e
            state_dict = safe_load_file(st_path)
        else:
            raise FileNotFoundError(
                f"No model weights found under '{model_path}'. Expected files: 'pytorch_model.bin' or 'model.safetensors'. "
                "If this is a Modal volume path, first download locally, e.g.:\n"
                "  modal volume get writing7-artifacts /vol/models/book_matcher_contrastive/final ./models/book_matcher_contrastive/final"
            )

        # Detect base encoder width from checkpoint
        hidden_size = None
        try:
            emb = state_dict.get('encoder.embeddings.word_embeddings.weight')
            if emb is not None:
                # shape: [vocab, hidden_size]
                hidden_size = int(emb.shape[1])
        except Exception:
            pass

        # Prefer loading from the local model directory if it contains a config.json
        # Otherwise, map common hidden sizes to a base pretrained name (may require network)
        base_model_map = {768: 'roberta-base', 1024: 'roberta-large'}
        detected_model_name = base_model_map.get(hidden_size, 'roberta-base')
        try:
            cfg_path = os.path.join(model_path, 'config.json')
            if os.path.exists(cfg_path):
                detected_model_name = model_path  # load base encoder from local files
        except Exception:
            pass

        # Detect projection, topic adversary, and pooling by inspecting weights
        use_projection = False
        pooling = 'mean'
        use_topic_adversary = False
        n_topics = 5

        # Projection head presence
        try:
            if any(k.startswith('proj.0.weight') for k in state_dict.keys()):
                use_projection = True
        except Exception:
            pass

        # Topic head presence
        try:
            topic_keys = [k for k in state_dict.keys() if k.startswith('topic_head.')]
            if topic_keys:
                use_topic_adversary = True
                # Try to infer the output dim from the last Linear layer weight in the Sequential
                # Find all 'topic_head.<idx>.weight' and take the one with max idx
                last_w = None
                last_idx = -1
                import re as _re
                for k, v in state_dict.items():
                    m = _re.match(r'^topic_head\.(\d+)\.weight$', k)
                    if m:
                        idx = int(m.group(1))
                        if idx > last_idx:
                            last_idx = idx
                            last_w = v
                if last_w is None:
                    # Fallback to common positions
                    last_w = state_dict.get('topic_head.3.weight') or state_dict.get('topic_head.2.weight')
                if last_w is not None and hasattr(last_w, 'shape'):
                    n_topics = int(last_w.shape[0])
        except Exception:
            pass

        # Attention pooling presence
        try:
            if any(k.startswith('attn_mlp.0.weight') for k in state_dict.keys()):
                pooling = 'attn'
            else:
                pooling = 'mean'
        except Exception:
            pooling = 'mean'

        # Initialize model architecture to match checkpoint (contrastive-only)
        self.model = ContrastiveBookMatcher(
            model_name=detected_model_name,
            pooling=pooling,
            use_projection=use_projection,
            use_topic_adversary=use_topic_adversary,
            n_topics=n_topics,
            topic_out_dim=n_topics,
            contrastive_only=True,
        )
        self.contrastive_only = True

        # Now load weights
        # Allow missing/unexpected keys for backwards/forwards compatibility.
        # Additionally, drop adversary head weights if their shapes don't match (not used at inference).
        try:
            current = self.model.state_dict()
            drop_keys = []
            for k, t in state_dict.items():
                if (k.startswith('topic_head.') or k.startswith('topic_head_pre.')) and k in current:
                    try:
                        if tuple(t.shape) != tuple(current[k].shape):
                            drop_keys.append(k)
                    except Exception:
                        drop_keys.append(k)
            for k in drop_keys:
                state_dict.pop(k, None)
                if verbose:
                    print({"warning": f"dropped_mismatched_key: {k}"})
        except Exception:
            pass
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer: prefer local files inside model_path; else fall back to detected base model
        tok_files = [
            os.path.join(model_path, 'tokenizer.json'),
            os.path.join(model_path, 'vocab.json'),
            os.path.join(model_path, 'merges.txt'),
            os.path.join(model_path, 'tokenizer_config.json'),
        ]
        try:
            if any(os.path.exists(p) for p in tok_files):
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            else:
                # Fall back to base model tokenizer (works if you didn't save tokenizer artifacts)
                self.tokenizer = AutoTokenizer.from_pretrained(detected_model_name)
        except Exception:
            # Last resort: base model
            self.tokenizer = AutoTokenizer.from_pretrained(detected_model_name)
        if verbose:
            print(
                f"Loaded contrastive-only model from {model_path} (base={detected_model_name}, "
                f"proj={use_projection}, pooling={pooling}, "
                f"topic_adv={use_topic_adversary}, n_topics={n_topics}) on {self.device}"
            )

        # Calibration (temperature + threshold)
        self.temperature = 1.0
        self.threshold = 0.5
        calib_file = calibration_path or os.path.join(model_path, '..', 'calibration.json')
        calib_file = os.path.normpath(calib_file)
        try:
            import json
            if os.path.exists(calib_file):
                with open(calib_file) as f:
                    calib = json.load(f)
                self.temperature = float(calib.get('temperature', 1.0))
                self.threshold = float(calib.get('threshold', 0.5))
        except Exception:
            pass
        if threshold is not None:
            self.threshold = threshold

        # Style similarity calibration (cosine -> [0,1])
        self._style_calib = None
        style_calib_file = style_calibration_path or os.path.join(model_path, '..', 'style_calibration.json')
        style_calib_file = os.path.normpath(style_calib_file)
        try:
            if os.path.exists(style_calib_file):
                with open(style_calib_file, 'r', encoding='utf-8') as f:
                    payload = __import__('json').load(f)
                self._style_calib = payload.get('style_calibration')
        except Exception:
            self._style_calib = None

    def _apply_style_calibration(self, cosine: float) -> float | None:
        import math
        calib = self._style_calib
        if not calib or not (isinstance(cosine, float) or isinstance(cosine, (int, np.floating))):
            return None
        try:
            method = calib.get('method')
            x = float(cosine)
            if method == 'logistic':
                a = float(calib.get('coef', 0.0))
                b = float(calib.get('intercept', 0.0))
                z = a * x + b
                # Stable sigmoid
                if z >= 0:
                    ez = math.exp(-z)
                    p = 1.0 / (1.0 + ez)
                else:
                    ez = math.exp(z)
                    p = ez / (1.0 + ez)
                return float(p)
            if method == 'isotonic':
                xs = calib.get('x_thresholds') or []
                ys = calib.get('y_values') or []
                if not xs or not ys or len(xs) != len(ys):
                    return None
                # Binary search for position, then linear interpolate
                import bisect
                i = bisect.bisect_left(xs, x)
                if i <= 0:
                    return float(ys[0])
                if i >= len(xs):
                    return float(ys[-1])
                x0, x1 = xs[i-1], xs[i]
                y0, y1 = ys[i-1], ys[i]
                # Guard against duplicates
                if x1 == x0:
                    return float(y0)
                t = (x - x0) / (x1 - x0)
                return float(y0 + t * (y1 - y0))
        except Exception:
            return None
        return None
    
    @torch.no_grad()
    def _embed_texts(self, texts, max_length: int = 512, use_projection: bool = True):
        """Embed raw texts into L2-normalized pooled embeddings using the encoder + pooling (+ projection).

        Returns np.ndarray of shape (N, D) with unit-norm rows.
        """
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        enc = self.tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        self.model.eval()
        out = self.model.encoder(enc['input_ids'], attention_mask=enc['attention_mask']).last_hidden_state
        pooled = self.model._pool_embeddings(out, enc['attention_mask'])  # type: ignore[attr-defined]
        if use_projection and getattr(self.model, 'proj', None) is not None:
            pooled = self.model.proj(pooled)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().numpy()

    @staticmethod
    def _split_sentences_simple(text: str):
        """Simple sentence splitting similar to prepare_data; avoids heavy deps at inference."""
        sentences = []
        pattern = r'[.!?]+["\'\)]*\s+(?=[A-Z])'
        parts = re.split(pattern, text)
        for part in parts:
            part = part.strip()
            if len(part) > 20:
                sentences.append(part)
        if not sentences and len(text.strip()) > 20:
            sentences.append(text.strip())
        return sentences

    @staticmethod
    def _make_chunks_from_sentences(sentences, chunk_size: int = 14, overlap: int = 4, min_chars: int = 200):
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(sentences), step):
            chunk = ' '.join(sentences[i:i + chunk_size]).strip()
            if len(chunk) >= int(min_chars):
                chunks.append(chunk)
        if not chunks and sentences:
            chunk = ' '.join(sentences[:chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    @torch.no_grad()
    def style_similarity(
        self,
        text1: str,
        text2: str,
        num_chunks = 'auto',
        chunk_size: int = 14,
        overlap: int = 4,
        aggregate: str = 'mean',  # 'mean' or 'topk_mean'
        topk: int = 5,
        max_length: int = 512,
    ) -> dict:
        """Compute a style similarity score using cosine between normalized pooled embeddings.

        If num_chunks>1, splits texts into sentence chunks and aggregates pairwise cosines.
        Returns {'cosine': float, 'aggregate': str, 'pairs': int}.
        """
        # Auto-decide chunks based on token/sentence length
        auto = (isinstance(num_chunks, str) and num_chunks == 'auto') or (isinstance(num_chunks, int) and num_chunks <= 0)
        if auto:
            # Estimate per-text chunks: if short (<=512 tokens and <= chunk_size sentences), use 1; else scale with length.
            def _estimate_nchunks(txt: str, max_chunks: int = 8) -> int:
                try:
                    # Token count without truncation, avoiding HF warning by using backend_tokenizer when available
                    n_tok = 0
                    backend = getattr(self.tokenizer, 'backend_tokenizer', None)
                    if backend is not None:
                        enc = backend.encode(txt, add_special_tokens=True)
                        n_tok = len(enc.ids)
                    else:
                        # Fallback: use encode (may warn), but guard
                        ids = self.tokenizer.encode(txt, add_special_tokens=True, truncation=False)
                        n_tok = len(ids)
                except Exception:
                    n_tok = 0
                n_sent = len(self._split_sentences_simple(txt))
                if (n_tok and n_tok <= max_length) and n_sent <= chunk_size:
                    return 1
                # Scale by whichever is larger: tokens/512 or sentences/chunk_size
                scale_tok = (n_tok / float(max_length)) if n_tok else 0.0
                scale_sent = (n_sent / float(chunk_size)) if n_sent else 0.0
                import math
                est = int(math.ceil(max(scale_tok, scale_sent)))
                return max(1, min(est, max_chunks))

            n1 = _estimate_nchunks(text1)
            n2 = _estimate_nchunks(text2)
        else:
            n1 = n2 = int(num_chunks)

        # Single-chunk fast path
        if n1 <= 1 and n2 <= 1:
            Z = self._embed_texts([text1, text2], max_length=max_length)
            if Z.shape[0] < 2:
                return {'cosine': float('nan'), 'aggregate': 'single', 'pairs': 0}
            cos = float((Z[0] * Z[1]).sum())
            out = {'cosine': cos, 'aggregate': 'single', 'pairs': 1}
            cal = self._apply_style_calibration(cos)
            if cal is not None:
                out['calibrated'] = cal
            return out

        # Multi-chunk: split to sentences, make chunks, limit to num_chunks
        s1 = self._split_sentences_simple(text1)
        s2 = self._split_sentences_simple(text2)
        c1 = self._make_chunks_from_sentences(s1, chunk_size=chunk_size, overlap=overlap)
        c2 = self._make_chunks_from_sentences(s2, chunk_size=chunk_size, overlap=overlap)
        if not c1:
            c1 = [text1]
        if not c2:
            c2 = [text2]
        c1 = c1[:max(1, n1)]
        c2 = c2[:max(1, n2)]
        Z1 = self._embed_texts(c1, max_length=max_length)
        Z2 = self._embed_texts(c2, max_length=max_length)
        if Z1.size == 0 or Z2.size == 0:
            return {'cosine': float('nan'), 'aggregate': aggregate, 'pairs': 0}
        # Pairwise cosine via dot products (already L2-normalized)
        M = Z1 @ Z2.T
        cos_vals = M.flatten()
        if aggregate == 'topk_mean':
            k = max(1, min(topk, cos_vals.shape[0]))
            idx = np.argpartition(-cos_vals, k - 1)[:k]
            score = float(cos_vals[idx].mean())
            agg = 'topk_mean'
        else:
            score = float(cos_vals.mean())
            agg = 'mean'
        out = {'cosine': score, 'aggregate': agg, 'pairs': int(M.size)}
        cal = self._apply_style_calibration(score)
        if cal is not None:
            out['calibrated'] = cal
        return out
    
    def predict(self, text1: str, text2: str) -> dict:
        """
        Predict whether two text chunks come from the same book.

        Uses cosine similarity between embeddings.

        Args:
            text1: First text chunk
            text2: Second text chunk

        Returns:
            dict with 'same_book' (bool), 'confidence' (float), 'probability' (float),
            'cosine' (float), and 'method' ('embedding')
        """
        # Get embeddings
        emb1 = self._embed_texts([text1], max_length=512, use_projection=True)
        emb2 = self._embed_texts([text2], max_length=512, use_projection=True)
        # Compute cosine similarity (embeddings are already normalized)
        cosine = float(np.dot(emb1[0], emb2[0]))
        # Map cosine [-1, 1] to probability-like [0, 1]
        # Using (cos + 1) / 2 as a simple mapping
        prob = (cosine + 1.0) / 2.0
        same_book = prob >= self.threshold
        confidence = prob if same_book else (1 - prob)
        return {
            'same_book': same_book,
            'confidence': confidence,
            'probability': prob,
            'cosine': cosine,
            'method': 'embedding',
        }
    
    def predict_batch(self, pairs: list) -> list:
        """Predict on multiple pairs."""
        results = []
        for text1, text2 in pairs:
            result = self.predict(text1, text2)
            results.append(result)
        return results


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict whether text belongs to a book')
    parser.add_argument('--model', type=str, default='models/book_matcher_contrastive/final',
                       help='Path to trained model')
    parser.add_argument('--text1', type=str, required=True,
                       help='First text chunk')
    parser.add_argument('--text2', type=str, required=True,
                       help='Second text chunk')
    parser.add_argument('--threshold', type=float, default=None, help='Decision threshold override (default: from calibration.json or 0.5)')
    parser.add_argument('--calibration', type=str, default=None, help='Path to calibration.json')
    parser.add_argument('--style-calibration', type=str, default=None, help='Path to style_calibration.json')
    # Style similarity options
    parser.add_argument('--style-sim', action='store_true', help='Compute style similarity (cosine) instead of classification')
    parser.add_argument('--num-chunks', default='auto', help="Chunks per text: integer or 'auto' for length-based; auto uses 1 if short (<=512 tokens and <=14 sentences)")
    parser.add_argument('--chunk-size', type=int, default=14, help='Sentences per chunk for style similarity')
    parser.add_argument('--overlap', type=int, default=4, help='Sentence overlap between chunks for style similarity')
    parser.add_argument('--aggregate', type=str, default='mean', choices=['mean','topk_mean'], help='Aggregation for multi-chunk cosine')
    parser.add_argument('--topk', type=int, default=5, help='Top-k for topk_mean aggregation')
    
    args = parser.parse_args()
    
    # Load model
    matcher = ContrastiveBookMatcherInference(
        args.model,
        threshold=args.threshold,
        calibration_path=args.calibration,
        style_calibration_path=args.style_calibration,
    )
    
    if args.style_sim:
        # Parse num_chunks which can be 'auto' or an int
        try:
            nc = int(args.num_chunks)
        except Exception:
            nc = 'auto'
        sim = matcher.style_similarity(
            args.text1,
            args.text2,
            num_chunks=nc,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            aggregate=args.aggregate,
            topk=args.topk,
        )
        # Also show a naive [0,1] mapping for convenience
        mapped = (sim['cosine'] + 1.0) / 2.0 if np.isfinite(sim['cosine']) else float('nan')
        print(f"\nStyle similarity (cosine): {sim['cosine']:.4f}  | agg={sim['aggregate']} pairs={sim['pairs']}")
        if 'calibrated' in sim:
            print(f"Calibrated [0,1]: {sim['calibrated']:.4f}")
        else:
            print(f"Naive [0,1] map: {mapped:.4f}")
    else:
        # Make prediction
        result = matcher.predict(args.text1, args.text2)
        print(f"\nPrediction: {'SAME BOOK' if result['same_book'] else 'DIFFERENT BOOKS'}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probability: {result['probability']:.4f}")
    
    print(f"\nText 1: {args.text1[:200]}...")
    print(f"Text 2: {args.text2[:200]}...")


if __name__ == '__main__':
    main()
