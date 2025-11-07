"""
Two-stage inference: bi-encoder (contrastive) prefilter + cross-encoder re-ranker.

Usage examples (local):
    python inference_two_stage.py \
        --bi-model /vol/models/book_matcher_contrastive/final \
        --cross-model /vol/models/book_matcher/final \
        --text1 "..." --text2 "..."

Logic:
    1) Run contrastive model; if it predicts negative (below threshold), return that result.
    2) If it predicts positive, run cross-encoder and use its probability/decision as final.
"""
from __future__ import annotations

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Dict

from inference_contrastive import ContrastiveBookMatcherInference


class TwoStageBookMatcher:
    def __init__(
        self,
        bi_model_dir: str = "/vol/models/book_matcher_contrastive/final",
        cross_model_dir: str = "/vol/models/book_matcher/final",
        prefilter_threshold: Optional[float] = None,
        cross_threshold: float = 0.5,
        calibration_path: Optional[str] = None,
    ) -> None:
        """Initialize two-stage matcher.

        - bi_model_dir: path to contrastive model dir containing weights + tokenizer.
        - cross_model_dir: path to cross-encoder model dir containing weights + tokenizer.
        - prefilter_threshold: override the contrastive decision threshold (else use calibration/defaults).
        - cross_threshold: threshold for cross-encoder decision (default 0.5).
        - calibration_path: optional path to contrastive calibration.json.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Bi-encoder (contrastive) matcher
        self.bi = ContrastiveBookMatcherInference(
            model_path=bi_model_dir,
            threshold=prefilter_threshold,
            calibration_path=calibration_path,
        )

        # Cross-encoder (sequence-pair) model
        if not os.path.exists(cross_model_dir):
            raise FileNotFoundError(f"Cross-encoder model dir not found: {cross_model_dir}")
        self.cross_tokenizer = AutoTokenizer.from_pretrained(cross_model_dir)
        self.cross_model = AutoModelForSequenceClassification.from_pretrained(cross_model_dir)
        self.cross_model.to(self.device)
        self.cross_model.eval()
        self.cross_threshold = float(cross_threshold)

    def _cross_predict(self, text1: str, text2: str) -> Dict[str, float | bool]:
        inputs = self.cross_tokenizer(
            text1,
            text2,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.cross_model(**inputs).logits
            probs = torch.softmax(logits, dim=1).detach().cpu()
        same_prob = float(probs[0, 1].item())
        same = same_prob >= self.cross_threshold
        conf = same_prob if same else (1.0 - same_prob)
        return {"same_book": same, "probability": same_prob, "confidence": conf}

    def predict(self, text1: str, text2: str) -> Dict[str, object]:
        """Run two-stage inference on a single pair."""
        # Stage 1: bi-encoder (fast)
        bi_res = self.bi.predict(text1, text2)
        bi_pass = bool(bi_res.get("same_book", False))

        if not bi_pass:
            # Short-circuit negative — faster and more precise overall
            return {
                "stage": "bi-only",
                "bi_probability": bi_res["probability"],
                "bi_confidence": bi_res["confidence"],
                "final_probability": bi_res["probability"],
                "same_book": False,
            }

        # Stage 2: cross-encoder (re-rank precision)
        cross_res = self._cross_predict(text1, text2)
        return {
            "stage": "two-stage",
            "bi_probability": bi_res["probability"],
            "bi_confidence": bi_res["confidence"],
            "cross_probability": cross_res["probability"],
            "cross_confidence": cross_res["confidence"],
            "final_probability": cross_res["probability"],
            "same_book": cross_res["same_book"],
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Two-stage (bi + cross) inference")
    parser.add_argument("--bi-model", dest="bi_model", type=str, default="models/book_matcher_contrastive/final")
    parser.add_argument("--cross-model", dest="cross_model", type=str, default="models/book_matcher/final")
    parser.add_argument("--text1", type=str, required=True)
    parser.add_argument("--text2", type=str, required=True)
    parser.add_argument("--prefilter-threshold", type=float, default=None, help="Override bi-encoder threshold")
    parser.add_argument("--cross-threshold", type=float, default=0.5)
    parser.add_argument("--calibration", type=str, default=None, help="Path to bi-encoder calibration.json")

    args = parser.parse_args()
    ts = TwoStageBookMatcher(
        bi_model_dir=args.bi_model,
        cross_model_dir=args.cross_model,
        prefilter_threshold=args.prefilter_threshold,
        cross_threshold=args.cross_threshold,
        calibration_path=args.calibration,
    )
    res = ts.predict(args.text1, args.text2)

    print("\nTwo-stage prediction:")
    print(res)


if __name__ == "__main__":
    main()

