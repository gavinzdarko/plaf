"""Membership inference against Google Gemini API via continuation accuracy.

Since Gemini doesn't expose logprobs, we use a continuation-based approach:
give the model the first half of a text and ask it to continue. If the model
has memorized the text, its continuation will closely match the actual second
half. We measure this via word overlap (Jaccard similarity).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from core.membership_probe import ProbeResults


def _word_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity between word sets."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class GeminiTarget:
    """Wrapper for Google Gemini API using continuation-based memorization detection."""

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-flash"):
        from google import genai

        if api_key is None:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def compute_memorization_score(self, text: str) -> float:
        """Measure how well Gemini can reproduce the second half of a text.

        Returns a "perplexity-like" score where LOWER = more memorized:
        - High word overlap with actual continuation → low score (memorized)
        - Low word overlap → high score (not memorized)
        """
        from google.genai import types

        words = text.split()
        if len(words) < 10:
            return 50.0

        # Split: give first 40% as prompt, expect last 60%
        split_point = max(5, len(words) * 2 // 5)
        prompt_text = " ".join(words[:split_point])
        actual_continuation = " ".join(words[split_point:])

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=(
                    f"Complete the following text by writing the next 2-3 sentences. "
                    f"Try to match the original source material as closely as possible. "
                    f"Only output the continuation, nothing else.\n\n"
                    f'"{prompt_text}..."'
                ),
                config=types.GenerateContentConfig(
                    max_output_tokens=150,
                    temperature=0.0,
                ),
            )

            generated = response.text.strip() if response.text else ""
            overlap = _word_overlap(generated, actual_continuation)

            # Convert overlap (0-1, higher=more memorized) to perplexity-like
            # (lower=more memorized) for compatibility with existing display
            # Scale: overlap 0.5+ → perplexity ~10, overlap 0.0 → perplexity ~100
            perplexity = max(1.0, 100.0 * (1.0 - overlap * 1.5))
            return perplexity

        except Exception as e:
            print(f"  Gemini error: {e}")
            return 50.0


class GeminiMembershipProbe:
    """Membership inference for Gemini via continuation accuracy."""

    def __init__(self, target: GeminiTarget):
        self.target = target

    def probe(
        self,
        training_texts: list[str],
        novel_texts: list[str],
        progress_callback=None,
    ) -> ProbeResults:
        total = len(training_texts) + len(novel_texts)

        train_scores = []
        for i, text in enumerate(training_texts):
            score = self.target.compute_memorization_score(text)
            train_scores.append(score)
            if progress_callback:
                progress_callback((i + 1) / total, f"Training text {i+1}/{len(training_texts)}")
            time.sleep(0.5)

        novel_scores = []
        for i, text in enumerate(novel_texts):
            score = self.target.compute_memorization_score(text)
            novel_scores.append(score)
            if progress_callback:
                progress_callback((len(training_texts) + i + 1) / total,
                                  f"Novel text {i+1}/{len(novel_texts)}")
            time.sleep(0.5)

        # Lower score = more memorized → negate for AUC computation
        all_scores = train_scores + novel_scores
        auc_scores = [-s for s in all_scores]
        labels = [1] * len(training_texts) + [0] * len(novel_texts)

        labels_arr = np.array(labels)
        scores_arr = np.array(auc_scores)

        if len(np.unique(labels_arr)) < 2:
            auc, tpr_at_5fpr = 0.5, 0.0
            roc_fpr_arr, roc_tpr_arr = [0.0, 1.0], [0.0, 1.0]
        else:
            roc_fpr_arr, roc_tpr_arr, _ = roc_curve(labels_arr, scores_arr)
            auc = float(roc_auc_score(labels_arr, scores_arr))
            idx = np.searchsorted(roc_fpr_arr, 0.05, side="right") - 1
            tpr_at_5fpr = float(roc_tpr_arr[max(idx, 0)])

        threshold = float(np.median(novel_scores))
        n_detected = sum(1 for s in train_scores if s < threshold)

        per_query = []
        for i, score in enumerate(all_scores):
            per_query.append({
                "input_idx": i,
                "confidence": float(score),
                "entropy": 0.0,
                "membership_score": float(-score),
                "is_predicted_member": bool(score < threshold),
            })

        return ProbeResults(
            auc=auc,
            tpr_at_5fpr=tpr_at_5fpr,
            n_detected_members=n_detected,
            confidence_scores_members=train_scores,
            confidence_scores_nonmembers=novel_scores,
            roc_fpr=[float(x) for x in roc_fpr_arr],
            roc_tpr=[float(x) for x in roc_tpr_arr],
            per_query_results=per_query,
        )
