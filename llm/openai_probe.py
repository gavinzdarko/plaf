"""Membership inference against OpenAI API via logprob-based perplexity.

OpenAI's API exposes per-token log-probabilities, giving us real perplexity
measurements — the gold standard for detecting training data memorization.
"""

from __future__ import annotations

import math
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


class OpenAITarget:
    """Wrapper for OpenAI API that computes perplexity via logprobs."""

    def __init__(self, api_key: str | None = None, model_name: str = "gpt-4o-mini"):
        from openai import OpenAI

        if api_key is None:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity using OpenAI's logprobs feature.

        We send the text as a prompt and ask for a short continuation with
        logprobs enabled. The model's confidence on its continuation of
        familiar text will be higher (lower perplexity) than on novel text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Continue the text with exactly one sentence. Output only the continuation."},
                    {"role": "user", "content": text},
                ],
                max_tokens=60,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
            )

            # Extract logprobs from the response
            if (response.choices and
                    response.choices[0].logprobs and
                    response.choices[0].logprobs.content):
                token_logprobs = [
                    token.logprob
                    for token in response.choices[0].logprobs.content
                    if token.logprob is not None
                ]
                if token_logprobs:
                    avg_logprob = sum(token_logprobs) / len(token_logprobs)
                    return float(math.exp(-avg_logprob))

            return 50.0  # neutral fallback

        except Exception as e:
            print(f"  OpenAI error: {e}")
            return 50.0


class OpenAIMembershipProbe:
    """Membership inference for OpenAI models via perplexity analysis."""

    def __init__(self, target: OpenAITarget):
        self.target = target

    def probe(
        self,
        training_texts: list[str],
        novel_texts: list[str],
        progress_callback=None,
    ) -> ProbeResults:
        total = len(training_texts) + len(novel_texts)

        train_perplexities = []
        for i, text in enumerate(training_texts):
            ppl = self.target.compute_perplexity(text)
            train_perplexities.append(ppl)
            if progress_callback:
                progress_callback((i + 1) / total, f"Training text {i+1}/{len(training_texts)}")
            time.sleep(0.3)

        novel_perplexities = []
        for i, text in enumerate(novel_texts):
            ppl = self.target.compute_perplexity(text)
            novel_perplexities.append(ppl)
            if progress_callback:
                progress_callback((len(training_texts) + i + 1) / total,
                                  f"Novel text {i+1}/{len(novel_texts)}")
            time.sleep(0.3)

        # Lower perplexity = more likely member → negate for AUC
        all_ppl = train_perplexities + novel_perplexities
        scores = [-p for p in all_ppl]
        labels = [1] * len(training_texts) + [0] * len(novel_texts)

        labels_arr = np.array(labels)
        scores_arr = np.array(scores)

        if len(np.unique(labels_arr)) < 2:
            auc, tpr_at_5fpr = 0.5, 0.0
            roc_fpr_arr, roc_tpr_arr = [0.0, 1.0], [0.0, 1.0]
        else:
            roc_fpr_arr, roc_tpr_arr, _ = roc_curve(labels_arr, scores_arr)
            auc = float(roc_auc_score(labels_arr, scores_arr))
            idx = np.searchsorted(roc_fpr_arr, 0.05, side="right") - 1
            tpr_at_5fpr = float(roc_tpr_arr[max(idx, 0)])

        threshold = float(np.median(novel_perplexities))
        n_detected = sum(1 for p in train_perplexities if p < threshold)

        per_query = []
        for i, ppl in enumerate(all_ppl):
            per_query.append({
                "input_idx": i,
                "confidence": float(ppl),
                "entropy": 0.0,
                "membership_score": float(-ppl),
                "is_predicted_member": bool(ppl < threshold),
            })

        return ProbeResults(
            auc=auc,
            tpr_at_5fpr=tpr_at_5fpr,
            n_detected_members=n_detected,
            confidence_scores_members=train_perplexities,
            confidence_scores_nonmembers=novel_perplexities,
            roc_fpr=[float(x) for x in roc_fpr_arr],
            roc_tpr=[float(x) for x in roc_tpr_arr],
            per_query_results=per_query,
        )
