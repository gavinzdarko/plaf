"""Membership inference for language models via perplexity analysis.

Language models assign lower perplexity to text they have seen during
training.  By comparing the perplexity distribution of known-training
texts vs. novel texts we can detect memorisation — a direct privacy risk
that reveals whether specific text was part of the training corpus.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.membership_probe import ProbeResults


# ── LLM target wrapper ─────────────────────────────────────────────────────

class LLMTarget:
    """Wrapper for a HuggingFace causal language model."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # GPT-2 has no pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of *text* under the language model.

        Perplexity = exp(mean per-token cross-entropy loss).
        Lower perplexity indicates the model "knows" the text better,
        which is the signal we exploit for membership inference.
        """
        if not text or not text.strip():
            return float("inf")

        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        input_ids = encodings.input_ids.to(self.device)

        # Need at least 2 tokens (1 context + 1 prediction)
        if input_ids.size(1) < 2:
            return float("inf")

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            # outputs.loss is the mean cross-entropy over all predicted tokens
            loss = outputs.loss.item()

        return math.exp(loss)

    def compute_batch_perplexity(self, texts: list[str]) -> list[float]:
        """Compute perplexity for each text independently."""
        return [self.compute_perplexity(t) for t in texts]


# ── LLM membership probe ───────────────────────────────────────────────────

class LLMMembershipProbe:
    """Membership inference for language models via perplexity analysis.

    Key insight: training texts receive lower perplexity from the model
    because it has memorised them.  We use *negative* perplexity as the
    membership score so that higher score → more likely member, matching
    the convention used by the tabular MembershipProbe.
    """

    def __init__(self, target: LLMTarget):
        self.target = target

    def probe(
        self,
        training_texts: list[str],
        novel_texts: list[str],
    ) -> ProbeResults:
        """Run membership inference on training (member) vs novel (non-member) texts.

        Parameters
        ----------
        training_texts : list[str]
            Texts known to be in the model's training data (label = 1).
        novel_texts : list[str]
            Texts known to NOT be in the training data (label = 0).

        Returns
        -------
        ProbeResults
            Same dataclass used by the tabular membership probe so the
            dashboard can render results identically.
        """
        # Compute perplexities
        ppl_members = self.target.compute_batch_perplexity(training_texts)
        ppl_nonmembers = self.target.compute_batch_perplexity(novel_texts)

        # Filter out infinite perplexities (empty / single-token texts)
        def _filter(ppls, texts):
            return [(p, t) for p, t in zip(ppls, texts) if math.isfinite(p)]

        valid_members = _filter(ppl_members, training_texts)
        valid_nonmembers = _filter(ppl_nonmembers, novel_texts)

        if not valid_members or not valid_nonmembers:
            return ProbeResults(
                auc=0.5,
                tpr_at_5fpr=0.0,
                n_detected_members=0,
                confidence_scores_members=ppl_members,
                confidence_scores_nonmembers=ppl_nonmembers,
                roc_fpr=[0.0, 1.0],
                roc_tpr=[0.0, 1.0],
                per_query_results=[],
            )

        member_ppls = [p for p, _ in valid_members]
        nonmember_ppls = [p for p, _ in valid_nonmembers]

        # Negative perplexity as score: lower ppl → higher score → more likely member
        all_ppls = member_ppls + nonmember_ppls
        scores = [-p for p in all_ppls]
        labels = [1] * len(member_ppls) + [0] * len(nonmember_ppls)

        scores_arr = np.array(scores)
        labels_arr = np.array(labels)

        # ROC / AUC
        if len(np.unique(labels_arr)) < 2:
            auc = 0.5
            fpr_arr = np.array([0.0, 1.0])
            tpr_arr = np.array([0.0, 1.0])
            tpr_at_5fpr = 0.0
        else:
            fpr_arr, tpr_arr, _ = roc_curve(labels_arr, scores_arr)
            auc = float(roc_auc_score(labels_arr, scores_arr))
            idx = np.searchsorted(fpr_arr, 0.05, side="right") - 1
            tpr_at_5fpr = float(tpr_arr[max(idx, 0)])

        # Threshold: median perplexity of novel texts
        threshold_ppl = float(np.median(nonmember_ppls))

        # Build per-query results
        per_query: list[dict[str, Any]] = []
        n_detected = 0

        for i, (ppl, lbl) in enumerate(zip(all_ppls, labels)):
            is_member = ppl < threshold_ppl
            if is_member:
                n_detected += 1
            per_query.append({
                "input_idx": i,
                "confidence": ppl,
                "entropy": 0.0,
                "membership_score": -ppl,
                "is_predicted_member": bool(is_member),
            })

        return ProbeResults(
            auc=auc,
            tpr_at_5fpr=tpr_at_5fpr,
            n_detected_members=n_detected,
            confidence_scores_members=member_ppls,
            confidence_scores_nonmembers=nonmember_ppls,
            roc_fpr=[float(x) for x in fpr_arr],
            roc_tpr=[float(x) for x in tpr_arr],
            per_query_results=per_query,
        )

    def run_query_budget_analysis(
        self,
        training_texts: list[str],
        novel_texts: list[str],
        steps: list[int] | None = None,
    ) -> dict[int, float]:
        """Measure attack AUC as a function of query budget.

        Returns {n_queries: AUC} showing leakage vs. attacker effort.
        """
        if steps is None:
            steps = [10, 25, 50, 100, 200]

        results: dict[int, float] = {}

        for n in steps:
            # Split budget roughly evenly between members and non-members
            n_each = max(n // 2, 1)
            subset_train = training_texts[:min(n_each, len(training_texts))]
            subset_novel = novel_texts[:min(n_each, len(novel_texts))]
            probe_result = self.probe(subset_train, subset_novel)
            results[n] = probe_result.auc

        return results
