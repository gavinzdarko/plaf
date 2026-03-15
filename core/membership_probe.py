"""Membership inference attack — black-box, no shadow models.

Implements the relaxed membership inference approach (Salem et al. 2019):
instead of training shadow models, we calibrate a statistical baseline from
known non-member confidence scores and flag outliers as likely training
members.  This works because overfit models assign systematically higher
confidence to data they memorised during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score, roc_curve

from core.target import TargetModel
from config import AttackConfig


# ── Return types ────────────────────────────────────────────────────────────

@dataclass
class ProbeResults:
    """Full output of a membership probe run."""
    auc: float
    tpr_at_5fpr: float
    n_detected_members: int
    confidence_scores_members: list[float]
    confidence_scores_nonmembers: list[float]
    roc_fpr: list[float]
    roc_tpr: list[float]
    per_query_results: list[dict[str, Any]]


# ── Core class ──────────────────────────────────────────────────────────────

class MembershipProbe:
    """Black-box membership inference via confidence calibration.

    1. *calibrate_baseline* — build a Beta-distribution model of confidence
       scores for data the model has **not** seen.
    2. *probe* — compare candidate-record confidence against that baseline;
       records whose confidence exceeds the 95th-percentile threshold are
       flagged as likely training members.
    """

    def __init__(self, target: TargetModel, config: AttackConfig):
        self.target = target
        self.config = config
        self.baseline_distribution = None
        self.threshold: float | None = None

    # ── Step 1: calibration ─────────────────────────────────────────────

    def calibrate_baseline(self, non_member_data: np.ndarray) -> None:
        """Fit a Beta distribution to max-confidence scores of known non-members."""
        if non_member_data.size == 0:
            raise ValueError("non_member_data must not be empty")

        proba = self.target.predict_proba(non_member_data)
        confidences = proba.max(axis=1)

        # Clamp to open interval (0, 1) — Beta.fit requires this
        eps = 1e-7
        confidences = np.clip(confidences, eps, 1 - eps)

        a, b, loc, scale = beta_dist.fit(confidences, floc=0, fscale=1)
        self.baseline_distribution = beta_dist(a, b, loc=loc, scale=scale)
        self.threshold = float(np.percentile(confidences, 95))

    # ── Step 2: probe ───────────────────────────────────────────────────

    def probe(self, candidate_data: np.ndarray, labels: np.ndarray) -> ProbeResults:
        """Run membership inference on *candidate_data* using calibrated baseline."""
        if self.baseline_distribution is None or self.threshold is None:
            raise RuntimeError("Call calibrate_baseline() before probe().")
        if candidate_data.size == 0:
            raise ValueError("candidate_data must not be empty")

        proba = self.target.predict_proba(candidate_data)
        confidences = proba.max(axis=1)

        per_query: list[dict[str, Any]] = []
        membership_scores = np.empty(len(confidences))

        for i, (conf, pvec) in enumerate(zip(confidences, proba)):
            # Entropy of full probability vector
            pvec_safe = np.clip(pvec, 1e-12, None)
            entropy = float(-np.sum(pvec_safe * np.log(pvec_safe)))

            # How unusual is this confidence under the non-member baseline?
            ms = float(1.0 - self.baseline_distribution.cdf(np.clip(conf, 1e-7, 1 - 1e-7)))
            membership_scores[i] = ms

            per_query.append({
                "input_idx": i,
                "confidence": float(conf),
                "entropy": entropy,
                "membership_score": ms,
                "is_predicted_member": bool(conf > self.threshold),
            })

        predicted_member = confidences > self.threshold

        # Ground-truth labels: 1 = actual member, 0 = non-member
        labels = np.asarray(labels, dtype=int)

        # Handle degenerate single-class case
        if len(np.unique(labels)) < 2:
            auc = 0.5
            roc_fpr_arr = [0.0, 1.0]
            roc_tpr_arr = [0.0, 1.0]
            tpr_at_5fpr = 0.0
        else:
            roc_fpr_arr, roc_tpr_arr, _ = roc_curve(labels, confidences)
            auc = float(roc_auc_score(labels, confidences))
            # TPR at ≤5 % FPR
            idx = np.searchsorted(roc_fpr_arr, 0.05, side="right") - 1
            tpr_at_5fpr = float(roc_tpr_arr[max(idx, 0)])

        # Split confidence scores by ground-truth membership
        conf_members = confidences[labels == 1].tolist()
        conf_nonmembers = confidences[labels == 0].tolist()

        return ProbeResults(
            auc=auc,
            tpr_at_5fpr=tpr_at_5fpr,
            n_detected_members=int(predicted_member.sum()),
            confidence_scores_members=conf_members,
            confidence_scores_nonmembers=conf_nonmembers,
            roc_fpr=[float(x) for x in roc_fpr_arr],
            roc_tpr=[float(x) for x in roc_tpr_arr],
            per_query_results=per_query,
        )

    # ── Step 3: query-budget analysis ───────────────────────────────────

    def run_query_budget_analysis(
        self,
        candidate_data: np.ndarray,
        labels: np.ndarray,
        steps: list[int] | None = None,
    ) -> dict[int, float]:
        """Measure attack AUC as a function of query budget.

        Returns a dict mapping *n_queries → AUC*, showing how quickly
        information leaks as the attacker issues more queries.
        """
        if steps is None:
            steps = [100, 500, 1000, 2000, 5000]

        n_total = len(candidate_data)
        results: dict[int, float] = {}

        for n in steps:
            n_use = min(n, n_total)
            subset_data = candidate_data[:n_use]
            subset_labels = labels[:n_use]
            probe_result = self.probe(subset_data, subset_labels)
            results[n] = probe_result.auc

        return results
