"""Attribute inference via systematic input sweeping.

For each sensitive feature the attacker wants to reconstruct, we hold all
other features fixed and sweep through every possible value of the target
feature.  The model's confidence response curve reveals which values it
"expects" — i.e., which values appeared in the training data.  By averaging
over many base points we obtain a robust reconstruction of the training
distribution for that attribute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy as kl_divergence

from core.target import TargetModel
from config import AttackConfig


# ── Return types ────────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    """Result of sweeping a single attribute across its domain."""
    attribute_name: str
    values: list[Any]
    mean_confidences: list[float]
    std_confidences: list[float]
    sensitivity_score: float
    reconstructed_distribution: list[float]


@dataclass
class AttributeLeakage:
    """Per-attribute leakage summary."""
    attribute_name: str
    sensitivity_score: float
    reconstructed_distribution: list[float]
    ground_truth_distribution: list[float]
    kl_divergence: float
    most_leaked_values: list[Any]


@dataclass
class ReconstructionReport:
    """Full report across all sensitive features."""
    attributes: list[AttributeLeakage]


# ── Core class ──────────────────────────────────────────────────────────────

class AttributeReconstructor:
    """Reconstruct sensitive training-data distributions via input sweeping.

    For each sensitive feature, the attacker varies that feature across its
    domain while keeping all other features fixed, then observes how the
    model's confidence changes.  A sharp confidence peak at a specific value
    indicates the model has memorised that value from training data.
    """

    def __init__(
        self,
        target: TargetModel,
        config: AttackConfig,
        feature_schema: dict[str, dict],
    ):
        self.target = target
        self.config = config
        self.feature_schema = feature_schema

    # ── Single-attribute sweep ──────────────────────────────────────────

    def sweep_attribute(self, attr_name: str, base_inputs: np.ndarray) -> SweepResult:
        """Sweep *attr_name* across its domain, averaging over base points.

        For each base input we substitute every possible value of the target
        attribute, query the model, and record the confidence for the positive
        class (readmission = 1).  Averaging across base points smooths noise.
        """
        schema = self.feature_schema[attr_name]
        values = schema["values"]
        col_idx = schema["col_idx"]

        n_base = min(self.config.n_base_points, len(base_inputs))
        if n_base == 0:
            raise ValueError("base_inputs must not be empty")

        rng = np.random.RandomState(42)
        base_indices = rng.choice(len(base_inputs), size=n_base, replace=False)

        # confidence_matrix[base_i, val_j] = P(readmission=1 | attr=val_j)
        confidence_matrix = np.zeros((n_base, len(values)))

        for bi, idx in enumerate(base_indices):
            batch = np.tile(base_inputs[idx], (len(values), 1))
            for vi, val in enumerate(values):
                batch[vi, col_idx] = val
            proba = self.target.predict_proba(batch)
            # Positive-class confidence (class 1)
            confidence_matrix[bi] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        mean_conf = confidence_matrix.mean(axis=0)
        std_conf = confidence_matrix.std(axis=0)

        # Sensitivity = max absolute gradient of confidence w.r.t. attribute
        if len(mean_conf) > 1:
            gradients = np.abs(np.diff(mean_conf))
            sensitivity_score = float(gradients.max())
        else:
            sensitivity_score = 0.0

        # Reconstruct distribution via softmax over mean confidences
        reconstructed = softmax(mean_conf).tolist()

        return SweepResult(
            attribute_name=attr_name,
            values=[v if not isinstance(v, (np.integer, np.floating)) else v.item() for v in values],
            mean_confidences=mean_conf.tolist(),
            std_confidences=std_conf.tolist(),
            sensitivity_score=sensitivity_score,
            reconstructed_distribution=reconstructed,
        )

    # ── Full reconstruction ─────────────────────────────────────────────

    def reconstruct_all(self, data: np.ndarray) -> ReconstructionReport:
        """Run attribute sweeps for every sensitive feature in the schema.

        Compares the reconstructed distribution to the ground-truth
        distribution in *data* via KL divergence.
        """
        leakages: list[AttributeLeakage] = []

        for attr_name, schema in self.feature_schema.items():
            if not schema.get("sensitive", False):
                continue

            sweep = self.sweep_attribute(attr_name, data)
            col_idx = schema["col_idx"]
            values = schema["values"]

            # Ground-truth distribution from data
            col_data = data[:, col_idx]
            gt_counts = np.array([np.sum(col_data == v) for v in values], dtype=float)
            gt_total = gt_counts.sum()
            if gt_total > 0:
                gt_dist = gt_counts / gt_total
            else:
                gt_dist = np.ones(len(values)) / len(values)

            # Smooth to avoid log(0) in KL
            eps = 1e-10
            p = np.array(sweep.reconstructed_distribution) + eps
            q = gt_dist + eps
            p /= p.sum()
            q /= q.sum()

            kl = float(kl_divergence(p, q))

            # Top-3 most-leaked values (highest reconstructed probability)
            top_indices = np.argsort(sweep.reconstructed_distribution)[::-1][:3]
            most_leaked = [sweep.values[i] for i in top_indices]

            leakages.append(AttributeLeakage(
                attribute_name=attr_name,
                sensitivity_score=sweep.sensitivity_score,
                reconstructed_distribution=sweep.reconstructed_distribution,
                ground_truth_distribution=gt_dist.tolist(),
                kl_divergence=kl,
                most_leaked_values=most_leaked,
            ))

        # Sort by KL divergence descending (most leakage first)
        leakages.sort(key=lambda x: x.kl_divergence, reverse=True)

        return ReconstructionReport(attributes=leakages)
