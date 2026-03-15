"""Defense wrappers that modify a TargetModel's prediction behavior."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from core.target import TargetModel


class DefendedModel:
    """TargetModel-compatible wrapper that intercepts predict_proba."""

    def __init__(self, base: TargetModel, transform):
        self._base = base
        self._transform = transform
        # Expose same attributes for compatibility
        self.model = base.model
        self.scaler = base.scaler
        self.device = base.device

    def predict_proba(self, X) -> np.ndarray:
        raw = self._base.predict_proba(X)
        return self._transform(raw)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def _renormalize(probs: np.ndarray) -> np.ndarray:
    """Re-normalize rows to sum to 1, handling edge cases."""
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return probs / row_sums


# ── Defense factories ───────────────────────────────────────────────────────

def apply_output_noise(model: TargetModel, scale: float = 0.1) -> DefendedModel:
    """Add Laplace noise to outputs, then re-normalize."""
    def transform(probs: np.ndarray) -> np.ndarray:
        noise = np.random.laplace(loc=0, scale=scale, size=probs.shape)
        noisy = np.clip(probs + noise, 0.0, None)
        return _renormalize(noisy)
    return DefendedModel(model, transform)


def apply_confidence_rounding(model: TargetModel, decimals: int = 1) -> DefendedModel:
    """Round output probabilities to *decimals* places."""
    def transform(probs: np.ndarray) -> np.ndarray:
        rounded = np.round(probs, decimals)
        return _renormalize(rounded)
    return DefendedModel(model, transform)


def apply_top_k_only(model: TargetModel, k: int = 1) -> DefendedModel:
    """Zero out all but the top-k probabilities per row."""
    def transform(probs: np.ndarray) -> np.ndarray:
        result = np.zeros_like(probs)
        for i, row in enumerate(probs):
            top_indices = np.argsort(row)[-k:]
            result[i, top_indices] = row[top_indices]
        return _renormalize(result)
    return DefendedModel(model, transform)


def apply_temperature_scaling(model: TargetModel, temperature: float = 3.0) -> DefendedModel:
    """Divide logits by temperature before softmax."""
    def transform(probs: np.ndarray) -> np.ndarray:
        # Convert probabilities back to logits, apply temperature, re-softmax
        eps = 1e-12
        logits = np.log(np.clip(probs, eps, None))
        scaled = logits / temperature
        # Stable softmax
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
    return DefendedModel(model, transform)


def load_dp_model(path: str | Path, scaler_path: str | Path) -> TargetModel:
    """Load an Opacus-trained DP model from disk."""
    return TargetModel.from_file(path, scaler_path)
