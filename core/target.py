"""Target model wrapper used by audit attacks."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
class TargetModel:
    """Wrap a trained PyTorch classifier with scaling and inference helpers."""

    def __init__(self, model: torch.nn.Module, scaler=None, device: str | None = None):
        self.model = model
        self.scaler = scaler
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    def _ensure_2d(self, X) -> np.ndarray:
        array = np.asarray(X, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def predict_proba(self, X) -> np.ndarray:
        inputs = self._ensure_2d(X)
        if self.scaler is None:
            raise ValueError("Scaler is not set. Load a fitted scaler before calling predict_proba.")
        scaled = self.scaler.transform(inputs)
        tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    @classmethod
    def from_file(cls, path, scaler_path, model_class=None, map_location=None):
        model_path = Path(path)
        scaler = joblib.load(scaler_path)
        device = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(payload, torch.nn.Module):
            model = payload
        elif isinstance(payload, dict) and "model" in payload:
            model = payload["model"]
        elif model_class is not None and isinstance(payload, dict) and "state_dict" in payload:
            model = model_class()
            model.load_state_dict(payload["state_dict"])
        elif model_class is not None and isinstance(payload, dict):
            model = model_class()
            model.load_state_dict(payload)
        else:
            raise ValueError("Unsupported model format. Provide model_class when loading a state dict.")

        return cls(model=model, scaler=scaler, device=device)
