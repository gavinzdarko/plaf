"""Train a differentially private model for PLAF using Opacus."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import DPConfig, DataConfig

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = [
    "age",
    "bmi",
    "num_prior_admissions",
    "primary_diagnosis",
    "insurance_type",
    "zip_code_region",
    "length_of_stay",
    "num_medications",
    "has_chronic_condition",
]


class DPMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.network(x)


def load_dataset():
    config = DataConfig()
    data = pd.read_csv(DATA_DIR / "healthcare_data.csv")
    transformer = joblib.load(DATA_DIR / "scaler.pkl")
    X = transformer.transform(data[FEATURE_COLUMNS]).astype(np.float32)
    y = data["readmission_30d"].to_numpy(dtype=np.int64)
    return train_test_split(
        X,
        y,
        test_size=config.test_split,
        random_state=config.random_seed,
        stratify=y,
    )


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        preds = logits.argmax(dim=1).cpu().numpy()
    return float((preds == y).mean())


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dp_config = DPConfig()
    X_train, X_test, y_train, y_test = load_dataset()

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = DPMLP(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=dp_config.epochs,
        target_epsilon=dp_config.epsilon,
        target_delta=dp_config.delta,
        max_grad_norm=dp_config.max_grad_norm,
    )

    for _ in range(dp_config.epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    epsilon_spent = privacy_engine.get_epsilon(dp_config.delta)
    train_acc = evaluate(model, X_train, y_train)
    test_acc = evaluate(model, X_test, y_test)
    torch.save(model, MODEL_DIR / "model_dp.pt")
    print(f"Model C (DP-SGD): epsilon={epsilon_spent:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
