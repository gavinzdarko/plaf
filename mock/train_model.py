"""Train overfit, regularized, and random-baseline models for PLAF."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from config import DataConfig, ModelConfig

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


class PLAFMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.network(x)


class MemorizingMLP(nn.Module):
    """Larger model designed to memorize training data for attack demos."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.network(x)

    def forward(self, x):
        return self.network(x)


def load_dataset(filename: str):
    config = DataConfig()
    data = pd.read_csv(DATA_DIR / filename)
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


def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        preds = logits.argmax(dim=1).cpu().numpy()
    return float((preds == y).mean())


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    early_stopping: bool = False,
    patience: int = 8,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(X_train_tensor))

        for start in range(0, len(X_train_tensor), batch_size):
            indices = permutation[start : start + batch_size]
            batch_X = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val, dtype=torch.float32))
            val_loss = criterion(val_logits, torch.tensor(y_val, dtype=torch.long)).item()

        if early_stopping:
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if early_stopping:
        model.load_state_dict(best_state)

    return model


def save_model(model: nn.Module, path: Path) -> None:
    torch.save(model, path)


def run_training(name: str, filename: str, dropout: float, weight_decay: float, epochs: int, batch_size: int, early_stopping: bool, output_name: str):
    X_train, X_test, y_train, y_test = load_dataset(filename)
    model = PLAFMLP(input_dim=X_train.shape[1], dropout=dropout)
    trained = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        lr=0.001,
        weight_decay=weight_decay,
        batch_size=batch_size,
        early_stopping=early_stopping,
    )
    train_acc = accuracy(trained, X_train, y_train)
    test_acc = accuracy(trained, X_test, y_test)
    save_model(trained, MODEL_DIR / output_name)
    print(f"{name}: train_acc={train_acc:.4f} test_acc={test_acc:.4f}")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config = ModelConfig()

    run_training(
        name="Model A (Overfit)",
        filename="healthcare_data.csv",
        dropout=0.0,
        weight_decay=0.0,
        epochs=config.epochs_overfit,
        batch_size=int((1 - DataConfig().test_split) * DataConfig().n_records),
        early_stopping=False,
        output_name="model_overfit.pt",
    )
    run_training(
        name="Model B (Regularized)",
        filename="healthcare_data.csv",
        dropout=config.dropout,
        weight_decay=config.l2_reg,
        epochs=config.epochs_regularized,
        batch_size=128,
        early_stopping=True,
        output_name="model_regularized.pt",
    )
    run_training(
        name="Model R (Random Baseline)",
        filename="healthcare_data_random.csv",
        dropout=0.0,
        weight_decay=0.0,
        epochs=config.epochs_overfit,
        batch_size=int((1 - DataConfig().test_split) * DataConfig().n_records),
        early_stopping=False,
        output_name="model_random.pt",
    )


if __name__ == "__main__":
    main()
