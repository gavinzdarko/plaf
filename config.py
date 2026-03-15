"""Central configuration for the Privacy Leakage Audit Framework."""

from dataclasses import dataclass, field


@dataclass
class AttackConfig:
    n_queries: int = 5000
    confidence_threshold: float = 0.95
    z_score_threshold: float = 2.5
    n_base_points: int = 50


@dataclass
class DataConfig:
    n_records: int = 5000
    test_split: float = 0.2
    random_seed: int = 42


@dataclass
class ModelConfig:
    hidden_layers: list[int] = field(default_factory=lambda: [128, 64, 32])
    epochs_overfit: int = 300
    epochs_regularized: int = 50
    dropout: float = 0.3
    l2_reg: float = 0.01


@dataclass
class DPConfig:
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    epochs: int = 100


SENSITIVE_FEATURES = [
    "primary_diagnosis",
    "insurance_type",
    "zip_code_region",
    "bmi",
    "has_chronic_condition",
]
