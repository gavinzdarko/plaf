"""End-to-end test of PLAF attack modules.

Loads the overfit model, runs membership inference and attribute
reconstruction, validates with the random baseline, and prints results.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split

from config import AttackConfig, DataConfig, SENSITIVE_FEATURES
from core.target import TargetModel
from core.membership_probe import MembershipProbe
from core.attribute_reconstructor import AttributeReconstructor
from validation.random_baseline import validate_random_baseline
from mock.train_model import PLAFMLP  # needed for torch.load unpickling

# ── Constants ───────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "age", "bmi", "num_prior_admissions", "primary_diagnosis",
    "insurance_type", "zip_code_region", "length_of_stay",
    "num_medications", "has_chronic_condition",
]

DIAGNOSES = ["Diabetes", "Heart Disease", "Cancer", "Respiratory", "Mental Health", "Other"]
INSURANCE_TYPES = ["Private", "Medicare", "Medicaid", "Uninsured"]
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Northwest", "Central", "Pacific"]


# ── Helpers ─────────────────────────────────────────────────────────────────

class IdentityScaler:
    """No-op scaler for already-transformed data."""
    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


class PipelineTargetModel:
    """Wraps a model + sklearn Pipeline to accept raw mixed-type arrays.

    Used by the attribute reconstructor which must sweep raw feature values
    (including categorical strings) through the full preprocessing pipeline.
    """

    def __init__(self, model: torch.nn.Module, pipeline, feature_columns: list[str]):
        self.model = model
        self.pipeline = pipeline
        self.feature_columns = feature_columns
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, X) -> np.ndarray:
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.feature_columns)
            # Restore numeric dtypes that get lost in object arrays
            for col in ["age", "bmi", "num_prior_admissions", "length_of_stay",
                        "num_medications", "has_chronic_condition"]:
                df[col] = pd.to_numeric(df[col])
        elif isinstance(X, pd.DataFrame):
            df = X
        else:
            raise TypeError(f"Expected ndarray or DataFrame, got {type(X)}")

        transformed = self.pipeline.transform(df).astype(np.float32)
        tensor = torch.tensor(transformed, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def build_feature_schema(df: pd.DataFrame) -> dict:
    """Build feature schema for the attribute reconstructor."""
    return {
        "primary_diagnosis": {
            "type": "categorical",
            "values": DIAGNOSES,
            "sensitive": True,
            "col_idx": FEATURE_COLUMNS.index("primary_diagnosis"),
        },
        "insurance_type": {
            "type": "categorical",
            "values": INSURANCE_TYPES,
            "sensitive": True,
            "col_idx": FEATURE_COLUMNS.index("insurance_type"),
        },
        "zip_code_region": {
            "type": "categorical",
            "values": REGIONS,
            "sensitive": True,
            "col_idx": FEATURE_COLUMNS.index("zip_code_region"),
        },
        "bmi": {
            "type": "continuous",
            "values": list(range(15, 51, 5)),  # binned for tractable sweep
            "sensitive": True,
            "col_idx": FEATURE_COLUMNS.index("bmi"),
        },
        "has_chronic_condition": {
            "type": "binary",
            "values": [0, 1],
            "sensitive": True,
            "col_idx": FEATURE_COLUMNS.index("has_chronic_condition"),
        },
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    data_config = DataConfig()
    attack_config = AttackConfig()

    # Load data
    df = pd.read_csv(PROJECT_ROOT / "data" / "healthcare_data.csv")
    pipeline = joblib.load(PROJECT_ROOT / "data" / "scaler.pkl")
    y = df["readmission_30d"].values

    # Transform features (same split as training)
    X_transformed = pipeline.transform(df[FEATURE_COLUMNS]).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y,
        test_size=data_config.test_split,
        random_state=data_config.random_seed,
        stratify=y,
    )

    # Also split raw data (for attribute reconstruction)
    X_raw = df[FEATURE_COLUMNS].values  # object array preserving strings
    X_train_raw, X_test_raw, _, _ = train_test_split(
        X_raw, y,
        test_size=data_config.test_split,
        random_state=data_config.random_seed,
        stratify=y,
    )

    # Load overfit model
    model_overfit = torch.load(
        PROJECT_ROOT / "models" / "model_overfit.pt",
        map_location="cpu", weights_only=False,
    )
    model_overfit.eval()

    # ── 1. Membership Probe ─────────────────────────────────────────────
    print("=" * 60)
    print("MEMBERSHIP INFERENCE ATTACK")
    print("=" * 60)

    target = TargetModel(model=model_overfit, scaler=IdentityScaler())
    probe = MembershipProbe(target, attack_config)

    # Calibrate with non-member (test) data
    probe.calibrate_baseline(X_test)

    # Candidates: train (members) + test (non-members)
    n = min(2000, len(X_train), len(X_test))
    candidate_data = np.vstack([X_train[:n], X_test[:n]])
    candidate_labels = np.array([1] * n + [0] * n)

    result = probe.probe(candidate_data, candidate_labels)
    print(f"  AUC:              {result.auc:.4f}")
    print(f"  TPR @ 5% FPR:     {result.tpr_at_5fpr:.4f}")
    print(f"  Detected members: {result.n_detected_members}")

    # Query-budget analysis
    budget = probe.run_query_budget_analysis(candidate_data, candidate_labels)
    print("\n  Query-budget curve:")
    for n_q, auc in sorted(budget.items()):
        print(f"    {n_q:>5} queries → AUC {auc:.4f}")

    # ── 2. Attribute Reconstruction ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("ATTRIBUTE RECONSTRUCTION ATTACK")
    print("=" * 60)

    pipeline_target = PipelineTargetModel(model_overfit, pipeline, FEATURE_COLUMNS)
    feature_schema = build_feature_schema(df)
    reconstructor = AttributeReconstructor(pipeline_target, attack_config, feature_schema)

    report = reconstructor.reconstruct_all(X_train_raw)

    print(f"\n  Top 3 leaked attributes (by KL divergence):")
    for attr in report.attributes[:3]:
        print(f"    {attr.attribute_name:25s}  KL={attr.kl_divergence:.4f}  "
              f"sensitivity={attr.sensitivity_score:.4f}  "
              f"leaked={attr.most_leaked_values}")

    print(f"\n  All attributes:")
    for attr in report.attributes:
        print(f"    {attr.attribute_name:25s}  KL={attr.kl_divergence:.4f}")

    # ── 3. Random Baseline Validation ───────────────────────────────────
    print("\n" + "=" * 60)
    print("RANDOM BASELINE VALIDATION")
    print("=" * 60)

    model_random = torch.load(
        PROJECT_ROOT / "models" / "model_random.pt",
        map_location="cpu", weights_only=False,
    )
    model_random.eval()
    target_random = TargetModel(model=model_random, scaler=IdentityScaler())

    baseline = validate_random_baseline(target_random, candidate_data, candidate_labels, attack_config)
    print(f"  Random-baseline AUC: {baseline['random_auc']:.4f}")
    print(f"  Valid (AUC ≤ 0.6):   {baseline['valid']}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Membership AUC:       {result.auc:.4f}")
    print(f"  TPR @ 5% FPR:         {result.tpr_at_5fpr:.4f}")
    top3 = report.attributes[:3]
    print(f"  Top leaked attrs:     {', '.join(a.attribute_name for a in top3)}")
    for a in top3:
        print(f"    {a.attribute_name}: KL={a.kl_divergence:.4f}")
    print(f"  Random baseline AUC:  {baseline['random_auc']:.4f} ({'VALID' if baseline['valid'] else 'INVALID'})")


if __name__ == "__main__":
    main()
