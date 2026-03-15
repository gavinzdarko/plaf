"""Generate synthetic healthcare datasets for PLAF demos."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DataConfig

DATA_DIR = PROJECT_ROOT / "data"

DIAGNOSES = ["Diabetes", "Heart Disease", "Cancer", "Respiratory", "Mental Health", "Other"]
INSURANCE_TYPES = ["Private", "Medicare", "Medicaid", "Uninsured"]
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Northwest", "Central", "Pacific"]


def _weighted_choice(rng: np.random.Generator, items: list[str], probs: list[float], size: int) -> np.ndarray:
    return rng.choice(items, size=size, p=np.asarray(probs, dtype=float))


def generate_dataset(config: DataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    size = config.n_records

    data = pd.DataFrame(
        {
            "age": rng.integers(18, 91, size=size),
            "bmi": np.round(rng.normal(loc=29.0, scale=6.0, size=size).clip(15, 50), 1),
            "num_prior_admissions": rng.integers(0, 16, size=size),
            "primary_diagnosis": _weighted_choice(rng, DIAGNOSES, [0.23, 0.18, 0.14, 0.16, 0.12, 0.17], size),
            "insurance_type": _weighted_choice(rng, INSURANCE_TYPES, [0.42, 0.2, 0.23, 0.15], size),
            "zip_code_region": _weighted_choice(rng, REGIONS, [0.12, 0.15, 0.14, 0.11, 0.14, 0.09, 0.12, 0.13], size),
            "length_of_stay": rng.integers(1, 31, size=size),
            "num_medications": rng.integers(0, 21, size=size),
            "has_chronic_condition": rng.binomial(1, 0.47, size=size),
        }
    )

    diagnosis_risk = data["primary_diagnosis"].map(
        {
            "Diabetes": 0.75,
            "Heart Disease": 0.7,
            "Cancer": 0.38,
            "Respiratory": 0.42,
            "Mental Health": 0.35,
            "Other": 0.22,
        }
    )
    insurance_risk = data["insurance_type"].map(
        {"Private": 0.18, "Medicare": 0.28, "Medicaid": 0.62, "Uninsured": 0.7}
    )
    region_risk = data["zip_code_region"].map(
        {
            "Northeast": 0.26,
            "Southeast": 0.36,
            "Midwest": 0.3,
            "Southwest": 0.31,
            "West": 0.24,
            "Northwest": 0.22,
            "Central": 0.34,
            "Pacific": 0.25,
        }
    )

    risk_score = (
        -4.7
        + 0.028 * data["age"]
        + 0.08 * data["bmi"]
        + 0.17 * data["num_prior_admissions"]
        + 0.09 * data["length_of_stay"]
        + 0.12 * data["num_medications"]
        + 1.35 * data["has_chronic_condition"]
        + diagnosis_risk
        + insurance_risk
        + region_risk
    )
    probs = 1 / (1 + np.exp(-risk_score))
    data["readmission_30d"] = rng.binomial(1, probs.clip(0.03, 0.97))

    return enforce_imbalance(data, rng)


def enforce_imbalance(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    positives = data.index[data["readmission_30d"] == 1]
    if len(positives) == 0:
        raise RuntimeError("No positive samples generated; adjust score calibration.")

    target_diag = int(np.ceil(0.7 * len(positives)))
    diag_mask = data.loc[positives, "primary_diagnosis"].isin(["Diabetes", "Heart Disease"])
    diag_current = int(diag_mask.sum())
    if diag_current < target_diag:
        candidates = positives[~diag_mask.to_numpy()]
        chosen = rng.choice(candidates, size=target_diag - diag_current, replace=False)
        data.loc[chosen, "primary_diagnosis"] = rng.choice(["Diabetes", "Heart Disease"], size=len(chosen), p=[0.55, 0.45])

    target_ins = int(np.ceil(0.6 * len(positives)))
    ins_mask = data.loc[positives, "insurance_type"].isin(["Medicaid", "Uninsured"])
    ins_current = int(ins_mask.sum())
    if ins_current < target_ins:
        candidates = positives[~ins_mask.to_numpy()]
        chosen = rng.choice(candidates, size=target_ins - ins_current, replace=False)
        data.loc[chosen, "insurance_type"] = rng.choice(["Medicaid", "Uninsured"], size=len(chosen), p=[0.6, 0.4])

    return data


def save_scaler(data: pd.DataFrame) -> None:
    feature_columns = [column for column in data.columns if column != "readmission_30d"]
    numeric_columns = ["age", "bmi", "num_prior_admissions", "length_of_stay", "num_medications", "has_chronic_condition"]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), numeric_columns),
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
                    ]
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    pipeline.fit(data[feature_columns])
    joblib.dump(pipeline, DATA_DIR / "scaler.pkl")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = DataConfig()

    data = generate_dataset(config)
    random_data = data.copy()
    rng = np.random.default_rng(config.random_seed + 1)
    random_data["readmission_30d"] = rng.integers(0, 2, size=len(random_data))

    data.to_csv(DATA_DIR / "healthcare_data.csv", index=False)
    random_data.to_csv(DATA_DIR / "healthcare_data_random.csv", index=False)
    save_scaler(data)

    positive_mask = data["readmission_30d"] == 1
    diagnosis_ratio = data.loc[positive_mask, "primary_diagnosis"].isin(["Diabetes", "Heart Disease"]).mean()
    insurance_ratio = data.loc[positive_mask, "insurance_type"].isin(["Medicaid", "Uninsured"]).mean()
    print(f"Saved {len(data)} records to {DATA_DIR / 'healthcare_data.csv'}")
    print(f"Positive diagnosis concentration: {diagnosis_ratio:.3f}")
    print(f"Positive insurance concentration: {insurance_ratio:.3f}")


if __name__ == "__main__":
    main()
