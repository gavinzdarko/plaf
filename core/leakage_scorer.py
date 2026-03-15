"""Privacy leakage scoring — combines membership and attribute signals."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.membership_probe import ProbeResults
from core.attribute_reconstructor import ReconstructionReport


# ── Return types ────────────────────────────────────────────────────────────

@dataclass
class Mitigation:
    priority: str          # CRITICAL / HIGH / MODERATE
    technique: str
    description: str
    estimated_score_reduction: float


@dataclass
class LeakageReport:
    overall_score: float
    membership_leakage: float
    attribute_leakage: float
    risk_level: str
    risk_color: str
    probe_results: ProbeResults
    recon_report: ReconstructionReport


# ── Helpers ─────────────────────────────────────────────────────────────────

def _risk(score: float) -> tuple[str, str]:
    if score <= 20:
        return "Minimal", "green"
    if score <= 40:
        return "Some Exposure", "yellow"
    if score <= 60:
        return "Significant", "orange"
    if score <= 80:
        return "Severe", "red"
    return "Compromised", "darkred"


# ── Core class ──────────────────────────────────────────────────────────────

class LeakageScorer:

    def compute_score(
        self,
        probe_results: ProbeResults,
        recon_report: ReconstructionReport,
        random_baseline_auc: float,
    ) -> LeakageReport:
        # --- Membership leakage (0-100) ---
        raw_membership = 100 * (probe_results.auc - 0.5) / 0.5
        baseline_contribution = 100 * (random_baseline_auc - 0.5) / 0.5
        membership_leakage = max(0.0, min(100.0, raw_membership - baseline_contribution))

        # --- Attribute leakage (0-100) ---
        if recon_report.attributes:
            attr_scores = []
            for attr in recon_report.attributes:
                score = 100 * (1 - attr.kl_divergence)
                attr_scores.append(max(0.0, min(100.0, score)))
            attribute_leakage = sum(attr_scores) / len(attr_scores)
        else:
            attribute_leakage = 0.0

        overall_score = 0.5 * membership_leakage + 0.5 * attribute_leakage
        risk_level, risk_color = _risk(overall_score)

        return LeakageReport(
            overall_score=overall_score,
            membership_leakage=membership_leakage,
            attribute_leakage=attribute_leakage,
            risk_level=risk_level,
            risk_color=risk_color,
            probe_results=probe_results,
            recon_report=recon_report,
        )

    def generate_mitigations(self, report: LeakageReport) -> list[Mitigation]:
        mitigations: list[Mitigation] = []

        if report.membership_leakage > 50:
            mitigations.extend([
                Mitigation(
                    priority="CRITICAL",
                    technique="DP-SGD Retraining",
                    description="Retrain the model using differentially-private SGD (Opacus) to bound per-sample influence.",
                    estimated_score_reduction=30.0,
                ),
                Mitigation(
                    priority="HIGH",
                    technique="Output Perturbation",
                    description="Add calibrated Laplace noise to prediction probabilities before returning them.",
                    estimated_score_reduction=15.0,
                ),
                Mitigation(
                    priority="HIGH",
                    technique="Query Rate Limiting",
                    description="Throttle prediction API to limit the number of queries an attacker can issue.",
                    estimated_score_reduction=10.0,
                ),
            ])

        if report.attribute_leakage > 50:
            mitigations.extend([
                Mitigation(
                    priority="CRITICAL",
                    technique="Confidence Rounding",
                    description="Round output probabilities to 1 decimal place to reduce information in the response.",
                    estimated_score_reduction=20.0,
                ),
                Mitigation(
                    priority="HIGH",
                    technique="Top-K Only",
                    description="Return only the top predicted class instead of the full probability vector.",
                    estimated_score_reduction=25.0,
                ),
                Mitigation(
                    priority="HIGH",
                    technique="Temperature Scaling",
                    description="Apply temperature scaling (T>1) to soften output distributions and obscure memorisation.",
                    estimated_score_reduction=12.0,
                ),
            ])

        # Always recommend
        mitigations.extend([
            Mitigation(
                priority="MODERATE",
                technique="Reduce Overfitting",
                description="Apply regularization (dropout, weight decay, early stopping) to reduce memorisation.",
                estimated_score_reduction=20.0,
            ),
            Mitigation(
                priority="MODERATE",
                technique="Audit Before Deployment",
                description="Run this audit framework as part of the CI/CD pipeline before every model release.",
                estimated_score_reduction=0.0,
            ),
        ])

        return mitigations
