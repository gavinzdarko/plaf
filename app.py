"""PLAF — Privacy Leakage Audit Framework — Streamlit Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import AttackConfig, DataConfig, SENSITIVE_FEATURES
from core.target import TargetModel


class IdentityScaler:
    """No-op scaler for already-transformed data."""
    def transform(self, X):
        return np.asarray(X, dtype=np.float32)
from core.membership_probe import MembershipProbe, ProbeResults
from core.attribute_reconstructor import AttributeReconstructor, ReconstructionReport
from core.leakage_scorer import LeakageScorer, LeakageReport
from core.defenses import (
    apply_output_noise,
    apply_confidence_rounding,
    apply_top_k_only,
    apply_temperature_scaling,
    load_dp_model,
)
from llm.llm_probe import LLMTarget, LLMMembershipProbe
from llm.gemini_probe import GeminiTarget, GeminiMembershipProbe
from llm.openai_probe import OpenAITarget, OpenAIMembershipProbe
from llm.extraction_probe import ExtractionProbe, ExtractionReport

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PLAF",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp { background-color: #0E1117; }
    .main-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        color: #E0E0E0;
        padding-bottom: 0;
        margin-bottom: 0;
    }
    .tagline {
        font-family: 'JetBrains Mono', monospace;
        color: #888;
        font-size: 0.95rem;
        margin-top: 0;
    }
    .risk-card {
        border-radius: 8px;
        padding: 16px 20px;
        font-family: 'JetBrains Mono', monospace;
        text-align: center;
    }
    .metric-box {
        background: #1A1D23;
        border: 1px solid #2A2D35;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-label { color: #888; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #E0E0E0; font-size: 1.6rem; font-weight: 700; }
    .mitigation-critical { border-left: 4px solid #FF4444; background: #1A1D23; padding: 12px 16px; margin: 6px 0; border-radius: 0 6px 6px 0; }
    .mitigation-high { border-left: 4px solid #FF8800; background: #1A1D23; padding: 12px 16px; margin: 6px 0; border-radius: 0 6px 6px 0; }
    .mitigation-moderate { border-left: 4px solid #FFCC00; background: #1A1D23; padding: 12px 16px; margin: 6px 0; border-radius: 0 6px 6px 0; }
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        color: #C0C0C0;
        font-size: 1.2rem;
        border-bottom: 1px solid #2A2D35;
        padding-bottom: 6px;
        margin-top: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(family="JetBrains Mono, monospace"),
    margin=dict(l=40, r=40, t=40, b=40),
)

MODELS = {
    "Overfit (Model A)": MODEL_DIR / "model_overfit.pt",
    "Regularized (Model B)": MODEL_DIR / "model_regularized.pt",
    "DP-SGD (Model C)": MODEL_DIR / "model_dp.pt",
}

DIAGNOSES = ["Diabetes", "Heart Disease", "Cancer", "Respiratory", "Mental Health", "Other"]
INSURANCE_TYPES = ["Private", "Medicare", "Medicaid", "Uninsured"]
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Northwest", "Central", "Pacific"]


# ── Data loading helpers ────────────────────────────────────────────────────

@st.cache_data
def load_data():
    import joblib
    data = pd.read_csv(DATA_DIR / "healthcare_data.csv")
    random_data = pd.read_csv(DATA_DIR / "healthcare_data_random.csv")
    scaler = joblib.load(DATA_DIR / "scaler.pkl")
    feature_cols = [c for c in data.columns if c != "readmission_30d"]
    X = scaler.transform(data[feature_cols]).astype(np.float32)
    X_random = scaler.transform(random_data[feature_cols]).astype(np.float32)
    y = data["readmission_30d"].to_numpy(dtype=np.int64)
    y_random = random_data["readmission_30d"].to_numpy(dtype=np.int64)
    return data, X, y, X_random, y_random, scaler


@st.cache_resource
def load_model(model_path: str, scaler_path: str):
    """Load model with IdentityScaler since data is already transformed."""
    import torch
    from mock.train_model import PLAFMLP, MemorizingMLP  # noqa: needed for unpickling
    from mock.train_model_dp import DPMLP  # noqa: needed for DP model unpickling
    import opacus  # noqa: needed for GradSampleModule unpickling
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    # Opacus wraps models in GradSampleModule — unwrap to get the actual model
    if hasattr(model, '_module'):
        model = model._module
    model.eval()
    return TargetModel(model=model, scaler=IdentityScaler())


def build_feature_schema(data: pd.DataFrame) -> dict:
    """Build the feature schema expected by AttributeReconstructor."""
    feature_cols = [c for c in data.columns if c != "readmission_30d"]
    schema = {}

    value_maps = {
        "primary_diagnosis": DIAGNOSES,
        "insurance_type": INSURANCE_TYPES,
        "zip_code_region": REGIONS,
    }

    for i, col in enumerate(feature_cols):
        if col in value_maps:
            vals = list(range(len(value_maps[col])))  # encoded values after transform
        elif col == "has_chronic_condition":
            vals = [0, 1]
        elif col == "bmi":
            vals = list(np.arange(15, 51, 5, dtype=float))
        else:
            continue

        schema[col] = {
            "col_idx": i,
            "values": vals,
            "sensitive": col in SENSITIVE_FEATURES,
        }

    return schema


# ── Attack pipeline ─────────────────────────────────────────────────────────

def run_audit(
    target: TargetModel,
    X: np.ndarray,
    y: np.ndarray,
    X_random: np.ndarray,
    y_random: np.ndarray,
    data: pd.DataFrame,
    config: AttackConfig,
) -> tuple[LeakageReport, dict[int, float]]:
    """Run full membership + attribute attack and score."""
    from sklearn.model_selection import train_test_split

    # Split into members (train) and non-members (test)
    dc = DataConfig()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=dc.test_split, random_state=dc.random_seed, stratify=y,
    )

    # --- Membership probe ---
    probe = MembershipProbe(target, config)
    probe.calibrate_baseline(X_test[:500])

    # The overfit model was trained on only the first 300 records of X_train
    # Use correct membership labels for accurate AUC
    overfit_idx_path = MODEL_DIR / "overfit_train_indices.npy"
    if overfit_idx_path.exists():
        n_actual_train = len(np.load(overfit_idx_path))
    else:
        n_actual_train = len(X_train)  # fallback: assume all of X_train

    n_members = min(n_actual_train, config.n_queries)
    n_nonmembers = min(len(X_test), n_members)
    candidate = np.vstack([X_train[:n_members], X_test[:n_nonmembers]])
    labels = np.concatenate([np.ones(n_members), np.zeros(n_nonmembers)])
    probe_results = probe.probe(candidate, labels)

    # Query-budget analysis
    budget_results = probe.run_query_budget_analysis(candidate, labels)

    # --- Random baseline AUC ---
    random_model = load_model(
        str(MODEL_DIR / "model_random.pt"),
        str(DATA_DIR / "scaler.pkl"),
    )
    random_probe = MembershipProbe(random_model, config)
    random_probe.calibrate_baseline(X_test[:500])
    random_results = random_probe.probe(candidate, labels)
    random_baseline_auc = random_results.auc

    # --- Attribute reconstruction ---
    feature_schema = build_feature_schema(data)
    reconstructor = AttributeReconstructor(target, config, feature_schema)
    recon_report = reconstructor.reconstruct_all(X)

    # --- Score ---
    scorer = LeakageScorer()
    report = scorer.compute_score(probe_results, recon_report, random_baseline_auc)

    return report, budget_results


# ── Visualization helpers ───────────────────────────────────────────────────

RISK_COLORS = {
    "green": "#22C55E",
    "yellow": "#EAB308",
    "orange": "#F97316",
    "red": "#EF4444",
    "darkred": "#991B1B",
}


def gauge_chart(score: float, risk_color: str) -> go.Figure:
    color = RISK_COLORS.get(risk_color, "#EF4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="/100", font=dict(size=40, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#444"),
            bar=dict(color=color),
            bgcolor="#1A1D23",
            borderwidth=0,
            steps=[
                dict(range=[0, 20], color="#1A3A1A"),
                dict(range=[20, 40], color="#3A3A1A"),
                dict(range=[40, 60], color="#3A2A1A"),
                dict(range=[60, 80], color="#3A1A1A"),
                dict(range=[80, 100], color="#4A1010"),
            ],
        ),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280)
    return fig


def roc_chart(fpr: list, tpr: list, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines", name=f"Attack ROC (AUC={auc:.3f})",
        line=dict(color="#EF4444", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Random (AUC=0.5)",
        line=dict(color="#555", dash="dash"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=350,
        showlegend=True,
        legend=dict(x=0.55, y=0.1),
    )
    return fig


def confidence_histogram(members: list, nonmembers: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=members, name="Members", opacity=0.7,
        marker_color="#EF4444", nbinsx=40,
    ))
    fig.add_trace(go.Histogram(
        x=nonmembers, name="Non-members", opacity=0.7,
        marker_color="#3B82F6", nbinsx=40,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="overlay",
        xaxis_title="Max Confidence",
        yaxis_title="Count",
        height=350,
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def attribute_bar_chart(attr) -> go.Figure:
    n_vals = len(attr.reconstructed_distribution)
    x_labels = [f"v{i}" for i in range(n_vals)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_labels, y=attr.reconstructed_distribution,
        name="Reconstructed", marker_color="#EF4444", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=x_labels, y=attr.ground_truth_distribution,
        name="Actual", marker_color="#3B82F6", opacity=0.85,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title=dict(text=attr.attribute_name, font=dict(size=14)),
        height=280,
        showlegend=True,
    )
    return fig


def sensitivity_heatmap(attributes) -> go.Figure:
    names = [a.attribute_name for a in attributes]
    scores = [[a.sensitivity_score] for a in attributes]
    fig = go.Figure(go.Heatmap(
        z=scores,
        y=names,
        x=["Sensitivity"],
        colorscale="YlOrRd",
        showscale=True,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(200, 50 * len(names)),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def budget_chart(budget_data: dict[str, dict[int, float]]) -> go.Figure:
    colors = ["#EF4444", "#3B82F6", "#22C55E", "#EAB308", "#A855F7"]
    fig = go.Figure()
    for i, (label, data) in enumerate(budget_data.items()):
        queries = sorted(data.keys())
        aucs = [data[q] for q in queries]
        fig.add_trace(go.Scatter(
            x=queries, y=aucs,
            mode="lines+markers", name=label,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#555", annotation_text="Random")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Number of Queries",
        yaxis_title="Attack AUC",
        height=350,
    )
    return fig


def perplexity_histogram(members: list, nonmembers: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=members, name="Known Training Data", opacity=0.7,
        marker_color="#EF4444", nbinsx=30,
    ))
    fig.add_trace(go.Histogram(
        x=nonmembers, name="Novel Text", opacity=0.7,
        marker_color="#3B82F6", nbinsx=30,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="overlay",
        xaxis_title="Perplexity (lower = more memorized)",
        yaxis_title="Count",
        height=350,
        legend=dict(x=0.55, y=0.98),
    )
    return fig


@st.cache_resource
def load_llm_target(model_path: str):
    return LLMTarget(model_path)


# ── Defense comparison pipeline ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_defense_comparison(
    _model_path: str,
    _scaler_path: str,
    n_queries: int,
    confidence_threshold: float,
    z_score_threshold: float,
):
    """Run the full audit for each defense and return comparison data."""
    from sklearn.model_selection import train_test_split

    data_df, X, y, X_random, y_random, scaler = load_data()
    dc = DataConfig()
    config = AttackConfig(
        n_queries=n_queries,
        confidence_threshold=confidence_threshold,
        z_score_threshold=z_score_threshold,
    )

    base_model = load_model(_model_path, _scaler_path)

    defenses = {
        "None": base_model,
        "Output Noise (0.1)": apply_output_noise(base_model, scale=0.1),
        "Confidence Rounding": apply_confidence_rounding(base_model, decimals=1),
        "Top-K Only": apply_top_k_only(base_model, k=1),
        "Temperature (T=3)": apply_temperature_scaling(base_model, temperature=3.0),
    }

    # Add DP-SGD model if it exists and we're not already auditing it
    dp_path = MODEL_DIR / "model_dp.pt"
    if dp_path.exists() and str(dp_path) != _model_path:
        dp_model = load_model(str(dp_path), _scaler_path)
        defenses["DP-SGD Model"] = dp_model

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=dc.test_split, random_state=dc.random_seed, stratify=y,
    )
    n_cand = min(config.n_queries, len(X_train), len(X_test))
    candidate = np.vstack([X_train[:n_cand], X_test[:n_cand]])
    labels = np.concatenate([np.ones(n_cand), np.zeros(n_cand)])

    results = {}
    budget_data = {}

    for name, defended in defenses.items():
        probe = MembershipProbe(defended, config)
        probe.calibrate_baseline(X_test[:500])
        pr = probe.probe(candidate, labels)

        # Accuracy on test set
        preds = defended.predict(X_test)
        acc = float((preds == y_test).mean())

        budget = probe.run_query_budget_analysis(candidate, labels)
        budget_data[name] = budget
        results[name] = {"auc": pr.auc, "accuracy": acc, "leakage": 100 * (pr.auc - 0.5) / 0.5}

    return results, budget_data


# ═════════════════════════════════════════════════════════════════════════════
#                              MAIN UI
# ═════════════════════════════════════════════════════════════════════════════

# Header
st.markdown('<p class="main-header">PLAF</p>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Privacy Leakage Audit Framework — Quantify, visualize, and mitigate ML model memorisation</p>', unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Model Selection")
    model_name = st.selectbox("Target Model", list(MODELS.keys()))

    st.markdown("### Attack Configuration")
    n_queries = st.slider("Query Budget", 500, 10000, 5000, step=500)
    conf_thresh = st.slider("Confidence Threshold", 0.5, 1.0, 0.95, step=0.01)
    z_thresh = st.slider("Z-Score Threshold", 1.0, 5.0, 2.5, step=0.1)

    st.markdown("---")
    run_audit_btn = st.button("Run Audit", type="primary", use_container_width=True)
    compare_btn = st.button("Compare Defenses", use_container_width=True)

    st.markdown("---")
    st.markdown("### LLM Audit")
    st.caption("Requires GPT-2 model downloaded to models/gpt2/")
    run_llm_btn = st.button("Run LLM Audit", use_container_width=True)

    st.markdown("---")
    st.markdown("### Gemini Audit")
    st.caption("Tests Gemini API for training data memorization")
    gemini_n = st.slider("Gemini samples per group", 5, 50, 10, step=5)
    run_gemini_btn = st.button("Run Gemini Audit", use_container_width=True)

    st.markdown("---")
    st.markdown("### OpenAI Audit")
    st.caption("Tests OpenAI API via real logprobs")
    openai_n = st.slider("OpenAI samples per group", 5, 50, 10, step=5)
    run_openai_btn = st.button("Run OpenAI Audit", use_container_width=True)
    run_extraction_btn = st.button("Run Extraction Attack", use_container_width=True)

# ── Load data ───────────────────────────────────────────────────────────────

data_df, X, y, X_random, y_random, scaler = load_data()
model_path = str(MODELS[model_name])
scaler_path = str(DATA_DIR / "scaler.pkl")

# ── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["Classification Audit", "LLM Audit (GPT-2)", "Gemini Audit (Live API)", "OpenAI Audit (Live API)"])

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — Classification Audit
# ═══════════════════════════════════════════════════════════════════════════

with tab1:

    # ── Run Audit ───────────────────────────────────────────────────────────────

    if run_audit_btn:
        config = AttackConfig(
            n_queries=n_queries,
            confidence_threshold=conf_thresh,
            z_score_threshold=z_thresh,
        )
        target = load_model(model_path, scaler_path)

        with st.spinner("Running membership inference probe..."):
            progress = st.progress(0, text="Initialising audit...")
            progress.progress(10, text="Loading model and data...")
            report, budget_results = run_audit(
                target, X, y, X_random, y_random, data_df, config,
            )
            progress.progress(80, text="Generating mitigations...")
            scorer = LeakageScorer()
            mitigations = scorer.generate_mitigations(report)
            progress.progress(100, text="Audit complete.")

        st.session_state["report"] = report
        st.session_state["budget_results"] = budget_results
        st.session_state["mitigations"] = mitigations

    # ── Display Results ─────────────────────────────────────────────────────────

    if "report" in st.session_state:
        report: LeakageReport = st.session_state["report"]
        budget_results = st.session_state["budget_results"]
        mitigations = st.session_state["mitigations"]
        pr = report.probe_results
        rr = report.recon_report

        # SECTION 1 — Leakage Score
        st.markdown('<p class="section-header">LEAKAGE SCORE</p>', unsafe_allow_html=True)
        col_gauge, col_metrics = st.columns([1, 1])

        with col_gauge:
            st.plotly_chart(gauge_chart(report.overall_score, report.risk_color), use_container_width=True)

        with col_metrics:
            rc = RISK_COLORS.get(report.risk_color, "#EF4444")
            st.markdown(f"""
            <div class="risk-card" style="border: 2px solid {rc}; color: {rc};">
                <div style="font-size: 1.8rem; font-weight: 700;">{report.risk_level}</div>
                <div style="font-size: 0.85rem; color: #888; margin-top: 8px;">Overall Risk Level</div>
            </div>
            """, unsafe_allow_html=True)

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Overall</div><div class="metric-value">{report.overall_score:.1f}</div></div>', unsafe_allow_html=True)
            with mc2:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Membership</div><div class="metric-value">{report.membership_leakage:.1f}</div></div>', unsafe_allow_html=True)
            with mc3:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Attribute</div><div class="metric-value">{report.attribute_leakage:.1f}</div></div>', unsafe_allow_html=True)

        # SECTION 2 — Membership Inference
        st.markdown('<p class="section-header">MEMBERSHIP INFERENCE ATTACK</p>', unsafe_allow_html=True)
        col_roc, col_hist = st.columns(2)

        with col_roc:
            st.plotly_chart(roc_chart(pr.roc_fpr, pr.roc_tpr, pr.auc), use_container_width=True)

        with col_hist:
            st.plotly_chart(
                confidence_histogram(pr.confidence_scores_members, pr.confidence_scores_nonmembers),
                use_container_width=True,
            )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">AUC</div><div class="metric-value">{pr.auc:.3f}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">TPR @ 5% FPR</div><div class="metric-value">{pr.tpr_at_5fpr:.3f}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Members Detected</div><div class="metric-value">{pr.n_detected_members:,}</div></div>', unsafe_allow_html=True)

        # SECTION 3 — Attribute Leakage
        st.markdown('<p class="section-header">ATTRIBUTE LEAKAGE</p>', unsafe_allow_html=True)

        if rr.attributes:
            # Bar charts — side by side
            attr_cols = st.columns(min(3, len(rr.attributes)))
            for i, attr in enumerate(rr.attributes):
                with attr_cols[i % len(attr_cols)]:
                    st.plotly_chart(attribute_bar_chart(attr), use_container_width=True)

            # Sensitivity heatmap
            st.plotly_chart(sensitivity_heatmap(rr.attributes), use_container_width=True)

            # Top leaked attributes callout
            st.markdown("**Top Leaked Attributes**")
            for attr in rr.attributes[:3]:
                kl_color = "#EF4444" if attr.kl_divergence < 0.5 else "#22C55E"
                st.markdown(
                    f'<div class="metric-box" style="display:inline-block; margin:4px; min-width:200px;">'
                    f'<div class="metric-label">{attr.attribute_name}</div>'
                    f'<div class="metric-value" style="color:{kl_color}">KL: {attr.kl_divergence:.3f}</div>'
                    f'<div style="color:#888;font-size:0.75rem">Sensitivity: {attr.sensitivity_score:.3f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No sensitive attributes found in the feature schema.")

        # SECTION 5 — Recommendations
        st.markdown('<p class="section-header">RECOMMENDATIONS</p>', unsafe_allow_html=True)

        for m in mitigations:
            css_class = f"mitigation-{m.priority.lower()}"
            reduction_text = f" (est. -{m.estimated_score_reduction:.0f} pts)" if m.estimated_score_reduction > 0 else ""
            st.markdown(
                f'<div class="{css_class}">'
                f'<strong>[{m.priority}]</strong> {m.technique}{reduction_text}<br>'
                f'<span style="color:#888;font-size:0.85rem">{m.description}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Defense Comparison ──────────────────────────────────────────────────────

    if compare_btn:
        with st.spinner("Running defense comparison (this may take a minute)..."):
            defense_results, defense_budgets = run_defense_comparison(
                model_path, scaler_path, n_queries, conf_thresh, z_thresh,
            )
        st.session_state["defense_results"] = defense_results
        st.session_state["defense_budgets"] = defense_budgets

    if "defense_results" in st.session_state:
        st.markdown('<p class="section-header">DEFENSE COMPARISON</p>', unsafe_allow_html=True)

        defense_results = st.session_state["defense_results"]
        defense_budgets = st.session_state["defense_budgets"]

        # Table
        rows = []
        for name, metrics in defense_results.items():
            rows.append({
                "Defense": name,
                "Leakage Score": f"{max(0, metrics['leakage']):.1f}",
                "Attack AUC": f"{metrics['auc']:.3f}",
                "Accuracy": f"{metrics['accuracy']:.3f}",
            })

        st.dataframe(
            pd.DataFrame(rows).set_index("Defense"),
            use_container_width=True,
        )

        # Query-budget analysis chart
        st.plotly_chart(budget_chart(defense_budgets), use_container_width=True)

        # Before/after summary
        if "None" in defense_results:
            baseline_leak = defense_results["None"]["leakage"]
            best_name = min(
                (k for k in defense_results if k != "None"),
                key=lambda k: defense_results[k]["leakage"],
                default=None,
            )
            if best_name:
                best_leak = defense_results[best_name]["leakage"]
                reduction = baseline_leak - best_leak
                st.markdown(
                    f'<div class="metric-box" style="border-color:#22C55E;">'
                    f'<div class="metric-label">Best Defense: {best_name}</div>'
                    f'<div class="metric-value" style="color:#22C55E;">'
                    f'{max(0,baseline_leak):.1f} → {max(0,best_leak):.1f} '
                    f'<span style="font-size:0.9rem">(↓{max(0,reduction):.1f})</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Empty state ─────────────────────────────────────────────────────────────

    if "report" not in st.session_state and not compare_btn:
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center; padding:60px 0; color:#555;">'
            '<p style="font-size:1.3rem;">Select a model and click <strong>Run Audit</strong> to begin.</p>'
            '<p style="font-size:0.9rem;">The audit will probe the model for membership inference and attribute leakage vulnerabilities.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — LLM Audit
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    import json

    st.markdown('<p class="section-header">LLM TRAINING DATA DETECTION</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#888; font-size:0.95rem;">'
        'Test whether a language model has memorized specific text from its training data using perplexity analysis.'
        '</p>',
        unsafe_allow_html=True,
    )

    gpt2_path = MODEL_DIR / "gpt2"
    training_json = DATA_DIR / "llm_known_training.json"
    novel_json = DATA_DIR / "llm_novel_text.json"

    if not gpt2_path.exists():
        st.warning("GPT-2 model not found at models/gpt2/. Run `python llm/download_gpt2.py` first.")
    elif not training_json.exists() or not novel_json.exists():
        st.warning("Test data not found. Run `python llm/test_data.py` to generate llm_known_training.json and llm_novel_text.json.")
    else:

        if run_llm_btn:
            with st.spinner("Loading GPT-2 and computing perplexities (this may take a minute)..."):
                progress_llm = st.progress(0, text="Loading GPT-2 model...")
                llm_target = load_llm_target(str(gpt2_path))
                progress_llm.progress(20, text="Loading test data...")

                with open(training_json) as f:
                    training_records = json.load(f)
                with open(novel_json) as f:
                    novel_records = json.load(f)

                training_texts = [r["text"] for r in training_records]
                novel_texts = [r["text"] for r in novel_records]

                progress_llm.progress(30, text="Running LLM membership probe...")
                llm_probe = LLMMembershipProbe(llm_target)
                llm_results = llm_probe.probe(training_texts, novel_texts)
                progress_llm.progress(100, text="LLM audit complete.")

            st.session_state["llm_results"] = llm_results
            st.session_state["llm_training_records"] = training_records
            st.session_state["llm_novel_records"] = novel_records

        if "llm_results" in st.session_state:
            llm_res: ProbeResults = st.session_state["llm_results"]
            llm_train_recs = st.session_state["llm_training_records"]
            llm_novel_recs = st.session_state["llm_novel_records"]

            # a) AUC gauge chart
            auc_score = llm_res.auc * 100  # scale to 0-100 for gauge
            if auc_score <= 60:
                auc_risk_color = "green"
            elif auc_score <= 75:
                auc_risk_color = "yellow"
            elif auc_score <= 85:
                auc_risk_color = "orange"
            else:
                auc_risk_color = "red"

            col_llm_gauge, col_llm_metrics = st.columns([1, 1])
            with col_llm_gauge:
                st.plotly_chart(gauge_chart(auc_score, auc_risk_color), use_container_width=True)
            with col_llm_metrics:
                gc = RISK_COLORS.get(auc_risk_color, "#EF4444")
                st.markdown(f"""
                <div class="risk-card" style="border: 2px solid {gc}; color: {gc};">
                    <div style="font-size: 1.8rem; font-weight: 700;">AUC: {llm_res.auc:.3f}</div>
                    <div style="font-size: 0.85rem; color: #888; margin-top: 8px;">Membership Inference AUC</div>
                </div>
                """, unsafe_allow_html=True)

                lm1, lm2, lm3 = st.columns(3)
                with lm1:
                    st.markdown(f'<div class="metric-box"><div class="metric-label">AUC</div><div class="metric-value">{llm_res.auc:.3f}</div></div>', unsafe_allow_html=True)
                with lm2:
                    st.markdown(f'<div class="metric-box"><div class="metric-label">TPR @ 5% FPR</div><div class="metric-value">{llm_res.tpr_at_5fpr:.3f}</div></div>', unsafe_allow_html=True)
                with lm3:
                    st.markdown(f'<div class="metric-box"><div class="metric-label">Detected</div><div class="metric-value">{llm_res.n_detected_members}</div></div>', unsafe_allow_html=True)

            # b) ROC curve
            st.markdown('<p class="section-header">ROC CURVE</p>', unsafe_allow_html=True)
            st.plotly_chart(roc_chart(llm_res.roc_fpr, llm_res.roc_tpr, llm_res.auc), use_container_width=True)

            # c) Perplexity distribution histogram
            st.markdown('<p class="section-header">PERPLEXITY DISTRIBUTION</p>', unsafe_allow_html=True)
            st.plotly_chart(
                perplexity_histogram(
                    llm_res.confidence_scores_members,
                    llm_res.confidence_scores_nonmembers,
                ),
                use_container_width=True,
            )

            # d) Sample results table
            st.markdown('<p class="section-header">SAMPLE RESULTS</p>', unsafe_allow_html=True)
            all_records = llm_train_recs + llm_novel_recs
            table_rows = []
            for i, pq in enumerate(llm_res.per_query_results[:10]):
                rec = all_records[pq["input_idx"]] if pq["input_idx"] < len(all_records) else {}
                text_preview = rec.get("text", "")[:80] + ("..." if len(rec.get("text", "")) > 80 else "")
                source = rec.get("label", "unknown")
                table_rows.append({
                    "Text": text_preview,
                    "Source": source,
                    "Perplexity": f"{pq['confidence']:.1f}",
                    "Predicted Member?": "Yes" if pq["is_predicted_member"] else "No",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            # e) Key finding callout
            mean_train_ppl = np.mean(llm_res.confidence_scores_members) if llm_res.confidence_scores_members else 0
            mean_novel_ppl = np.mean(llm_res.confidence_scores_nonmembers) if llm_res.confidence_scores_nonmembers else 0
            if mean_novel_ppl > 0:
                pct_lower = ((mean_novel_ppl - mean_train_ppl) / mean_novel_ppl) * 100
            else:
                pct_lower = 0
            st.markdown(
                f'<div class="metric-box" style="border-color:#EF4444; margin-top:16px;">'
                f'<div class="metric-label">Key Finding</div>'
                f'<div style="color:#E0E0E0; font-size:0.95rem; margin-top:8px;">'
                f'GPT-2 shows <strong style="color:#EF4444">{pct_lower:.1f}%</strong> lower perplexity on known training data '
                f'(mean {mean_train_ppl:.1f}) vs novel text (mean {mean_novel_ppl:.1f}) '
                f'&mdash; confirming memorization of training corpus.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        else:
            st.markdown("---")
            st.markdown(
                '<div style="text-align:center; padding:60px 0; color:#555;">'
                '<p style="font-size:1.3rem;">Click <strong>Run LLM Audit</strong> in the sidebar to begin.</p>'
                '<p style="font-size:0.9rem;">The audit will test GPT-2 for training data memorization using perplexity analysis.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — Gemini Audit (Live API)
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    import json as json_mod

    st.markdown('<p class="section-header">GEMINI API — LIVE MEMORIZATION TEST</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#888; font-size:0.95rem;">'
        'Test a production Google Gemini model for training data memorization. '
        'We give Gemini the first half of known training texts (Wikipedia, books) vs novel texts, '
        'and measure how accurately it can reproduce the second half. Higher accuracy = memorized.'
        '</p>',
        unsafe_allow_html=True,
    )

    env_path = PROJECT_ROOT / ".env"
    training_json_g = DATA_DIR / "llm_known_training.json"
    novel_json_g = DATA_DIR / "llm_novel_text.json"

    if not env_path.exists():
        st.warning("No .env file found. Create ~/plaf/.env with GEMINI_API_KEY=your-key")
    elif not training_json_g.exists() or not novel_json_g.exists():
        st.warning("Test data not found. Run `python3 llm/test_data.py` first.")
    else:
        if run_gemini_btn:
            with st.spinner("Querying Gemini API (this takes ~1 min per 10 samples)..."):
                progress_g = st.progress(0, text="Connecting to Gemini...")

                gemini_target = GeminiTarget()
                progress_g.progress(5, text="Loading test data...")

                with open(training_json_g) as f:
                    g_train_recs = json_mod.load(f)
                with open(novel_json_g) as f:
                    g_novel_recs = json_mod.load(f)

                # Use subset based on slider
                g_train_texts = [r["text"] for r in g_train_recs[:gemini_n]]
                g_novel_texts = [r["text"] for r in g_novel_recs[:gemini_n]]

                gemini_probe = GeminiMembershipProbe(gemini_target)

                def gemini_progress(frac, msg):
                    progress_g.progress(int(5 + frac * 90), text=msg)

                g_results = gemini_probe.probe(g_train_texts, g_novel_texts, progress_callback=gemini_progress)
                progress_g.progress(100, text="Gemini audit complete.")

            st.session_state["gemini_results"] = g_results
            st.session_state["gemini_train_recs"] = g_train_recs[:gemini_n]
            st.session_state["gemini_novel_recs"] = g_novel_recs[:gemini_n]

        if "gemini_results" in st.session_state:
            g_res: ProbeResults = st.session_state["gemini_results"]
            g_train = st.session_state["gemini_train_recs"]
            g_novel = st.session_state["gemini_novel_recs"]

            # AUC gauge
            g_auc_pct = g_res.auc * 100
            if g_auc_pct <= 60:
                g_color = "green"
            elif g_auc_pct <= 75:
                g_color = "yellow"
            elif g_auc_pct <= 85:
                g_color = "orange"
            else:
                g_color = "red"

            gc1, gc2 = st.columns([1, 1])
            with gc1:
                st.plotly_chart(gauge_chart(g_auc_pct, g_color), use_container_width=True)
            with gc2:
                gc = RISK_COLORS.get(g_color, "#EF4444")
                st.markdown(f"""
                <div class="risk-card" style="border: 2px solid {gc}; color: {gc};">
                    <div style="font-size: 1.8rem; font-weight: 700;">AUC: {g_res.auc:.3f}</div>
                    <div style="font-size: 0.85rem; color: #888; margin-top: 8px;">Gemini Memorization Score</div>
                </div>
                """, unsafe_allow_html=True)

                gm1, gm2, gm3 = st.columns(3)
                with gm1:
                    st.markdown(f'<div class="metric-box"><div class="metric-label">AUC</div><div class="metric-value">{g_res.auc:.3f}</div></div>', unsafe_allow_html=True)
                with gm2:
                    st.markdown(f'<div class="metric-box"><div class="metric-label">TPR @ 5% FPR</div><div class="metric-value">{g_res.tpr_at_5fpr:.3f}</div></div>', unsafe_allow_html=True)
                with gm3:
                    st.markdown(f'<div class="metric-box"><div class="metric-label">Detected</div><div class="metric-value">{g_res.n_detected_members}</div></div>', unsafe_allow_html=True)

            # ROC curve
            st.markdown('<p class="section-header">ROC CURVE</p>', unsafe_allow_html=True)
            st.plotly_chart(roc_chart(g_res.roc_fpr, g_res.roc_tpr, g_res.auc), use_container_width=True)

            # Memorization score distribution
            st.markdown('<p class="section-header">MEMORIZATION SCORE DISTRIBUTION</p>', unsafe_allow_html=True)
            st.plotly_chart(
                perplexity_histogram(g_res.confidence_scores_members, g_res.confidence_scores_nonmembers),
                use_container_width=True,
            )

            # Sample results
            st.markdown('<p class="section-header">SAMPLE RESULTS</p>', unsafe_allow_html=True)
            all_g_recs = g_train + g_novel
            g_rows = []
            for pq in g_res.per_query_results[:10]:
                idx = pq["input_idx"]
                rec = all_g_recs[idx] if idx < len(all_g_recs) else {}
                text_preview = rec.get("text", "")[:80] + ("..." if len(rec.get("text", "")) > 80 else "")
                source = rec.get("label", "unknown")
                g_rows.append({
                    "Text": text_preview,
                    "Source": source,
                    "Score": f"{pq['confidence']:.1f}",
                    "Memorized?": "Yes" if pq["is_predicted_member"] else "No",
                })
            st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True)

            # Key finding
            mean_train_s = np.mean(g_res.confidence_scores_members) if g_res.confidence_scores_members else 0
            mean_novel_s = np.mean(g_res.confidence_scores_nonmembers) if g_res.confidence_scores_nonmembers else 0
            if mean_novel_s > 0:
                pct_diff = ((mean_novel_s - mean_train_s) / mean_novel_s) * 100
            else:
                pct_diff = 0
            st.markdown(
                f'<div class="metric-box" style="border-color:#EF4444; margin-top:16px;">'
                f'<div class="metric-label">Key Finding</div>'
                f'<div style="color:#E0E0E0; font-size:0.95rem; margin-top:8px;">'
                f'Gemini ({g_res.per_query_results[0]["confidence"]:.0f} model) reproduces known training text '
                f'<strong style="color:#EF4444">{abs(pct_diff):.1f}%</strong> more accurately than novel text '
                f'(mean score {mean_train_s:.1f} vs {mean_novel_s:.1f}) '
                f'&mdash; indicating memorization of training corpus.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        else:
            st.markdown("---")
            st.markdown(
                '<div style="text-align:center; padding:60px 0; color:#555;">'
                '<p style="font-size:1.3rem;">Click <strong>Run Gemini Audit</strong> in the sidebar.</p>'
                '<p style="font-size:0.9rem;">Tests Google Gemini for training data memorization via continuation accuracy.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4 — OpenAI Audit (Live API with real logprobs)
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    import json as json_mod2

    st.markdown('<p class="section-header">OPENAI API — LOGPROB PERPLEXITY TEST</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#888; font-size:0.95rem;">'
        'Test OpenAI models using real per-token log-probabilities. '
        'Lower perplexity on known training text vs novel text = memorization detected.'
        '</p>',
        unsafe_allow_html=True,
    )

    training_json_o = DATA_DIR / "llm_known_training.json"
    novel_json_o = DATA_DIR / "llm_novel_text.json"

    if not training_json_o.exists() or not novel_json_o.exists():
        st.warning("Test data not found. Run `python3 llm/test_data.py` first.")
    else:
        if run_openai_btn:
            with st.spinner("Querying OpenAI API..."):
                progress_o = st.progress(0, text="Connecting to OpenAI...")

                openai_target = OpenAITarget()
                progress_o.progress(5, text="Loading test data...")

                with open(training_json_o) as f:
                    o_train_recs = json_mod2.load(f)
                with open(novel_json_o) as f:
                    o_novel_recs = json_mod2.load(f)

                o_train_texts = [r["text"] for r in o_train_recs[:openai_n]]
                o_novel_texts = [r["text"] for r in o_novel_recs[:openai_n]]

                openai_probe = OpenAIMembershipProbe(openai_target)

                def openai_progress(frac, msg):
                    progress_o.progress(int(5 + frac * 90), text=msg)

                o_results = openai_probe.probe(o_train_texts, o_novel_texts, progress_callback=openai_progress)
                progress_o.progress(100, text="OpenAI audit complete.")

            st.session_state["openai_results"] = o_results
            st.session_state["openai_train_recs"] = o_train_recs[:openai_n]
            st.session_state["openai_novel_recs"] = o_novel_recs[:openai_n]

        if "openai_results" in st.session_state:
            o_res: ProbeResults = st.session_state["openai_results"]
            o_train = st.session_state["openai_train_recs"]
            o_novel = st.session_state["openai_novel_recs"]

            o_auc_pct = o_res.auc * 100
            if o_auc_pct <= 60:
                o_color = "green"
            elif o_auc_pct <= 75:
                o_color = "yellow"
            elif o_auc_pct <= 85:
                o_color = "orange"
            else:
                o_color = "red"

            oc1, oc2 = st.columns([1, 1])
            with oc1:
                st.plotly_chart(gauge_chart(o_auc_pct, o_color), use_container_width=True)
            with oc2:
                oc = RISK_COLORS.get(o_color, "#EF4444")
                st.markdown(f"""
                <div class="risk-card" style="border: 2px solid {oc}; color: {oc};">
                    <div style="font-size: 1.8rem; font-weight: 700;">AUC: {o_res.auc:.3f}</div>
                    <div style="font-size: 0.85rem; color: #888; margin-top: 8px;">OpenAI Memorization Score (via logprobs)</div>
                </div>
                """, unsafe_allow_html=True)

                om1, om2, om3 = st.columns(3)
                with om1:
                    st.markdown(f'<div class="metric-box"><div class="metric-label">AUC</div><div class="metric-value">{o_res.auc:.3f}</div></div>', unsafe_allow_html=True)
                with om2:
                    st.markdown(f'<div class="metric-box"><div class="metric-label">TPR @ 5% FPR</div><div class="metric-value">{o_res.tpr_at_5fpr:.3f}</div></div>', unsafe_allow_html=True)
                with om3:
                    st.markdown(f'<div class="metric-box"><div class="metric-label">Detected</div><div class="metric-value">{o_res.n_detected_members}</div></div>', unsafe_allow_html=True)

            st.markdown('<p class="section-header">ROC CURVE</p>', unsafe_allow_html=True)
            st.plotly_chart(roc_chart(o_res.roc_fpr, o_res.roc_tpr, o_res.auc), use_container_width=True)

            st.markdown('<p class="section-header">PERPLEXITY DISTRIBUTION</p>', unsafe_allow_html=True)
            st.plotly_chart(
                perplexity_histogram(o_res.confidence_scores_members, o_res.confidence_scores_nonmembers),
                use_container_width=True,
            )

            st.markdown('<p class="section-header">SAMPLE RESULTS</p>', unsafe_allow_html=True)
            all_o_recs = o_train + o_novel
            o_rows = []
            for pq in o_res.per_query_results[:10]:
                idx = pq["input_idx"]
                rec = all_o_recs[idx] if idx < len(all_o_recs) else {}
                text_preview = rec.get("text", "")[:80] + ("..." if len(rec.get("text", "")) > 80 else "")
                source = rec.get("label", "unknown")
                o_rows.append({
                    "Text": text_preview,
                    "Source": source,
                    "Perplexity": f"{pq['confidence']:.3f}",
                    "Memorized?": "Yes" if pq["is_predicted_member"] else "No",
                })
            st.dataframe(pd.DataFrame(o_rows), use_container_width=True, hide_index=True)

            mean_train_o = np.mean(o_res.confidence_scores_members) if o_res.confidence_scores_members else 0
            mean_novel_o = np.mean(o_res.confidence_scores_nonmembers) if o_res.confidence_scores_nonmembers else 0
            if mean_novel_o > 0:
                pct_diff_o = ((mean_novel_o - mean_train_o) / mean_novel_o) * 100
            else:
                pct_diff_o = 0
            st.markdown(
                f'<div class="metric-box" style="border-color:#EF4444; margin-top:16px;">'
                f'<div class="metric-label">Key Finding</div>'
                f'<div style="color:#E0E0E0; font-size:0.95rem; margin-top:8px;">'
                f'OpenAI (gpt-4o-mini) shows <strong style="color:#EF4444">{abs(pct_diff_o):.1f}%</strong> lower perplexity on known training data '
                f'(mean {mean_train_o:.3f}) vs novel text (mean {mean_novel_o:.3f}) '
                f'&mdash; real logprob-based memorization detection.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        else:
            st.markdown("---")
            st.markdown(
                '<div style="text-align:center; padding:60px 0; color:#555;">'
                '<p style="font-size:1.3rem;">Click <strong>Run OpenAI Audit</strong> in the sidebar.</p>'
                '<p style="font-size:0.9rem;">Tests OpenAI for training data memorization using real per-token log-probabilities.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── Active Extraction Attack ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">ACTIVE EXTRACTION ATTACK</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#888; font-size:0.95rem;">'
        'Actively attempt to extract memorized training data using prompt manipulation strategies: '
        'direct recall, prefix completion, persona manipulation, and template extraction. '
        'Measures refusal rate, specificity, and cross-query consistency.'
        '</p>',
        unsafe_allow_html=True,
    )

    if run_extraction_btn:
        with st.spinner("Running extraction attacks against OpenAI (20 prompts x 2 queries each)..."):
            progress_ex = st.progress(0, text="Starting extraction...")

            from openai import OpenAI as OAI
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
            oai_client = OAI(api_key=os.getenv("OPENAI_API_KEY"))

            def query_openai(prompt: str) -> str:
                resp = oai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ""

            extraction = ExtractionProbe(query_openai)

            def ex_progress(frac, msg):
                progress_ex.progress(int(frac * 100), text=msg)

            ex_report = extraction.run(progress_callback=ex_progress)
            progress_ex.progress(100, text="Extraction complete.")

        st.session_state["extraction_report"] = ex_report

    if "extraction_report" in st.session_state:
        ex = st.session_state["extraction_report"]

        # Risk color
        ex_colors = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "orange", "CRITICAL": "red"}
        ex_color = ex_colors.get(ex.risk_level, "red")
        ec = RISK_COLORS.get(ex_color, "#EF4444")

        exc1, exc2, exc3, exc4 = st.columns(4)
        with exc1:
            st.markdown(
                f'<div class="risk-card" style="border: 2px solid {ec}; color: {ec};">'
                f'<div style="font-size: 1.4rem; font-weight: 700;">{ex.risk_level}</div>'
                f'<div style="color:#888; font-size:0.75rem;">Extraction Risk</div></div>',
                unsafe_allow_html=True)
        with exc2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Leak Rate</div>'
                f'<div class="metric-value">{ex.overall_leak_rate:.0%}</div></div>',
                unsafe_allow_html=True)
        with exc3:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Specificity</div>'
                f'<div class="metric-value">{ex.overall_specificity:.2f}</div></div>',
                unsafe_allow_html=True)
        with exc4:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Consistency</div>'
                f'<div class="metric-value">{ex.overall_consistency:.2f}</div></div>',
                unsafe_allow_html=True)

        # Strategy breakdown
        st.markdown("**Strategy Breakdown**")
        strategy_names = list(set(r.strategy for r in ex.results))
        for strat in strategy_names:
            strat_results = [r for r in ex.results if r.strategy == strat]
            refused = sum(1 for r in strat_results if r.refused)
            leaked = sum(1 for r in strat_results if r.risk_level in ("HIGH", "CRITICAL"))
            avg_spec = np.mean([r.specificity_score for r in strat_results if not r.refused]) if any(not r.refused for r in strat_results) else 0

            if leaked > 0:
                strat_css = "mitigation-critical"
            elif refused < len(strat_results):
                strat_css = "mitigation-high"
            else:
                strat_css = "mitigation-moderate"

            st.markdown(
                f'<div class="{strat_css}">'
                f'<strong>{strat}</strong> — '
                f'{len(strat_results) - refused}/{len(strat_results)} responded, '
                f'{leaked} high-risk, avg specificity {avg_spec:.2f}'
                f'</div>',
                unsafe_allow_html=True)

        # Sample extractions
        st.markdown("**Sample Extractions**")
        ex_rows = []
        for r in ex.results:
            ex_rows.append({
                "Strategy": r.strategy,
                "Prompt": r.prompt[:60] + "...",
                "Refused": "Yes" if r.refused else "No",
                "Specificity": f"{r.specificity_score:.2f}",
                "Consistency": f"{r.consistency_score:.2f}",
                "Risk": r.risk_level,
                "Response Preview": r.response[:80] + "..." if len(r.response) > 80 else r.response,
            })
        st.dataframe(pd.DataFrame(ex_rows), use_container_width=True, hide_index=True)
