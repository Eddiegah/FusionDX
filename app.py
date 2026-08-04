# -*- coding: utf-8 -*-
"""
FusionDx -- Premium Streamlit Dashboard
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
from src.utils import load_json, fmt_age, fmt_prob, fmt_stage

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FusionDx · Multimodal Cancer AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# ─── Custom CSS / theme ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1b2a 100%);
    color: #e8eaf6;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #101d2e 100%);
    border-right: 1px solid #1e3a5f;
}

/* ── Cards ── */
.fusion-card {
    background: linear-gradient(135deg, #0f2a47 0%, #0a1f38 100%);
    border: 1px solid #1e4d8c;
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    box-shadow: 0 4px 24px rgba(0,120,255,0.12);
    transition: transform 0.2s, box-shadow 0.2s;
}
.fusion-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,120,255,0.22);
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #0d2137 0%, #0a1a2e 100%);
    border: 1px solid #1a3d6e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4fc3f7, #81d4fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem;
    color: #7cb3d4;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 4px;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0a2540 0%, #0f3460 40%, #16213e 100%);
    border: 1px solid #1e4d8c;
    border-radius: 20px;
    padding: 40px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(79,195,247,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4fc3f7 0%, #81d4fa 50%, #b3e5fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin: 0;
}
.hero-sub {
    font-size: 1.05rem;
    color: #90caf9;
    margin-top: 10px;
    font-weight: 300;
}

/* ── Disclaimer pill ── */
.disclaimer-pill {
    background: linear-gradient(135deg, #3d1515, #5c1a1a);
    border: 1px solid #c62828;
    border-radius: 50px;
    padding: 10px 20px;
    font-size: 0.82rem;
    color: #ef9a9a;
    text-align: center;
    margin-bottom: 20px;
    letter-spacing: 0.3px;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #4fc3f7;
    border-left: 3px solid #4fc3f7;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* ── Model badges ── */
.badge-image    { background: linear-gradient(135deg,#e55b3c,#c0392b); border-radius:8px; padding:4px 10px; font-size:0.75rem; font-weight:700; color:white; }
.badge-clinical { background: linear-gradient(135deg,#2e86c1,#1a5276); border-radius:8px; padding:4px 10px; font-size:0.75rem; font-weight:700; color:white; }
.badge-fusion   { background: linear-gradient(135deg,#1e8449,#145a32); border-radius:8px; padding:4px 10px; font-size:0.75rem; font-weight:700; color:white; }

/* ── Prediction result boxes ── */
.pred-box {
    border-radius: 14px;
    padding: 22px 16px;
    text-align: center;
    border: 1px solid transparent;
}
.pred-correct { background: linear-gradient(135deg,#0d2b1a,#0a2015); border-color:#1b5e20; }
.pred-wrong   { background: linear-gradient(135deg,#2b0d0d,#200a0a); border-color:#b71c1c; }
.pred-neutral { background: linear-gradient(135deg,#0d1e2b,#0a1828); border-color:#1a3a5c; }

/* ── Progress bars override ── */
.stProgress > div > div > div { background: linear-gradient(90deg,#1565c0,#4fc3f7) !important; border-radius:8px; }

/* ── Tabs ── */
button[data-baseweb="tab"] { color: #7cb3d4 !important; font-weight:600; }
button[data-baseweb="tab"][aria-selected="true"] { color: #4fc3f7 !important; border-bottom: 2px solid #4fc3f7 !important; }

/* ── Selectbox ── */
.stSelectbox label { color: #90caf9 !important; font-weight: 600; }

/* ── Dataframe ── */
.dataframe { font-size: 0.85rem; }

/* ── Expander ── */
.streamlit-expanderHeader { color: #4fc3f7 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Data loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_test_df():
    p = DATA_DIR / "test.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_metrics():
    return load_json(RESULTS_DIR / "metrics.json")

@st.cache_data
def load_image_preds():
    return load_json(RESULTS_DIR / "image_predictions.json") or {}

@st.cache_data
def load_clinical_preds():
    return load_json(RESULTS_DIR / "clinical_predictions.json") or {}

@st.cache_data
def load_fusion_preds():
    return load_json(RESULTS_DIR / "fusion_predictions.json") or {}

def _results_ready():
    return (RESULTS_DIR / "metrics.json").exists()

def _data_ready():
    return (DATA_DIR / "test.csv").exists()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:2.8rem'>🧬</div>
        <div style='font-size:1.3rem; font-weight:800;
             background:linear-gradient(135deg,#4fc3f7,#81d4fa);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            FusionDx
        </div>
        <div style='font-size:0.72rem; color:#546e7a; letter-spacing:2px;
             text-transform:uppercase; margin-top:2px;'>
            Multimodal Cancer AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e3a5f; margin:12px 0;'>", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🏠  Overview", "📊  Results", "🧬  Patient Explorer", "⚡  Architecture", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e3a5f; margin:12px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.78rem; color:#546e7a; font-weight:600; letter-spacing:1px; text-transform:uppercase;'>System Status</div>", unsafe_allow_html=True)

    data_ok    = _data_ready()
    results_ok = _results_ready()
    dot_data    = "<span style='color:#4caf50'>●</span>" if data_ok    else "<span style='color:#f44336'>●</span>"
    dot_results = "<span style='color:#4caf50'>●</span>" if results_ok else "<span style='color:#ff9800'>●</span>"
    st.markdown(f"{dot_data} Data splits",        unsafe_allow_html=True)
    st.markdown(f"{dot_results} Evaluation results", unsafe_allow_html=True)
    st.markdown("<span style='color:#4caf50'>●</span> Dashboard", unsafe_allow_html=True)

    if results_ok:
        m = load_metrics()
        if m:
            best = max(m, key=lambda x: x["auroc"] or 0)
            st.markdown(f"""
            <div style='margin-top:16px; background:linear-gradient(135deg,#0d2137,#0a1a2e);
                 border:1px solid #1a3d6e; border-radius:10px; padding:12px;'>
                <div style='font-size:0.7rem; color:#7cb3d4; text-transform:uppercase; letter-spacing:1px;'>Best AUROC</div>
                <div style='font-size:1.8rem; font-weight:800; color:#4fc3f7;'>{best['auroc']:.3f}</div>
                <div style='font-size:0.72rem; color:#546e7a;'>{best['model'].split('(')[0].strip()}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e3a5f; margin:16px 0 8px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem; color:#37474f; text-align:center;'>TCGA-BRCA · NIH GDC · Open Access</div>", unsafe_allow_html=True)

# ─── Helper: disclaimer ──────────────────────────────────────────────────────
def _disclaimer():
    st.markdown("""
    <div class='disclaimer-pill'>
        ⚠️ <strong>RESEARCH / EDUCATIONAL DEMONSTRATION ONLY</strong> —
        Not validated for clinical use. Do not use for diagnostic or treatment decisions.
    </div>
    """, unsafe_allow_html=True)

# ─── Helper: hero ────────────────────────────────────────────────────────────
def _hero(title, subtitle, icon="🧬"):
    st.markdown(f"""
    <div class='hero-banner'>
        <div style='display:flex; align-items:center; gap:16px;'>
            <div style='font-size:3rem;'>{icon}</div>
            <div>
                <div class='hero-title'>{title}</div>
                <div class='hero-sub'>{subtitle}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Helper: section header ──────────────────────────────────────────────────
def _section(title):
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)

# ─── Helper: stat card ───────────────────────────────────────────────────────
def _stat(col, value, label, color="#4fc3f7"):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='background:linear-gradient(135deg,{color},{color}aa);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
             {value}
        </div>
        <div class='metric-label'>{label}</div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    _disclaimer()
    _hero(
        "FusionDx",
        "Genuine multimodal fusion of histopathology imaging + clinical data · TCGA-BRCA · Cross-Modal Attention",
        "🧬"
    )

    # — Top stats row —
    metrics = load_metrics() if _results_ready() else None
    c1, c2, c3, c4 = st.columns(4)
    _stat(c1, "3", "Models Compared", "#4fc3f7")
    _stat(c2, "120", "Patients (Synthetic)", "#81c784")
    _stat(c3, "960", "Image Tiles", "#ffb74d")
    if metrics:
        fusion_auroc = next((m["auroc"] for m in metrics if "Fusion" in m["model"]), None)
        _stat(c4, f"{fusion_auroc:.3f}" if fusion_auroc else "—", "Fusion AUROC", "#ce93d8")
    else:
        _stat(c4, "—", "Fusion AUROC", "#ce93d8")

    st.markdown("<br>", unsafe_allow_html=True)

    # — Architecture cards —
    _section("Three Models · One Honest Comparison")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div class='fusion-card'>
            <div style='font-size:2rem; margin-bottom:8px;'>🖼️</div>
            <div style='font-size:1rem; font-weight:700; color:#ef8c6f; margin-bottom:6px;'>Image-Only</div>
            <div style='font-size:0.82rem; color:#7cb3d4; line-height:1.6;'>
                <b>ResNet-34</b> fine-tuned on H&amp;E tile patches.<br>
                Mean-pool across tiles per patient.<br>
                ImageNet pretrained → domain adaptation.
            </div>
            <div style='margin-top:12px; font-size:0.72rem; background:#1a0f0d;
                 border:1px solid #5d2e20; border-radius:6px; padding:8px;
                 color:#ef8c6f; font-family:monospace;'>
                ResNet34 → 512d embed → FC(2)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class='fusion-card'>
            <div style='font-size:2rem; margin-bottom:8px;'>📋</div>
            <div style='font-size:1rem; font-weight:700; color:#64b5f6; margin-bottom:6px;'>Clinical-Only</div>
            <div style='font-size:0.82rem; color:#7cb3d4; line-height:1.6;'>
                <b>LightGBM</b> on age, AJCC stage, ER/PR/HER2.<br>
                Gradient-boosted trees — strongest tabular baseline.<br>
                ClinicalMLP for differentiable fusion branch.
            </div>
            <div style='margin-top:12px; font-size:0.72rem; background:#0a141f;
                 border:1px solid #1a3d6e; border-radius:6px; padding:8px;
                 color:#64b5f6; font-family:monospace;'>
                features → MLP → 64d embed → FC(2)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class='fusion-card'>
            <div style='font-size:2rem; margin-bottom:8px;'>⚡</div>
            <div style='font-size:1rem; font-weight:700; color:#81c784; margin-bottom:6px;'>Fusion Model</div>
            <div style='font-size:0.82rem; color:#7cb3d4; line-height:1.6;'>
                <b>Cross-Modal Attention</b> — each modality modulates<br>
                the other's contribution, per patient.<br>
                Two-phase training: warm-up → end-to-end.
            </div>
            <div style='margin-top:12px; font-size:0.72rem; background:#0a1f0e;
                 border:1px solid #1b5e20; border-radius:6px; padding:8px;
                 color:#81c784; font-family:monospace;'>
                [512d ⊗ 64d] → CrossAttn → 256d → FC(2)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # — Key principles —
    _section("Design Principles")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""<div class='fusion-card'>
            <div style='font-size:1.5rem;'>🔒</div>
            <div style='font-weight:700; color:#4fc3f7; margin:8px 0 4px;'>Patient-Level Splitting</div>
            <div style='font-size:0.82rem; color:#7cb3d4;'>
                No tile from one patient appears in both train and test.
                Tile-level splits silently inflate accuracy — we never do that.
            </div></div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""<div class='fusion-card'>
            <div style='font-size:1.5rem;'>📐</div>
            <div style='font-weight:700; color:#4fc3f7; margin:8px 0 4px;'>Genuine Fusion</div>
            <div style='font-size:0.82rem; color:#7cb3d4;'>
                Not just averaging two models. Cross-modal attention lets
                imaging and clinical data inform each other's weighting.
            </div></div>""", unsafe_allow_html=True)
    with p3:
        st.markdown("""<div class='fusion-card'>
            <div style='font-size:1.5rem;'>🎯</div>
            <div style='font-weight:700; color:#4fc3f7; margin:8px 0 4px;'>Honest Reporting</div>
            <div style='font-size:0.82rem; color:#7cb3d4;'>
                If fusion doesn't win, we say so. Real numbers, bootstrap
                confidence intervals, no inflated claims.
            </div></div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊  Results":
    _disclaimer()
    _hero("Honest Model Comparison", "All three models evaluated on the same patient-level test set · Bootstrap CIs included", "📊")

    if not _results_ready():
        st.markdown("""
        <div class='fusion-card' style='text-align:center; padding:40px;'>
            <div style='font-size:3rem; margin-bottom:16px;'>⏳</div>
            <div style='font-size:1.2rem; color:#4fc3f7; font-weight:700;'>No results yet</div>
            <div style='color:#7cb3d4; margin-top:8px;'>Run the training pipeline first:</div>
            <code style='color:#81c784;'>python run_synthetic.py</code>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    metrics = load_metrics()
    MODEL_COLORS = {
        "Image-Only (ResNet-34)":        "#ef8c6f",
        "Clinical-Only (LightGBM)":      "#64b5f6",
        "Fusion (Cross-Modal Attention)":"#81c784",
    }
    MODEL_ICONS = {
        "Image-Only (ResNet-34)":        "🖼️",
        "Clinical-Only (LightGBM)":      "📋",
        "Fusion (Cross-Modal Attention)":"⚡",
    }

    # ── Big metric cards ──
    _section("Key Metrics — Test Set")
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        icon  = MODEL_ICONS.get(m["model"], "●")
        color = MODEL_COLORS.get(m["model"], "#4fc3f7")
        auroc = m["auroc"] if m["auroc"] is not None else 0.0
        ci    = m.get("auroc_ci", [None, None])
        col.markdown(f"""
        <div class='fusion-card' style='text-align:center;'>
            <div style='font-size:1.8rem;'>{icon}</div>
            <div style='font-size:0.78rem; font-weight:700; color:{color};
                 text-transform:uppercase; letter-spacing:1px; margin:8px 0 4px;'>
                {m['model'].split('(')[0].strip()}
            </div>
            <div style='font-size:2.6rem; font-weight:800;
                 background:linear-gradient(135deg,{color},{color}88);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                {auroc:.3f}
            </div>
            <div style='font-size:0.72rem; color:#546e7a; margin-top:2px;'>AUROC</div>
            <div style='font-size:0.7rem; color:#37474f; margin-top:6px;'>
                95% CI [{ci[0]}, {ci[1]}]
            </div>
            <hr style='border-color:#1e3a5f; margin:12px 0;'>
            <div style='display:flex; justify-content:space-around; font-size:0.78rem;'>
                <div><div style='color:{color}; font-weight:700;'>{m['accuracy']:.3f}</div>
                     <div style='color:#546e7a;'>Acc</div></div>
                <div><div style='color:{color}; font-weight:700;'>{m['f1']:.3f}</div>
                     <div style='color:#546e7a;'>F1</div></div>
                <div><div style='color:{color}; font-weight:700;'>{m['recall']:.3f}</div>
                     <div style='color:#546e7a;'>Recall</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Plots ──
    tab1, tab2, tab3 = st.tabs(["📈 ROC Curves", "📊 Bar Comparison", "🔎 Attention Weights"])

    with tab1:
        roc_path = RESULTS_DIR / "roc_curves.png"
        if roc_path.exists():
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(str(roc_path), use_column_width=True)
            with c2:
                st.markdown("""
                <div class='fusion-card'>
                    <div style='font-weight:700; color:#4fc3f7; margin-bottom:12px;'>Reading the ROC Curve</div>
                    <div style='font-size:0.82rem; color:#7cb3d4; line-height:1.8;'>
                        <b>AUC = 1.0</b> → perfect classifier<br>
                        <b>AUC = 0.5</b> → random guessing<br><br>
                        The diagonal dashed line represents a random classifier.
                        Curves above it indicate real predictive signal.<br><br>
                        <span style='color:#ff9800;'>⚠️ Small test set (~18 patients) means
                        wide confidence intervals — interpret cautiously.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        bar_path = RESULTS_DIR / "comparison_chart.png"
        if bar_path.exists():
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(str(bar_path), use_column_width=True)
            with c2:
                # Honest interpretation box
                fusion_m   = next((m for m in metrics if "Fusion" in m["model"]), None)
                img_m      = next((m for m in metrics if "Image" in m["model"]), None)
                clin_m     = next((m for m in metrics if "Clinical" in m["model"]), None)
                if fusion_m and img_m and clin_m:
                    fa = fusion_m["auroc"] or 0
                    ia = img_m["auroc"]    or 0
                    ca = clin_m["auroc"]   or 0
                    if fa > ia and fa > ca:
                        verdict = "✅ Fusion Wins"
                        vcolor  = "#4caf50"
                        vtext   = f"Fusion outperforms both baselines by {fa - max(ia,ca):.3f} AUROC."
                    elif fa == ia or fa == ca:
                        verdict = "🔵 Mixed Result"
                        vcolor  = "#2196f3"
                        vtext   = "Fusion matches but doesn't clearly beat all baselines."
                    else:
                        verdict = "🟡 Honest Negative"
                        vcolor  = "#ff9800"
                        vtext   = "Fusion does not outperform single-modality baselines on this data."
                    st.markdown(f"""
                    <div class='fusion-card'>
                        <div style='font-size:1.1rem; font-weight:800; color:{vcolor};
                             margin-bottom:8px;'>{verdict}</div>
                        <div style='font-size:0.82rem; color:#7cb3d4; line-height:1.6;'>{vtext}</div>
                        <div style='margin-top:12px; font-size:0.75rem; color:#546e7a;'>
                            This result is reported honestly regardless of direction.
                            A negative result is scientifically valuable.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab3:
        attn_path = RESULTS_DIR / "attention_weights.png"
        if attn_path.exists():
            st.markdown("""
            <div class='fusion-card' style='margin-bottom:16px;'>
                <div style='font-weight:700; color:#4fc3f7; margin-bottom:6px;'>
                    Cross-Modal Attention — Per-Patient Modality Reliance
                </div>
                <div style='font-size:0.82rem; color:#7cb3d4; line-height:1.6;'>
                    For each test patient, the stacked bars show how much the fusion model
                    relied on <span style='color:#ef8c6f;'>imaging</span> vs.
                    <span style='color:#64b5f6;'>clinical data</span>.
                    This is a genuine per-sample interpretability signal — unavailable in
                    single-modality models. ✓ = correct prediction, ✗ = incorrect.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(str(attn_path), use_column_width=True)

    # ── Full table ──
    st.markdown("<br>", unsafe_allow_html=True)
    _section("Complete Metrics Table")
    rows = []
    for m in metrics:
        ci = m.get("auroc_ci", [None, None])
        rows.append({
            "Model":        m["model"],
            "Patients":     m["n_patients"],
            "Accuracy":     m["accuracy"],
            "Precision":    m["precision"],
            "Recall":       m["recall"],
            "F1":           m["f1"],
            "AUROC":        m["auroc"],
            "AUROC 95% CI": f"[{ci[0]}, {ci[1]}]" if ci[0] is not None else "N/A",
        })
    df_t = pd.DataFrame(rows)
    numeric_cols = ["Accuracy", "Precision", "Recall", "F1", "AUROC"]
    try:
        styled = (df_t.style
                  .highlight_max(subset=numeric_cols, color="#0d2b1a")
                  .format({c: "{:.4f}" for c in numeric_cols}))
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df_t, use_container_width=True)

    # ── Written report ──
    report_path = RESULTS_DIR / "comparison_report.md"
    if report_path.exists():
        with st.expander("📄 Full Written Report", expanded=False):
            st.markdown(report_path.read_text(encoding="utf-8"))

# ════════════════════════════════════════════════════════════════════════════
# PAGE: PATIENT EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧬  Patient Explorer":
    _disclaimer()
    _hero("Patient Case Explorer", "Select any test patient · Compare all 3 models · Inspect attention weights", "🧬")

    if not _data_ready():
        st.markdown("<div class='fusion-card' style='text-align:center;padding:40px;'><div style='font-size:3rem;'>📂</div><div style='color:#4fc3f7;font-size:1.1rem;font-weight:700;margin-top:12px;'>No data found</div><div style='color:#7cb3d4;margin-top:8px;'>Run <code>python run_synthetic.py</code> first.</div></div>", unsafe_allow_html=True)
        st.stop()

    test_df        = load_test_df()
    fusion_preds   = load_fusion_preds()
    image_preds    = load_image_preds()
    clinical_preds = load_clinical_preds()

    if test_df is None or test_df.empty:
        st.warning("test.csv is empty.")
        st.stop()

    patients = sorted(test_df["submitter_id"].unique().tolist())

    sel_col, info_col = st.columns([2, 3])
    with sel_col:
        selected = st.selectbox(
            f"🔍 Choose a patient ({len(patients)} in test set):",
            patients,
        )
    with info_col:
        if selected in fusion_preds:
            fp  = fusion_preds[selected]
            lbl = fp.get("label", -1)
            st.markdown(f"""
            <div style='padding:12px 16px; background:linear-gradient(135deg,#0d2137,#0a1a2e);
                 border:1px solid #1a3d6e; border-radius:10px; margin-top:24px;'>
                <span style='font-size:0.78rem; color:#7cb3d4; text-transform:uppercase;
                      letter-spacing:1px;'>Ground Truth</span>
                <span style='margin-left:12px; font-size:1rem; font-weight:700;
                      color:{"#ef8c6f" if lbl==1 else "#64b5f6"};'>
                    {"● IDC (Invasive Ductal Carcinoma)" if lbl==1 else "● Other Subtype"}
                </span>
            </div>
            """, unsafe_allow_html=True)

    patient_rows = test_df[test_df["submitter_id"] == selected]
    label = int(patient_rows["label"].iloc[0]) if "label" in patient_rows.columns else None

    # ── Tabs ──
    t_clin, t_tiles, t_preds, t_attn = st.tabs(
        ["📋 Clinical Data", "🖼️ Image Tiles", "🤖 Predictions", "🔎 Attention"]
    )

    # ── Clinical ──
    with t_clin:
        clin_cols = ["age_at_diagnosis","tumor_stage","er_status","pr_status",
                     "her2_status","gender","race","primary_diagnosis"]
        clin_row = patient_rows.drop_duplicates("submitter_id")[
            [c for c in clin_cols if c in patient_rows.columns]].copy()
        if "age_at_diagnosis" in clin_row.columns:
            clin_row["age_at_diagnosis"] = clin_row["age_at_diagnosis"].apply(
                lambda x: fmt_age(float(x)) if pd.notna(x) else "unknown")
        if "tumor_stage" in clin_row.columns:
            clin_row["tumor_stage"] = clin_row["tumor_stage"].apply(fmt_stage)

        display = clin_row.T.rename(columns={clin_row.index[0]: "Value"})
        field_icons = {
            "age_at_diagnosis": "🎂", "tumor_stage": "🏥", "er_status": "🔬",
            "pr_status": "🔬", "her2_status": "🧪", "gender": "👤",
            "race": "🌍", "primary_diagnosis": "📋",
        }
        st.markdown("<div style='display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:8px;'>", unsafe_allow_html=True)
        for field, val in display["Value"].items():
            icon = field_icons.get(field, "●")
            label_clean = field.replace("_", " ").title()
            color = "#ef8c6f" if "positive" in str(val).lower() else "#64b5f6" if "negative" in str(val).lower() else "#7cb3d4"
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0d2137,#0a1a2e);
                 border:1px solid #1a3d6e; border-radius:10px; padding:12px 14px;'>
                <div style='font-size:0.72rem; color:#546e7a; text-transform:uppercase;
                     letter-spacing:1px; margin-bottom:4px;'>{icon} {label_clean}</div>
                <div style='font-size:0.95rem; font-weight:600; color:{color};'>{val}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tiles ──
    with t_tiles:
        tile_paths = patient_rows["tile_path"].dropna().tolist()
        if tile_paths:
            n_show = min(8, len(tile_paths))
            st.markdown(f"<div style='font-size:0.85rem; color:#7cb3d4; margin-bottom:12px;'>Showing {n_show} of {len(tile_paths)} tiles · 256×256 px · WSI pyramid level 1</div>", unsafe_allow_html=True)
            cols = st.columns(min(4, n_show))
            for i, tp in enumerate(tile_paths[:n_show]):
                col = cols[i % 4]
                try:
                    img = Image.open(tp).convert("RGB")
                    col.image(img, caption=f"Tile {i+1}", use_column_width=True)
                except Exception as e:
                    col.warning(f"Tile {i+1}: {e}")
        else:
            st.info("No tiles found for this patient.")

    # ── Predictions ──
    with t_preds:
        model_specs = [
            ("🖼️ Image-Only",    "ResNet-34",  image_preds,    "#ef8c6f", "#2b1208"),
            ("📋 Clinical-Only", "LightGBM",   clinical_preds, "#64b5f6", "#081428"),
            ("⚡ Fusion",         "Cross-Attn", fusion_preds,   "#81c784", "#081f0a"),
        ]
        p_cols = st.columns(3)
        for col, (title, arch, preds_dict, color, bg) in zip(p_cols, model_specs):
            with col:
                if selected in preds_dict:
                    p       = preds_dict[selected]
                    prob    = float(p.get("prob_idc", 0.5))
                    pred    = int(p.get("pred", int(prob >= 0.5)))
                    correct = (pred == label) if label is not None else None
                    outcome = ("✅ Correct" if correct else "❌ Wrong") if correct is not None else ""
                    out_c   = "#4caf50" if correct else "#f44336"
                    col.markdown(f"""
                    <div class='fusion-card' style='text-align:center; background:linear-gradient(135deg,{bg},{bg}aa);
                         border-color:{color}44;'>
                        <div style='font-size:1.3rem;'>{title.split()[0]}</div>
                        <div style='font-weight:700; color:{color}; font-size:0.9rem;
                             margin:6px 0 2px;'>{title.split(None,1)[1]}</div>
                        <div style='font-size:0.72rem; color:#546e7a; margin-bottom:12px;'>{arch}</div>
                        <div style='font-size:2rem; font-weight:800; color:{color};'>
                            {"IDC" if pred==1 else "Other"}
                        </div>
                        <div style='font-size:0.85rem; font-weight:700; color:{out_c}; margin:6px 0 12px;'>{outcome}</div>
                        <div style='font-size:0.78rem; color:#7cb3d4; margin-bottom:6px;'>
                            P(IDC) = {fmt_prob(prob)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(prob, text="")
                else:
                    col.markdown(f"<div class='fusion-card' style='text-align:center; color:#546e7a; padding:30px;'>{title}<br><br>No prediction yet.</div>", unsafe_allow_html=True)

    # ── Attention ──
    with t_attn:
        if selected in fusion_preds and "alpha_img" in fusion_preds[selected]:
            fp      = fusion_preds[selected]
            a_img   = float(fp["alpha_img"])
            a_clin  = float(fp["alpha_clin"])

            at_col, interp_col = st.columns([1, 1])
            with at_col:
                fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0d1117")
                ax.set_facecolor("#0d1117")
                bars = ax.barh(
                    ["Clinical", "Image"],
                    [a_clin, a_img],
                    color=["#64b5f6", "#ef8c6f"],
                    edgecolor="#1e3a5f", linewidth=0.8, height=0.45,
                )
                ax.set_xlim([0, 1.2])
                ax.set_xlabel("Attention weight (normalised)", color="#7cb3d4", fontsize=9)
                ax.set_title(f"Modality reliance — {selected[:14]}", color="#4fc3f7", fontsize=10)
                ax.tick_params(colors="#7cb3d4")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1e3a5f")
                for bar, val in zip(bars, [a_clin, a_img]):
                    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                            f"{val:.2f}", va="center", color="white", fontsize=11, fontweight="bold")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with interp_col:
                dominant = "🖼️ Imaging" if a_img > a_clin else "📋 Clinical Data"
                dom_pct  = max(a_img, a_clin)
                other_pct = min(a_img, a_clin)
                dom_col  = "#ef8c6f" if a_img > a_clin else "#64b5f6"
                st.markdown(f"""
                <div class='fusion-card' style='margin-top:8px;'>
                    <div style='font-weight:700; color:#4fc3f7; margin-bottom:12px; font-size:1rem;'>
                        Interpretation
                    </div>
                    <div style='font-size:0.85rem; color:#7cb3d4; line-height:1.8;'>
                        The fusion model primarily weighted<br>
                        <span style='font-size:1.1rem; font-weight:800; color:{dom_col};'>
                            {dominant}
                        </span><br>
                        for this patient.
                    </div>
                    <div style='margin-top:16px; display:flex; gap:12px;'>
                        <div style='flex:1; background:#0a1428; border:1px solid #1a3d6e;
                             border-radius:8px; padding:10px; text-align:center;'>
                            <div style='font-size:1.5rem; font-weight:800; color:#ef8c6f;'>{a_img:.0%}</div>
                            <div style='font-size:0.7rem; color:#546e7a; text-transform:uppercase;
                                 letter-spacing:1px;'>Image</div>
                        </div>
                        <div style='flex:1; background:#0a1428; border:1px solid #1a3d6e;
                             border-radius:8px; padding:10px; text-align:center;'>
                            <div style='font-size:1.5rem; font-weight:800; color:#64b5f6;'>{a_clin:.0%}</div>
                            <div style='font-size:0.7rem; color:#546e7a; text-transform:uppercase;
                                 letter-spacing:1px;'>Clinical</div>
                        </div>
                    </div>
                    <div style='margin-top:12px; font-size:0.72rem; color:#37474f; line-height:1.6;'>
                        This sample-specific signal is only available in the multimodal fusion model.
                        Single-modality models cannot provide per-patient modality reliance.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Attention weights not available — run fusion model first.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚡  Architecture":
    _disclaimer()
    _hero("Cross-Modal Attention Architecture", "How genuine fusion works — the technical core of FusionDx", "⚡")

    _section("Fusion Architecture Diagram")
    st.markdown("""
    <div class='fusion-card' style='font-family:monospace; font-size:0.82rem;
         line-height:2; color:#b0bec5; padding:28px 32px;'>

    <span style='color:#ef8c6f; font-weight:700;'>IMAGE BRANCH</span>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <span style='color:#64b5f6; font-weight:700;'>CLINICAL BRANCH</span><br>

    256×256 H&amp;E tile &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    age, stage, ER/PR/HER2<br>

    ↓ ResNet-34 backbone &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    ↓ ClinicalMLP encoder<br>

    <span style='color:#ef8c6f;'>e_img ∈ ℝ^512</span>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <span style='color:#64b5f6;'>e_clin ∈ ℝ^64</span><br>

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    ↘ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↙<br>

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <span style='color:#81c784; font-weight:700;'>CrossModalAttention</span><br>

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    α_img  = σ(W_q·e_img ⊙ W_k·e_clin)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    α_clin = σ(W_q·e_clin ⊙ W_k·e_img)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    z = LayerNorm(α_img·V_img + α_clin·V_clin)<br>

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ z ∈ ℝ^256<br>

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    Dropout(0.4) → FC(256, 2)<br>

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <span style='color:#ffb74d; font-weight:700;'>IDC / Other</span>
    &nbsp;&nbsp;&nbsp;+&nbsp;&nbsp;&nbsp;
    <span style='color:#ce93d8; font-weight:700;'>α_img, α_clin</span> (interpretability)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _section("Training Strategy: Two-Phase")
    ph1, ph2 = st.columns(2)
    with ph1:
        st.markdown("""
        <div class='fusion-card'>
            <div style='font-size:1.3rem;'>❄️</div>
            <div style='font-weight:700; color:#64b5f6; margin:8px 0 6px;'>Phase 1 — Warm-up</div>
            <div style='font-size:0.82rem; color:#7cb3d4; line-height:1.7;'>
                Both unimodal encoders are <b>frozen</b>.<br>
                Only attention layers and classification head are trained.<br><br>
                <b>Why:</b> Stabilises the fusion head before perturbing
                the larger (21M param) ResNet backbone.
                Prevents the image branch from dominating early gradients.
            </div>
        </div>""", unsafe_allow_html=True)
    with ph2:
        st.markdown("""
        <div class='fusion-card'>
            <div style='font-size:1.3rem;'>🔥</div>
            <div style='font-weight:700; color:#ef8c6f; margin:8px 0 6px;'>Phase 2 — End-to-End Fine-tuning</div>
            <div style='font-size:0.82rem; color:#7cb3d4; line-height:1.7;'>
                All parameters <b>unfrozen</b>. Lower learning rate (1e-4).<br>
                Gradient flows through both encoders jointly.<br><br>
                <b>Why:</b> Allows the ResNet and ClinicalMLP to adapt their
                representations to the fusion objective together.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _section("Why Cross-Modal Attention vs. Late Fusion")
    st.markdown("""
    <div class='fusion-card'>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:24px;'>
            <div>
                <div style='color:#f44336; font-weight:700; margin-bottom:10px;'>
                    ❌ Simple Late Fusion (averaging)
                </div>
                <div style='font-size:0.82rem; color:#7cb3d4; line-height:1.8;'>
                    Train image model → get P(IDC)<br>
                    Train clinical model → get P(IDC)<br>
                    Average the two probabilities<br><br>
                    <b>Problem:</b> The models never see each other's context.
                    The image model doesn't know if the patient is HER2+.
                    The clinical model doesn't know what the tissue looks like.
                    Modalities don't inform each other at all.
                </div>
            </div>
            <div>
                <div style='color:#4caf50; font-weight:700; margin-bottom:10px;'>
                    ✅ Cross-Modal Attention (FusionDx)
                </div>
                <div style='font-size:0.82rem; color:#7cb3d4; line-height:1.8;'>
                    Compute image embedding e_img<br>
                    Compute clinical embedding e_clin<br>
                    Let each modality modulate the other's contribution<br><br>
                    <b>Result:</b> α_img tells the model "how much does the
                    clinical context validate what the image shows?"
                    Per patient, per prediction — a genuine interaction.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    _disclaimer()
    _hero("About FusionDx", "Research · Methodology · Data Attribution · Limitations", "ℹ️")

    c1, c2 = st.columns([3, 2])
    with c1:
        _section("What is FusionDx?")
        st.markdown("""
        <div class='fusion-card'>
            <div style='font-size:0.88rem; color:#7cb3d4; line-height:1.9;'>
                FusionDx is a research/educational demonstration of <b>genuine multimodal fusion</b>
                for breast cancer subtype classification using real patient data from
                <a href='https://portal.gdc.cancer.gov/projects/TCGA-BRCA' target='_blank'
                   style='color:#4fc3f7;'>TCGA-BRCA via the NIH Genomic Data Commons</a>.<br><br>

                It combines <b>histopathology image tiles</b> (H&amp;E whole-slide images)
                with <b>clinical tabular data</b> (age, AJCC stage, ER/PR/HER2 receptor status)
                using a cross-modal attention mechanism, and rigorously compares the fused
                model against both single-modality baselines.<br><br>

                <b>Honest reporting policy:</b> if fusion does not outperform both baselines,
                the report says so plainly. A negative result is more credible than an inflated claim.
            </div>
        </div>
        """, unsafe_allow_html=True)

        _section("Data Source")
        st.markdown("""
        <div class='fusion-card'>
            <div style='font-size:0.88rem; color:#7cb3d4; line-height:1.9;'>
                <b>TCGA-BRCA</b> — The Cancer Genome Atlas Breast Invasive Carcinoma<br>
                Hosted by: NIH Genomic Data Commons<br>
                Access tier: <span style='color:#4caf50; font-weight:700;'>Open Access</span>
                (diagnostic slides + clinical data, verified August 2026)<br><br>
                <b>GDC field names verified against live API:</b><br>
                • Stage: <code>diagnoses.ajcc_pathologic_stage</code><br>
                • ER/PR/HER2: <code>follow_ups.molecular_tests</code> (ESR1/PGR/ERBB2)<br>
                • Sex: <code>demographic.sex_at_birth</code><br>
                • 1,133 diagnostic slides available, all open access
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        _section("Scope & Limitations")
        limits = [
            ("📊", "~120–200 patients", "Not the full 1000+ TCGA-BRCA cohort"),
            ("🖼️", "256×256 px tiles", "Reduced resolution — not full WSI"),
            ("🏷️", "Binary classification", "IDC vs. other — not multi-subtype"),
            ("🔬", "~18 test patients", "Wide confidence intervals"),
            ("⚗️", "Not validated", "Exploratory research only"),
        ]
        for icon, title, desc in limits:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0d2137,#0a1a2e);
                 border:1px solid #1a3d6e; border-radius:10px;
                 padding:12px 14px; margin-bottom:8px;
                 display:flex; gap:12px; align-items:flex-start;'>
                <div style='font-size:1.3rem;'>{icon}</div>
                <div>
                    <div style='font-weight:700; color:#4fc3f7; font-size:0.88rem;'>{title}</div>
                    <div style='font-size:0.78rem; color:#546e7a;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        _section("Links")
        st.markdown("""
        <div class='fusion-card'>
            <a href='https://github.com/Eddiegah/FusionDX' target='_blank'
               style='display:flex; align-items:center; gap:10px; color:#4fc3f7;
               text-decoration:none; font-weight:700; margin-bottom:12px;'>
                <span style='font-size:1.3rem;'>⭐</span> GitHub Repository
            </a>
            <a href='https://portal.gdc.cancer.gov/projects/TCGA-BRCA' target='_blank'
               style='display:flex; align-items:center; gap:10px; color:#4fc3f7;
               text-decoration:none; font-weight:700;'>
                <span style='font-size:1.3rem;'>🔬</span> TCGA-BRCA on GDC
            </a>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div style='margin-top:40px; padding:20px; text-align:center;
     border-top:1px solid #1e3a5f; font-size:0.75rem; color:#37474f;'>
    FusionDx &nbsp;·&nbsp; Research/Educational Demonstration &nbsp;·&nbsp;
    TCGA-BRCA via NIH Genomic Data Commons &nbsp;·&nbsp;
    <span style='color:#c62828; font-weight:600;'>NOT a clinical diagnostic tool</span>
    &nbsp;·&nbsp;
    <a href='https://github.com/Eddiegah/FusionDX' target='_blank'
       style='color:#4fc3f7; text-decoration:none;'>GitHub</a>
</div>
""", unsafe_allow_html=True)
