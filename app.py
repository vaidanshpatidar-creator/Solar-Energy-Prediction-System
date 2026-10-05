"""
☀️  Solar Energy Prediction System — Streamlit Application
Random Forest Model | Interactive Dashboard
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="☀️ Solar Energy Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(__file__)
DATA_PATH   = os.path.join(BASE, "data",   "solar_data.csv")
MODEL_PATH  = os.path.join(BASE, "models", "rf_solar_model.pkl")
SCALER_PATH = os.path.join(BASE, "models", "scaler.pkl")
METRICS_PATH= os.path.join(BASE, "models", "metrics.json")
IMP_PATH    = os.path.join(BASE, "models", "feature_importance.csv")

FEATURE_COLS = [
    "month", "day", "hour", "day_of_week",
    "solar_radiation", "temperature",
    "cloud_cover", "humidity", "wind_speed",
]

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
DAY_NAMES   = ["Monday","Tuesday","Wednesday","Thursday",
               "Friday","Saturday","Sunday"]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #06090f 0%, #0d1b2a 40%, #0a1628 70%, #0f0c29 100%);
    color: #f0f0f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: rgba(6,9,15,0.92);
    border-right: 1px solid rgba(255,215,0,0.15);
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,215,0,0.2) !important; }

/* ── metric cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,215,0,0.04) 100%);
    border: 1px solid rgba(255,165,0,0.35);
    border-radius: 18px;
    padding: 22px 18px;
    text-align: center;
    backdrop-filter: blur(8px);
}
.metric-card h2 { font-size: 2rem; color: #FFD700; margin: 0; font-weight: 800; }
.metric-card p  { font-size: 0.82rem; color: #a0a8c0; margin: 6px 0 0; }

/* ── section header ── */
.section-header {
    background: linear-gradient(90deg, #f7971e, #ffd200);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 4px;
}

/* ── prediction box ── */
.pred-box {
    background: linear-gradient(135deg, rgba(247,151,30,0.12), rgba(255,210,0,0.08));
    border: 2px solid #FFD700;
    border-radius: 24px;
    padding: 36px;
    text-align: center;
    box-shadow: 0 0 40px rgba(255,215,0,0.12);
}
.pred-box h1 { font-size: 3.5rem; color: #FFD700; margin: 0; font-weight: 800; }
.pred-box p  { font-size: 1rem; color: #d0d0e0; margin: 8px 0 0; }

/* ── info banner ── */
.info-banner {
    background: rgba(255,215,0,0.08);
    border-left: 4px solid #FFD700;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 16px;
    color: #f0f0f0;
}

/* ── HERO ── */
.hero-wrapper { position: relative; text-align: center; padding: 60px 20px 40px; }
.hero-sun { font-size: 6rem; line-height: 1; filter: drop-shadow(0 0 30px rgba(255,215,0,0.6)); display: block; margin-bottom: 14px; }
.hero-title { font-size: 3.2rem; font-weight: 800; background: linear-gradient(90deg,#FFD700 0%,#FF8C00 50%,#FFD700 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px; line-height: 1.15; margin: 0 0 16px; }
.hero-subtitle { font-size: 1.1rem; color: #8899bb; max-width: 680px; margin: 0 auto 28px; line-height: 1.7; }
.hero-badge { display: inline-block; background: rgba(255,215,0,0.12); border: 1px solid rgba(255,215,0,0.35); border-radius: 50px; padding: 6px 18px; font-size: 0.8rem; color: #FFD700; margin: 0 4px 8px; font-weight: 600; letter-spacing: 0.5px; }

/* ── pipeline ── */
.pipeline-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 0; margin: 20px 0; }
.pipeline-step { text-align: center; padding: 20px 10px; position: relative; }
.pipeline-step::after { content: '\2192'; position: absolute; right: -8px; top: 50%; transform: translateY(-50%); color: rgba(255,215,0,0.4); font-size: 1.4rem; }
.pipeline-step:last-child::after { display: none; }
.pipeline-icon { font-size: 2.2rem; display: block; margin-bottom: 8px; filter: drop-shadow(0 0 8px rgba(255,215,0,0.3)); }
.pipeline-label { font-size: 0.78rem; color: #FFD700; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; }
.pipeline-desc { font-size: 0.72rem; color: #7788aa; margin-top: 4px; line-height: 1.4; }

/* ── feature cards ── */
.feat-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; margin: 14px 0; }
.feat-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 18px 16px; }
.feat-card:hover { border-color: rgba(255,215,0,0.3); background: rgba(255,215,0,0.05); }
.feat-icon { font-size: 1.6rem; margin-bottom: 8px; display: block; }
.feat-name { font-size: 0.88rem; font-weight: 700; color: #e0e4f0; }
.feat-desc { font-size: 0.76rem; color: #6677aa; margin-top: 3px; line-height: 1.4; }

/* ── detect vs predict ── */
.dvp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 16px 0; }
.dvp-card { border-radius: 20px; padding: 28px 24px; }
.dvp-card.detect { background: linear-gradient(135deg,rgba(0,180,255,0.08),rgba(0,100,200,0.05)); border: 1.5px solid rgba(0,180,255,0.3); }
.dvp-card.predict { background: linear-gradient(135deg,rgba(255,165,0,0.1),rgba(255,100,0,0.05)); border: 1.5px solid rgba(255,165,0,0.35); }
.dvp-icon { font-size: 2.4rem; margin-bottom: 10px; display: block; }
.dvp-title { font-size: 1.15rem; font-weight: 800; margin-bottom: 8px; }
.dvp-card.detect .dvp-title { color: #00C8FF; }
.dvp-card.predict .dvp-title { color: #FFB347; }
.dvp-body { font-size: 0.82rem; color: #8899bb; line-height: 1.65; }
.dvp-pill { display: inline-block; border-radius: 6px; padding: 2px 9px; font-size: 0.72rem; font-weight: 700; margin: 3px 2px 0; }
.dvp-card.detect .dvp-pill { background: rgba(0,200,255,0.15); color: #00C8FF; }
.dvp-card.predict .dvp-pill { background: rgba(255,180,70,0.15); color: #FFB347; }

/* ── stat strip ── */
.stat-strip { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 20px 0 10px; }
.stat-item { background: rgba(255,215,0,0.06); border: 1px solid rgba(255,215,0,0.2); border-radius: 14px; padding: 18px 12px; text-align: center; }
.stat-num { font-size: 1.9rem; font-weight: 800; color: #FFD700; line-height: 1; }
.stat-lbl { font-size: 0.74rem; color: #7788aa; margin-top: 5px; text-transform: uppercase; letter-spacing: 0.6px; }

/* ── tech stack ── */
.tech-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0; }
.tech-chip { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); border-radius: 8px; padding: 6px 14px; font-size: 0.78rem; color: #aabbd0; font-weight: 600; }

/* ── section box ── */
.section-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 20px; padding: 28px 26px; margin-bottom: 20px; }
.section-box-title { font-size: 1.1rem; font-weight: 700; color: #FFD700; margin-bottom: 14px; }

/* ── divider ── */
hr { border-color: rgba(255,255,255,0.07); margin: 28px 0; }

/* ── slider text ── */
.stSlider label { color: #f0f0f0 !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: rgba(255,215,0,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers / caching
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔄 Loading model …")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


@st.cache_data(show_spinner="📂 Loading dataset …")
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


def load_importance():
    if not os.path.exists(IMP_PATH):
        return None
    return pd.read_csv(IMP_PATH)


def predict(model, scaler, inputs: dict) -> float:
    df = pd.DataFrame([inputs])[FEATURE_COLS]
    scaled = scaler.transform(df)
    return float(model.predict(scaled)[0])


def energy_level(val: float):
    if val < 2:   return "🔴 Low",      "#FF6B6B"
    if val < 8:   return "🟡 Moderate", "#FFD700"
    if val < 15:  return "🟢 High",     "#00FF88"
    return             "⚡ Excellent",  "#00CFFF"


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ☀️ Solar Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home & Predict",
         "📊 Data Explorer",
         "🌡️ Feature Analysis",
         "📈 Model Performance",
         "🔮 Batch Prediction"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
**Model:** Random Forest Regressor  
**Features:** 9 environmental + temporal  
**Target:** Energy Output (kWh)  
**Dataset:** 10,000 synthetic samples
    """)

# ── Auto-train if model is missing ───────────────────────────────────────────
model, scaler = load_model()
if model is None:
    st.info("🚀 First launch detected — training model, please wait …")
    with st.spinner("Training Random Forest …"):
        os.chdir(BASE)
        from generate_dataset import generate_solar_dataset
        generate_solar_dataset(output_path=DATA_PATH)
        from train_model import load_data as td_load, train
        train(td_load(DATA_PATH))
        st.cache_resource.clear()
        st.cache_data.clear()
    st.success("✅ Model trained! Reloading …")
    st.rerun()

df      = load_data()
metrics = load_metrics()
fi_df   = load_importance()


# ===============================================================================
# PAGE 1 — Home & Predict
# ===============================================================================

if page == "🏠 Home & Predict":

    # ── HERO ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:60px 20px 36px;'>
        <div style='font-size:5rem; line-height:1; filter:drop-shadow(0 0 30px rgba(255,215,0,0.6));
                    margin-bottom:16px;'>&#9728;&#65039;</div>
        <div style='font-size:2.8rem; font-weight:800; line-height:1.15; margin-bottom:16px;
                    background:linear-gradient(90deg,#FFD700 0%,#FF8C00 50%,#FFD700 100%);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            Solar Energy Prediction System
        </div>
        <p style='font-size:1.05rem; color:#8899bb; max-width:680px; margin:0 auto 28px; line-height:1.75;'>
            An end-to-end Machine Learning platform that
            <b style='color:#FFD700;'>detects</b> real-time environmental conditions and
            <b style='color:#FFB347;'>predicts</b> solar panel energy output using a
            <b style='color:#FFD700;'>Random Forest</b> model — helping operators, engineers
            and researchers optimise renewable energy generation.
        </p>
        <div>
            <span style='display:inline-block; background:rgba(255,215,0,0.12);
                border:1px solid rgba(255,215,0,0.35); border-radius:50px;
                padding:6px 16px; font-size:0.78rem; color:#FFD700;
                margin:4px; font-weight:600;'>Random Forest ML</span>
            <span style='display:inline-block; background:rgba(255,215,0,0.12);
                border:1px solid rgba(255,215,0,0.35); border-radius:50px;
                padding:6px 16px; font-size:0.78rem; color:#FFD700;
                margin:4px; font-weight:600;'>9 Feature Inputs</span>
            <span style='display:inline-block; background:rgba(255,215,0,0.12);
                border:1px solid rgba(255,215,0,0.35); border-radius:50px;
                padding:6px 16px; font-size:0.78rem; color:#FFD700;
                margin:4px; font-weight:600;'>10,000 Training Samples</span>
            <span style='display:inline-block; background:rgba(255,215,0,0.12);
                border:1px solid rgba(255,215,0,0.35); border-radius:50px;
                padding:6px 16px; font-size:0.78rem; color:#FFD700;
                margin:4px; font-weight:600;'>R&#178; = 0.9883</span>
            <span style='display:inline-block; background:rgba(255,215,0,0.12);
                border:1px solid rgba(255,215,0,0.35); border-radius:50px;
                padding:6px 16px; font-size:0.78rem; color:#FFD700;
                margin:4px; font-weight:600;'>Real-Time Prediction</span>
            <span style='display:inline-block; background:rgba(255,215,0,0.12);
                border:1px solid rgba(255,215,0,0.35); border-radius:50px;
                padding:6px 16px; font-size:0.78rem; color:#FFD700;
                margin:4px; font-weight:600;'>Batch Simulation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── LIVE MODEL STATS ────────────────────────────────────────────────────────
    if metrics:
        st.markdown("""<div style='text-align:center; margin-bottom:10px;'>
            <span style='font-size:0.72rem; text-transform:uppercase; letter-spacing:2.5px; color:#445566;'>
                Live Model Performance Metrics
            </span></div>""", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        _card = ("background:linear-gradient(135deg,rgba(255,255,255,0.05),rgba(255,215,0,0.04));"
                 "border:1px solid rgba(255,165,0,0.35); border-radius:18px;"
                 "padding:22px 18px; text-align:center;")
        for col, label, val, sub in zip(
            [c1, c2, c3, c4],
            ["R&#178; Score", "MAE", "RMSE", "CV R&#178;"],
            [metrics["R2"], metrics["MAE"], metrics["RMSE"], metrics["CV_R2"]],
            ["Variance explained", "Mean Abs Error (kWh)", "Root Mean Sq Error (kWh)", "5-Fold Cross-Validation"],
        ):
            col.markdown(f"""
            <div style='{_card}'>
                <div style='font-size:2rem; color:#FFD700; font-weight:800; margin:0;'>{val}</div>
                <p style='font-size:0.82rem; color:#a0a8c0; margin:6px 0 0;'>
                    <b>{label}</b><br>
                    <span style='font-size:0.7rem; color:#667788;'>{sub}</span>
                </p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);'>", unsafe_allow_html=True)

    # ── WHAT THIS SYSTEM DOES ───────────────────────────────────────────────────
    _box = ("background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);"
            "border-radius:20px; padding:28px 26px; margin-bottom:20px;")
    _title_style = "font-size:1.1rem; font-weight:700; color:#FFD700; margin-bottom:14px;"

    st.markdown(f"""
    <div style='{_box}'>
        <div style='{_title_style}'>&#127919; What This System Does</div>
        <p style='color:#7788aa; font-size:0.88rem; line-height:1.85; margin:0 0 20px;'>
            This platform combines <b style='color:#e0e4f0;'>environmental condition detection</b>
            and <b style='color:#e0e4f0;'>ML-based energy forecasting</b> into a single unified
            dashboard. It reads 9 real-world measurable parameters and outputs a precise energy
            generation estimate in <b style='color:#FFD700;'>kWh</b>.
            The system is designed to serve:
        </p>
        <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:14px;'>
            <div style='background:rgba(0,200,255,0.07); border:1px solid rgba(0,200,255,0.2);
                        border-radius:14px; padding:18px;'>
                <div style='font-size:1.6rem; margin-bottom:8px;'>&#127981;</div>
                <div style='font-size:0.86rem; font-weight:700; color:#00C8FF; margin-bottom:6px;'>Solar Farm Operators</div>
                <div style='font-size:0.76rem; color:#5566aa; line-height:1.65;'>
                    Forecast daily output, plan grid dispatch and reduce curtailment losses.
                </div>
            </div>
            <div style='background:rgba(255,165,0,0.07); border:1px solid rgba(255,165,0,0.2);
                        border-radius:14px; padding:18px;'>
                <div style='font-size:1.6rem; margin-bottom:8px;'>&#128300;</div>
                <div style='font-size:0.86rem; font-weight:700; color:#FFB347; margin-bottom:6px;'>Researchers &amp; Engineers</div>
                <div style='font-size:0.76rem; color:#5566aa; line-height:1.65;'>
                    Study feature relationships, model behaviour and seasonal energy patterns.
                </div>
            </div>
            <div style='background:rgba(100,255,150,0.05); border:1px solid rgba(100,255,150,0.18);
                        border-radius:14px; padding:18px;'>
                <div style='font-size:1.6rem; margin-bottom:8px;'>&#127968;</div>
                <div style='font-size:0.86rem; font-weight:700; color:#66FF99; margin-bottom:6px;'>Residential Users</div>
                <div style='font-size:0.76rem; color:#5566aa; line-height:1.65;'>
                    Estimate home solar panel yield and optimise appliance scheduling.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ML PIPELINE ─────────────────────────────────────────────────────────────
    _step = ("text-align:center; padding:20px 12px; position:relative;"
             "background:rgba(255,255,255,0.03); border-radius:14px;")
    _step_icon = "font-size:2rem; display:block; margin-bottom:8px;"
    _step_lbl  = "font-size:0.75rem; color:#FFD700; font-weight:700; text-transform:uppercase; letter-spacing:0.8px;"
    _step_desc = "font-size:0.71rem; color:#5566aa; margin-top:5px; line-height:1.45;"
    _arrow     = "font-size:1.4rem; color:rgba(255,215,0,0.35); display:flex; align-items:center;"

    st.markdown(f"""
    <div style='{_box}'>
        <div style='{_title_style}'>&#128260; End-to-End ML Pipeline</div>
        <div style='display:grid; grid-template-columns:1fr 28px 1fr 28px 1fr 28px 1fr 28px 1fr; gap:6px; align-items:center;'>
            <div style='{_step}'><span style='{_step_icon}'>&#128225;</span>
                <div style='{_step_lbl}'>1. Data Collection</div>
                <div style='{_step_desc}'>10,000 physics-based synthetic records with seasonal &amp; diurnal patterns</div>
            </div>
            <div style='{_arrow}'>&#8594;</div>
            <div style='{_step}'><span style='{_step_icon}'>&#9986;&#65039;</span>
                <div style='{_step_lbl}'>2. Preprocessing</div>
                <div style='{_step_desc}'>StandardScaler, 80/20 train-test split, feature validation</div>
            </div>
            <div style='{_arrow}'>&#8594;</div>
            <div style='{_step}'><span style='{_step_icon}'>&#127794;</span>
                <div style='{_step_lbl}'>3. RF Training</div>
                <div style='{_step_desc}'>200 trees, sqrt feature sampling, parallel n_jobs=-1</div>
            </div>
            <div style='{_arrow}'>&#8594;</div>
            <div style='{_step}'><span style='{_step_icon}'>&#128202;</span>
                <div style='{_step_lbl}'>4. Evaluation</div>
                <div style='{_step_desc}'>MAE, RMSE, R&#178;, 5-Fold CV; model persisted via joblib</div>
            </div>
            <div style='{_arrow}'>&#8594;</div>
            <div style='{_step}'><span style='{_step_icon}'>&#9889;</span>
                <div style='{_step_lbl}'>5. Live Inference</div>
                <div style='{_step_desc}'>Sub-second prediction with gauge + AI insight output</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 9 FEATURE CARDS ─────────────────────────────────────────────────────────
    _fcard = ("background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.09);"
              "border-radius:14px; padding:18px 16px;")
    _fname = "font-size:0.88rem; font-weight:700; color:#e0e4f0; margin:6px 0 3px;"
    _fdesc = "font-size:0.75rem; color:#5566aa; line-height:1.45;"

    features = [
        ("&#9728;&#65039;", "Solar Radiation (W/m&#178;)",
         "Primary driver — shortwave irradiance hitting the panel surface (0-1100 W/m&#178;). Peaks at solar noon in summer."),
        ("&#127777;&#65039;", "Temperature (&#176;C)",
         "PV cell efficiency degrades 0.4%/&#176;C above 25&#176;C — detected as both performance risk and thermal stress."),
        ("&#9729;&#65039;", "Cloud Cover (%)",
         "Heavy overcast reduces radiation by up to 80%. The single strongest attenuator of energy output."),
        ("&#128167;", "Humidity (%)",
         "High humidity causes optical diffusion and soiling — secondary attenuation factor modelled in the RF."),
        ("&#128168;", "Wind Speed (m/s)",
         "Convective cooling partially offsets thermal losses. Follows Weibull distribution in dataset."),
        ("&#128336;", "Hour of Day (0-23)",
         "Solar angle changes hourly — output follows a Gaussian bell curve peaked at solar noon (12:00)."),
        ("&#128197;", "Month (1-12)",
         "Summer months produce 2x winter output due to longer days and higher solar declination angle."),
        ("&#128198;", "Day of Month (1-31)",
         "Fine-grained temporal signal encoding sub-monthly irradiance and maintenance scheduling patterns."),
        ("&#128467;&#65039;", "Day of Week (0-6)",
         "Encodes weekly demand cycles — useful for grid load-balancing and industrial integration."),
    ]

    feat_rows = [features[i:i+3] for i in range(0, 9, 3)]
    feat_html = f"<div style='{_box}'><div style='{_title_style}'>&#127758; 9 Input Features — Detected &amp; Used</div>"
    for row in feat_rows:
        feat_html += "<div style='display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:14px;'>"
        for icon, name, desc in row:
            feat_html += f"""
            <div style='{_fcard}'>
                <div style='font-size:1.6rem; margin-bottom:6px;'>{icon}</div>
                <div style='{_fname}'>{name}</div>
                <div style='{_fdesc}'>{desc}</div>
            </div>"""
        feat_html += "</div>"
    feat_html += "</div>"
    st.markdown(feat_html, unsafe_allow_html=True)

    # ── STATS + TECH STACK ──────────────────────────────────────────────────────
    _stat_item = ("background:rgba(255,215,0,0.06); border:1px solid rgba(255,215,0,0.2);"
                  "border-radius:14px; padding:18px 12px; text-align:center;")
    _chip = ("display:inline-block; background:rgba(255,255,255,0.05);"
             "border:1px solid rgba(255,255,255,0.12); border-radius:8px;"
             "padding:5px 13px; font-size:0.77rem; color:#aabbd0;"
             "font-weight:600; margin:3px;")

    col_s, col_t = st.columns(2)

    with col_s:
        st.markdown(f"""
        <div style='{_box}'>
            <div style='{_title_style}'>&#128230; Project Statistics</div>
            <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:12px;'>
                <div style='{_stat_item}'><div style='font-size:1.9rem;font-weight:800;color:#FFD700;'>10K</div>
                    <div style='font-size:0.73rem;color:#5566aa;margin-top:4px;text-transform:uppercase;'>Training Records</div></div>
                <div style='{_stat_item}'><div style='font-size:1.9rem;font-weight:800;color:#FFD700;'>9</div>
                    <div style='font-size:0.73rem;color:#5566aa;margin-top:4px;text-transform:uppercase;'>Input Features</div></div>
                <div style='{_stat_item}'><div style='font-size:1.9rem;font-weight:800;color:#FFD700;'>200</div>
                    <div style='font-size:0.73rem;color:#5566aa;margin-top:4px;text-transform:uppercase;'>Decision Trees</div></div>
                <div style='{_stat_item}'><div style='font-size:1.9rem;font-weight:800;color:#FFD700;'>98.8%</div>
                    <div style='font-size:0.73rem;color:#5566aa;margin-top:4px;text-transform:uppercase;'>R&#178; Accuracy</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_t:
        st.markdown(f"""
        <div style='{_box}'>
            <div style='{_title_style}'>&#128295; Technology Stack</div>
            <div style='margin-bottom:14px;'>
                <div style='font-size:0.7rem;color:#445566;text-transform:uppercase;
                            letter-spacing:1.2px;margin-bottom:8px;'>ML &amp; Data Science</div>
                <span style='{_chip}'>scikit-learn</span>
                <span style='{_chip}'>RandomForest</span>
                <span style='{_chip}'>pandas</span>
                <span style='{_chip}'>numpy</span>
                <span style='{_chip}'>joblib</span>
            </div>
            <div style='margin-bottom:14px;'>
                <div style='font-size:0.7rem;color:#445566;text-transform:uppercase;
                            letter-spacing:1.2px;margin-bottom:8px;'>Visualisation</div>
                <span style='{_chip}'>Plotly Express</span>
                <span style='{_chip}'>Gauge Charts</span>
                <span style='{_chip}'>Heatmaps</span>
                <span style='{_chip}'>Scatter Plots</span>
            </div>
            <div>
                <div style='font-size:0.7rem;color:#445566;text-transform:uppercase;
                            letter-spacing:1.2px;margin-bottom:8px;'>Interface</div>
                <span style='{_chip}'>Streamlit</span>
                <span style='{_chip}'>Custom CSS</span>
                <span style='{_chip}'>Inline HTML5</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── LIVE PREDICTION FORM ─────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; margin-bottom:22px;'>
        <div style='font-size:1.55rem; font-weight:800;
                    background:linear-gradient(90deg,#f7971e,#ffd200);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            &#9889; Live Energy Prediction
        </div>
        <div style='font-size:0.84rem; color:#445566; margin-top:7px;'>
            Set the environmental conditions below — the model predicts output instantly
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### &#128197; Temporal")
            month       = st.selectbox("Month", range(1, 13),
                            format_func=lambda x: MONTH_NAMES[x-1])
            day         = st.slider("Day of Month", 1, 31, 15)
            hour        = st.slider("Hour of Day", 0, 23, 12,
                            help="0 = midnight, 12 = noon")
            day_of_week = st.selectbox("Day of Week", range(7),
                            format_func=lambda x: DAY_NAMES[x])

        with col2:
            st.markdown("##### &#9729; Atmospheric")
            solar_rad   = st.slider("Solar Radiation (W/m2)", 0, 1100, 600)
            cloud_cover = st.slider("Cloud Cover (%)", 0, 100, 20)
            humidity    = st.slider("Humidity (%)", 10, 100, 50)

        with col3:
            st.markdown("##### &#127777; Thermal & Wind")
            temperature = st.slider("Temperature (C)", -10, 50, 25)
            wind_speed  = st.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0, step=0.5)

        submit = st.form_submit_button("Predict Energy Output", use_container_width=True)

    if submit:
        inputs = dict(
            month=month, day=day, hour=hour, day_of_week=day_of_week,
            solar_radiation=solar_rad, temperature=temperature,
            cloud_cover=cloud_cover, humidity=humidity, wind_speed=wind_speed,
        )
        result = predict(model, scaler, inputs)
        level, color = energy_level(result)

        st.markdown("<br>", unsafe_allow_html=True)
        cl, cm, cr = st.columns([1, 2, 1])
        with cm:
            _pred_box = ("background:linear-gradient(135deg,rgba(247,151,30,0.12),rgba(255,210,0,0.08));"
                         "border:2px solid #FFD700; border-radius:24px; padding:36px;"
                         "text-align:center; box-shadow:0 0 40px rgba(255,215,0,0.12);")
            st.markdown(f"""
            <div style='{_pred_box}'>
                <p style='color:#6677aa; font-size:0.82rem; text-transform:uppercase;
                          letter-spacing:2.5px; margin-bottom:8px;'>Predicted Solar Energy Output</p>
                <div style='font-size:3.5rem; color:#FFD700; font-weight:800; margin:0;'>
                    {result:.2f} <span style='font-size:1.6rem;'>kWh</span>
                </div>
                <p style='font-size:1.2rem; color:{color}; font-weight:700; margin-top:12px;'>{level}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Factor breakdown cards
        cloud_impact = round(0.8 * cloud_cover, 1)
        temp_loss    = round(max(temperature - 25, 0) * 0.4, 2)
        night        = (hour < 6) or (hour > 20)
        _mini_card = ("background:linear-gradient(135deg,rgba(255,255,255,0.05),rgba(255,215,0,0.04));"
                      "border:1px solid rgba(255,165,0,0.35); border-radius:18px;"
                      "padding:22px 18px; text-align:center;")

        fa, fb, fc = st.columns(3)
        fa.markdown(f"""
        <div style='{_mini_card}'>
            <div style='font-size:1.6rem; color:#FFD700; font-weight:800;'>{cloud_cover}%</div>
            <p style='font-size:0.82rem; color:#a0a8c0; margin:6px 0 0;'>Cloud Cover Detected<br>
            <span style='color:#FF6B6B; font-size:0.72rem;'>Attenuating ~{cloud_impact:.0f}% radiation</span></p>
        </div>""", unsafe_allow_html=True)
        fb.markdown(f"""
        <div style='{_mini_card}'>
            <div style='font-size:1.6rem; color:#FFD700; font-weight:800;'>{temperature}&#176;C</div>
            <p style='font-size:0.82rem; color:#a0a8c0; margin:6px 0 0;'>Temperature Detected<br>
            <span style='color:#FFB347; font-size:0.72rem;'>
            {f"-{temp_loss:.2f}% PV efficiency loss" if temp_loss > 0 else "Optimal thermal range"}</span></p>
        </div>""", unsafe_allow_html=True)
        fc.markdown(f"""
        <div style='{_mini_card}'>
            <div style='font-size:1.6rem; color:#FFD700; font-weight:800;'>{"Night" if night else "Day"}</div>
            <p style='font-size:0.82rem; color:#a0a8c0; margin:6px 0 0;'>Time Detected<br>
            <span style='color:{"#FF6B6B" if night else "#00FF88"}; font-size:0.72rem;'>
            {"Night — output suppressed to ~2%" if night else f"Hour {hour:02d}:00 — solar active"}</span></p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = result,
            title = {"text": "Energy Output (kWh)", "font": {"color": "#8899aa", "size": 15}},
            number= {"font": {"color": "#FFD700", "size": 52}, "suffix": " kWh"},
            delta = {"reference": 10, "increasing": {"color": "#00FF88"},
                     "decreasing": {"color": "#FF6B6B"}},
            gauge = {
                "axis":  {"range": [0, 25], "tickcolor": "#445566",
                          "tickfont": {"color": "#7788aa"}},
                "bar":   {"color": "#FFD700", "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0,  2],  "color": "rgba(255,107,107,0.15)"},
                    {"range": [2,  8],  "color": "rgba(255,215,0,0.12)"},
                    {"range": [8,  15], "color": "rgba(0,255,136,0.12)"},
                    {"range": [15, 25], "color": "rgba(0,207,255,0.12)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": result},
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font_color   ="#f0f0f0",
            height=320,
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # AI Insight
        if result < 2:
            tip_icon, tip = "&#128308;", "Heavy cloud cover or nighttime detected. Battery storage and load-shifting recommended for this period."
        elif result < 8:
            tip_icon, tip = "&#128993;", "Moderate output detected. Partial cloud cover or off-peak solar hour is reducing efficiency. Check panel tilt angle."
        elif result < 15:
            tip_icon, tip = "&#128994;", "Good energy output detected. Ensure panels are clean and optimally oriented. Consider pre-charging battery storage."
        else:
            tip_icon, tip = "&#9889;", "Excellent output detected! Peak generation conditions active. Export surplus energy to the grid for maximum ROI."
        _banner = ("background:rgba(255,215,0,0.08); border-left:4px solid #FFD700;"
                   "border-radius:10px; padding:14px 20px; margin-bottom:16px; color:#f0f0f0;")
        st.markdown(f"""
        <div style='{_banner}'>
            {tip_icon} <b>AI Insight:</b> {tip}
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Data Explorer":
    st.markdown("<div class='section-header'>📊 Dataset Explorer</div>", unsafe_allow_html=True)
    st.markdown(f"Dataset contains **{len(df):,} records** with **{len(df.columns)} features**.")

    tab1, tab2, tab3 = st.tabs(["🗂️ Raw Data", "📈 Distributions", "🔥 Correlations"])

    with tab1:
        n = st.slider("Rows to display", 10, 200, 50)
        st.dataframe(df.head(n), use_container_width=True)
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        try:
            st.dataframe(df.describe().T.style.background_gradient(cmap="YlOrRd"), use_container_width=True)
        except Exception:
            st.dataframe(df.describe().T, use_container_width=True)

    with tab2:
        col = st.selectbox("Select feature", df.columns.tolist())
        fig = px.histogram(
            df, x=col, nbins=60, color_discrete_sequence=["#FFD700"],
            title=f"Distribution of {col}",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Monthly energy trend
        monthly = df.groupby("month")["energy_output"].mean().reset_index()
        monthly["month_name"] = monthly["month"].apply(lambda x: MONTH_NAMES[x-1])
        fig2 = px.bar(
            monthly, x="month_name", y="energy_output",
            color="energy_output", color_continuous_scale="YlOrRd",
            title="Average Energy Output by Month",
            template="plotly_dark",
            labels={"energy_output": "Avg Energy (kWh)", "month_name": "Month"},
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        numeric_df = df.select_dtypes(include=np.number)
        corr = numeric_df.corr()
        fig_heatmap = px.imshow(
            corr, color_continuous_scale="RdYlGn",
            title="Feature Correlation Heatmap",
            template="plotly_dark", text_auto=".2f",
        )
        fig_heatmap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=550,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Feature Analysis
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🌡️ Feature Analysis":
    st.markdown("<div class='section-header'>🌡️ Feature Analysis</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⚡ Feature Importance", "📉 Scatter Relations", "⏰ Time Patterns"])

    with tab1:
        if fi_df is not None:
            fig = px.bar(
                fi_df, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="YlOrRd",
                title="Random Forest Feature Importance",
                template="plotly_dark",
                labels={"importance": "Importance Score", "feature": "Feature"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False,
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Top Features")
            for i, row in fi_df.iterrows():
                pct = row["importance"] * 100
                st.markdown(f"""
                <div style='margin-bottom:8px;'>
                  <span style='color:#f0f0f0;'>{row['feature']}</span>
                  <div style='background:#333; border-radius:10px; height:14px; margin-top:4px;'>
                    <div style='background:linear-gradient(90deg,#f7971e,#ffd200);
                                width:{min(pct*5,100):.1f}%; height:14px; border-radius:10px;'></div>
                  </div>
                  <small style='color:#b0b0c0;'>{pct:.2f}%</small>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        x_feat = st.selectbox("X-axis feature", FEATURE_COLS, index=4)   # solar_radiation
        y_feat = "energy_output"
        sample = df.sample(min(2000, len(df)), random_state=1)
        fig = px.scatter(
            sample, x=x_feat, y=y_feat,
            color="cloud_cover", color_continuous_scale="YlOrRd",
            title=f"{x_feat} vs {y_feat}",
            template="plotly_dark",
            opacity=0.6,
            labels={x_feat: x_feat.replace("_", " ").title(),
                    y_feat: "Energy Output (kWh)"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        hourly  = df.groupby("hour")["energy_output"].mean().reset_index()
        fig_h = px.line(
            hourly, x="hour", y="energy_output",
            title="Average Energy Output by Hour of Day",
            template="plotly_dark",
            markers=True,
            color_discrete_sequence=["#FFD700"],
            labels={"hour": "Hour", "energy_output": "Avg Energy (kWh)"},
        )
        fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_h, use_container_width=True)

        # Monthly heatmap: hour × month
        pivot = df.groupby(["hour", "month"])["energy_output"].mean().reset_index()
        pivot_wide = pivot.pivot(index="hour", columns="month", values="energy_output")
        pivot_wide.columns = [MONTH_NAMES[c-1] for c in pivot_wide.columns]
        fig_hm = px.imshow(
            pivot_wide, color_continuous_scale="YlOrRd",
            title="Energy Output Heatmap: Hour × Month",
            template="plotly_dark",
            labels={"x": "Month", "y": "Hour", "color": "Avg kWh"},
        )
        fig_hm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420)
        st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Model Performance":
    st.markdown("<div class='section-header'>📈 Model Performance Dashboard</div>", unsafe_allow_html=True)

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, desc in zip(
            [c1, c2, c3, c4],
            ["R² Score", "MAE", "RMSE", "CV R²"],
            [metrics["R2"], metrics["MAE"], metrics["RMSE"], metrics["CV_R2"]],
            ["Variance explained", "Mean Absolute Error (kWh)",
             "Root Mean Sq Error (kWh)", "5-Fold Cross-Validation R²"],
        ):
            col.markdown(f"""
            <div class='metric-card'>
                <h2>{val}</h2>
                <p><b>{label}</b><br>{desc}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs predicted scatter (sample from df with model re-prediction)
    if df is not None and model is not None:
        sample = df.sample(min(1000, len(df)), random_state=7)
        X_s = scaler.transform(sample[FEATURE_COLS])
        y_pred = model.predict(X_s)
        y_true = sample["energy_output"].values

        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode="markers",
            marker=dict(color="#FFD700", opacity=0.5, size=5),
            name="Predictions",
        ))
        max_val = max(y_true.max(), y_pred.max()) * 1.05
        fig_ap.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", line=dict(color="#FF6B6B", dash="dash", width=2),
            name="Ideal (y=x)",
        ))
        fig_ap.update_layout(
            title="Actual vs Predicted Energy Output",
            xaxis_title="Actual (kWh)", yaxis_title="Predicted (kWh)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=450,
        )
        st.plotly_chart(fig_ap, use_container_width=True)

        # Residual distribution
        residuals = y_pred - y_true
        fig_res = px.histogram(
            x=residuals, nbins=60,
            title="Residuals Distribution (Predicted − Actual)",
            color_discrete_sequence=["#7B68EE"],
            template="plotly_dark",
            labels={"x": "Residual (kWh)", "y": "Count"},
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="#FFD700")
        fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("""
    <div style='background:rgba(255,215,0,0.08); border-left:4px solid #FFD700;
                border-radius:10px; padding:14px 20px; color:#f0f0f0;'>
        &#8505;&#65039; <b>Methodology:</b> Random Forest with 200 estimators, max_depth=20,
        trained on 8,000 samples (80/20 split). Evaluation includes 5-fold cross-validation
        to ensure generalisability.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Batch Prediction":
    st.markdown("<div class='section-header'>🔮 Batch Prediction</div>", unsafe_allow_html=True)
    st.markdown("Upload a CSV file with the required columns or generate a simulation.")

    tab1, tab2 = st.tabs(["📤 Upload CSV", "🎲 Simulate Day"])

    with tab1:
        st.markdown(f"**Required columns:** `{', '.join(FEATURE_COLS)}`")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            try:
                up_df = pd.read_csv(uploaded)
                missing = [c for c in FEATURE_COLS if c not in up_df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    X_up = scaler.transform(up_df[FEATURE_COLS])
                    up_df["predicted_energy_kWh"] = model.predict(X_up).round(3)
                    st.success(f"✅ Predicted {len(up_df):,} rows.")
                    st.dataframe(up_df.tail(20), use_container_width=True)
                    csv_out = up_df.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Results", csv_out,
                                       "solar_predictions.csv", "text/csv")
                    fig = px.line(
                        up_df.reset_index(), x=up_df.index, y="predicted_energy_kWh",
                        title="Predicted Energy Output", template="plotly_dark",
                        color_discrete_sequence=["#FFD700"],
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        st.markdown("### Simulate a Full Day (hourly)")
        col1, col2 = st.columns(2)
        with col1:
            sim_month = st.selectbox("Month", range(1, 13),
                            format_func=lambda x: MONTH_NAMES[x-1], key="sim_m")
            sim_day   = st.slider("Day", 1, 31, 15, key="sim_d")
            sim_dow   = st.selectbox("Day of Week", range(7),
                            format_func=lambda x: DAY_NAMES[x], key="sim_dow")
        with col2:
            sim_cloud = st.slider("Cloud Cover (%)", 0, 100, 20, key="sim_cl")
            sim_humid = st.slider("Humidity (%)", 10, 100, 50, key="sim_hu")
            sim_wind  = st.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0, key="sim_ws")

        if st.button("🚀 Run 24-Hour Simulation", use_container_width=True):
            rows = []
            for h in range(24):
                # Physics-based approximations for radiation & temp
                hour_f = np.exp(-0.5 * ((h - 12) / 3.5) ** 2)
                month_f = 0.5 + 0.5 * np.sin((sim_month - 3) * np.pi / 6)
                rad = float(np.clip(1000 * hour_f * month_f, 0, 1100))
                temp_base = 15 + 10 * np.sin((sim_month - 1) * np.pi / 6)
                temp = float(temp_base + 5 * np.sin((h - 6) * np.pi / 12))
                rows.append(dict(
                    month=sim_month, day=sim_day, hour=h, day_of_week=sim_dow,
                    solar_radiation=round(rad, 1), temperature=round(temp, 1),
                    cloud_cover=sim_cloud, humidity=sim_humid, wind_speed=sim_wind,
                ))

            sim_df = pd.DataFrame(rows)
            X_sim  = scaler.transform(sim_df[FEATURE_COLS])
            sim_df["predicted_energy_kWh"] = model.predict(X_sim).round(3)

            total = sim_df["predicted_energy_kWh"].sum()
            peak  = sim_df["predicted_energy_kWh"].max()
            peak_h= int(sim_df.loc[sim_df["predicted_energy_kWh"].idxmax(), "hour"])

            st.markdown("<br>", unsafe_allow_html=True)
            cc1, cc2, cc3 = st.columns(3)
            cc1.markdown(f"<div class='metric-card'><h2>{total:.1f}</h2><p>Total kWh (Day)</p></div>",
                         unsafe_allow_html=True)
            cc2.markdown(f"<div class='metric-card'><h2>{peak:.2f}</h2><p>Peak kWh</p></div>",
                         unsafe_allow_html=True)
            cc3.markdown(f"<div class='metric-card'><h2>{peak_h:02d}:00</h2><p>Peak Hour</p></div>",
                         unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            fig_sim = go.Figure()
            fig_sim.add_trace(go.Bar(
                x=sim_df["hour"], y=sim_df["solar_radiation"],
                name="Solar Radiation (W/m²)", marker_color="#FF8C00", opacity=0.4,
                yaxis="y2",
            ))
            fig_sim.add_trace(go.Scatter(
                x=sim_df["hour"], y=sim_df["predicted_energy_kWh"],
                name="Energy Output (kWh)", mode="lines+markers",
                line=dict(color="#FFD700", width=3),
                marker=dict(size=8),
            ))
            fig_sim.update_layout(
                title=f"24-Hour Energy Simulation — {MONTH_NAMES[sim_month-1]} {sim_day}",
                xaxis=dict(title="Hour of Day", dtick=1),
                yaxis=dict(title="Energy Output (kWh)", side="left"),
                yaxis2=dict(title="Solar Radiation (W/m²)", overlaying="y",
                            side="right", showgrid=False),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(x=0.01, y=0.99),
                height=450,
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            csv_out = sim_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Simulation", csv_out,
                               "simulation_results.csv", "text/csv")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#555; font-size:0.8rem;'>
    ☀️ Solar Energy Prediction System · Random Forest ML · Built with Streamlit & Plotly
</p>
""", unsafe_allow_html=True)
