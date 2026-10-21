import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PulseGuard AI | Heart Risk Intelligence",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Premium CSS – Bold, Vibrant, Cinematic
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0a0e17;
        color: #f8fafc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0a0e17 100%);
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border-radius: 20px;
        padding: 1.75rem 1.25rem;
        border: 1px solid #334155;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        text-align: center;
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-6px);
        border-color: #38bdf8;
        box-shadow: 0 15px 50px rgba(56, 189, 248, 0.25);
    }

    /* Risk Badges – Bold & Glowing */
    .risk-badge {
        display: inline-block;
        padding: 1rem 2.5rem;
        border-radius: 60px;
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin: 1.25rem 0;
        text-transform: uppercase;
    }
    .low-risk {
        background: linear-gradient(135deg, #059669, #34d399);
        color: #fff;
        box-shadow: 0 0 40px rgba(52, 211, 153, 0.55);
    }
    .moderate-risk {
        background: linear-gradient(135deg, #d97706, #fbbf24);
        color: #111;
        box-shadow: 0 0 40px rgba(251, 191, 36, 0.55);
    }
    .high-risk {
        background: linear-gradient(135deg, #dc2626, #f87171);
        color: #fff;
        box-shadow: 0 0 40px rgba(248, 113, 113, 0.55);
    }
    .critical-risk {
        background: linear-gradient(135deg, #7f1d1d, #ef4444);
        color: #fff;
        box-shadow: 0 0 50px rgba(239, 68, 68, 0.7);
        animation: critical-pulse 1.4s infinite;
    }
    @keyframes critical-pulse {
        0%   { box-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
        50%  { box-shadow: 0 0 55px rgba(239, 68, 68, 0.9); }
        100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
    }

    /* Info / Advice Box */
    .info-box {
        background: linear-gradient(90deg, rgba(14, 165, 233, 0.12), rgba(14, 165, 233, 0.04));
        border-left: 5px solid #0ea5e9;
        padding: 1.25rem 1.5rem;
        border-radius: 0 16px 16px 0;
        margin: 1.25rem 0;
        color: #e0f2fe;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* Feature Chips */
    .feature-chip {
        display: inline-block;
        background: #1e293b;
        color: #94a3b8;
        padding: 0.4rem 1rem;
        border-radius: 30px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.25rem;
        border: 1px solid #334155;
    }

    /* Primary Button */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.9rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        width: 100%;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.45) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(14, 165, 233, 0.65) !important;
        background: linear-gradient(135deg, #0369a1, #38bdf8) !important;
    }

    /* Inputs */
    .stSlider > div > div > div {
        background: #0ea5e9 !important;
    }
    .stSelectbox > div > div {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
    }

    /* Hero Title Gradient */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #22d3ee, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        line-height: 1.1;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.25rem;
        font-weight: 500;
        margin-top: 0;
    }

    /* Section Divider */
    .section-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #38bdf8, transparent);
        margin: 2rem 0;
        border: none;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        padding: 2.5rem 0 1.5rem;
        border-top: 1px solid #1e293b;
        margin-top: 3rem;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model.pkl"
    if not model_path.exists():
        model_path = Path("/home/workdir/attachments/model.pkl")
    return joblib.load(model_path)

model = load_model()

# ─────────────────────────────────────────────
# Constants & Helpers
# ─────────────────────────────────────────────
RISK_COLORS = {
    "Low Risk": "#34d399",
    "Moderate Risk": "#fbbf24",
    "High Risk": "#f87171",
    "Critical Risk": "#ef4444",
}

RISK_ICONS = {
    "Low Risk": "✅",
    "Moderate Risk": "⚠️",
    "High Risk": "🔴",
    "Critical Risk": "🚨",
}

def get_risk_advice(risk: str) -> str:
    advice = {
        "Low Risk": "Patient status appears stable. Maintain a healthy lifestyle, regular check-ups, and avoid major risk factors (smoking, obesity, sedentary habits).",
        "Moderate Risk": "Some indicators require attention. Recommend cardiology consultation, improved diet, increased physical activity, and closer monitoring of blood pressure and glucose.",
        "High Risk": "Elevated risk level detected. Advise specialist review soon, additional diagnostic tests, and strict adherence to any prescribed treatment. Avoid heavy physical stress.",
        "Critical Risk": "Critical condition requiring immediate medical attention. Contact emergency services or go to the nearest hospital. Do not ignore severe chest pain or shortness of breath.",
    }
    return advice.get(risk, "")

def create_gauge(value: float, title: str, max_val: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 15, "color": "#94a3b8", "family": "Inter"}},
        number={"font": {"size": 32, "color": "#f8fafc", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#475569", "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": "#0f172a",
            "borderwidth": 2,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, max_val * 0.35], "color": "#064e3b"},
                {"range": [max_val * 0.35, max_val * 0.65], "color": "#78350f"},
                {"range": [max_val * 0.65, max_val], "color": "#7f1d1d"},
            ],
            "threshold": {
                "line": {"color": "#f8fafc", "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(l=25, r=25, t=50, b=25),
        font={"color": "#e2e8f0", "family": "Inter"},
    )
    return fig

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([0.8, 5.2])
with col_logo:
    st.markdown("<div style='font-size: 4rem; text-align: center; line-height: 1;'>❤️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <div class="hero-title">PulseGuard AI</div>
    <p class="hero-subtitle">Intelligent Heart Risk Assessment System · Decision Tree ML Engine</p>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-line">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR – Patient Inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Patient Profile")
    st.caption("Enter vital signs accurately for reliable prediction")

    st.markdown("#### Demographics")
    age = st.slider("Age (years)", min_value=18, max_value=95, value=55, step=1)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    smoker = st.selectbox("Smoker Status", options=["No", "Yes"])
    diabetes = st.selectbox("Diabetes History", options=["No", "Yes"])

    st.markdown("#### Vital Signs")
    heart_rate = st.slider("Heart Rate (bpm)", 40, 180, 85)
    sbp = st.slider("Systolic Blood Pressure (mmHg)", 80, 220, 130)
    oxygen = st.slider("Oxygen Saturation SpO₂ (%)", 70.0, 100.0, 96.0, 0.1)
    chest_pain = st.slider("Chest Pain Severity (0–10)", 0.0, 10.0, 2.0, 0.1)
    temperature = st.slider("Body Temperature (°C)", 35.0, 40.0, 36.8, 0.1)

    st.markdown("---")
    predict_btn = st.button("🔍  ANALYZE RISK NOW", use_container_width=True)

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("### 📋 Current Input Summary")
chips = f"""
<span class="feature-chip">Age: {age}</span>
<span class="feature-chip">Gender: {gender}</span>
<span class="feature-chip">Smoker: {smoker}</span>
<span class="feature-chip">Diabetes: {diabetes}</span>
<span class="feature-chip">HR: {heart_rate} bpm</span>
<span class="feature-chip">SBP: {sbp} mmHg</span>
<span class="feature-chip">SpO₂: {oxygen}%</span>
<span class="feature-chip">Chest Pain: {chest_pain}</span>
<span class="feature-chip">Temp: {temperature}°C</span>
"""
st.markdown(chips, unsafe_allow_html=True)

if predict_btn:
    input_df = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "smoker_status": [smoker],
        "diabetes_history": [diabetes],
        "heart_rate": [heart_rate],
        "systolic_blood_pressure": [sbp],
        "oxygen_saturation": [oxygen],
        "chest_pain_severity": [chest_pain],
        "body_temperature": [temperature],
    })

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = model.named_steps["model"].classes_
    proba_dict = dict(zip(classes, proba))

    st.markdown('<hr class="section-line">', unsafe_allow_html=True)
    st.markdown("## 🩺 Risk Assessment Result")

    risk_class = {
        "Low Risk": "low-risk",
        "Moderate Risk": "moderate-risk",
        "High Risk": "high-risk",
        "Critical Risk": "critical-risk",
    }[prediction]

    st.markdown(f"""
    <div style="text-align: center;">
        <div class="risk-badge {risk_class}">
            {RISK_ICONS[prediction]}  {prediction}
        </div>
    </div>
    """, unsafe_allow_html=True)

    confidence = proba_dict[prediction] * 100
    st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; margin-bottom: 1.75rem; font-size: 1.15rem;">
        Prediction Confidence: <strong style="color: #38bdf8; font-size: 1.4rem;">{confidence:.1f}%</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <strong>💡 Clinical Recommendation</strong><br><br>
        {get_risk_advice(prediction)}
    </div>
    """, unsafe_allow_html=True)

    # Charts Row
    col1, col2 = st.columns([1.25, 1])

    with col1:
        st.markdown("#### Probability Distribution")
        proba_df = pd.DataFrame({
            "Level": list(classes),
            "Probability %": [round(p * 100, 1) for p in proba],
            "Color": [RISK_COLORS[c] for c in classes],
        }).sort_values("Probability %", ascending=True)

        fig_bar = go.Figure(go.Bar(
            x=proba_df["Probability %"],
            y=proba_df["Level"],
            orientation="h",
            marker_color=proba_df["Color"],
            text=[f"{v}%" for v in proba_df["Probability %"]],
            textposition="outside",
            textfont=dict(color="#f8fafc", size=14, family="Inter"),
            marker_line=dict(width=0),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=10, r=50, t=20, b=20),
            xaxis=dict(
                range=[0, 110],
                gridcolor="#1e293b",
                color="#94a3b8",
                title=None,
            ),
            yaxis=dict(color="#e2e8f0", title=None),
            font=dict(color="#e2e8f0", family="Inter"),
            bargap=0.35,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("#### Key Vitals Snapshot")
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(create_gauge(heart_rate, "Heart Rate", 180, "#38bdf8"), use_container_width=True)
        with g2:
            st.plotly_chart(create_gauge(sbp, "Systolic BP", 220, "#a78bfa"), use_container_width=True)

        g3, g4 = st.columns(2)
        with g3:
            st.plotly_chart(create_gauge(oxygen, "SpO₂ %", 100, "#22d3ee"), use_container_width=True)
        with g4:
            st.plotly_chart(create_gauge(chest_pain, "Chest Pain", 10, "#f43f5e"), use_container_width=True)

    # Insights
    st.markdown('<hr class="section-line">', unsafe_allow_html=True)
    st.markdown("### 📊 What Drives the Prediction?")
    st.markdown("""
    <div class="info-box">
        The tuned <strong>Decision Tree</strong> model weighs several clinical signals:
        <ul style="margin-top: 0.75rem; margin-bottom: 0;">
            <li><strong>Oxygen saturation</strong>, <strong>systolic blood pressure</strong>, and <strong>chest pain severity</strong> are the strongest indicators of critical risk.</li>
            <li>Elevated heart rate combined with low SpO₂ significantly increases the probability of High / Critical classification.</li>
            <li>Advanced age + diabetes or smoking history raises risk even when other vitals are only moderately abnormal.</li>
            <li>Healthy ranges (SpO₂ &gt; 95%, SBP &lt; 140, low chest pain) strongly favor Low or Moderate Risk.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

else:
    # Empty / Welcome State
    st.markdown("""
    <div style="text-align: center; padding: 3.5rem 1rem 2rem; color: #64748b;">
        <div style="font-size: 4.5rem; margin-bottom: 1rem;">🩺</div>
        <h3 style="color: #94a3b8; font-weight: 600;">Enter patient data in the sidebar</h3>
        <p style="font-size: 1.1rem;">Then click <strong style="color: #38bdf8;">ANALYZE RISK NOW</strong> to generate the intelligent assessment</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ℹ️ Model Overview")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.1rem; font-weight: 800; color: #38bdf8;">22,500</div>
            <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.3rem;">Training Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.1rem; font-weight: 800; color: #34d399;">96.4%</div>
            <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.3rem;">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.7rem; font-weight: 800; color: #fbbf24; line-height: 1.3;">Decision<br>Tree</div>
            <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.3rem;">Algorithm</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.1rem; font-weight: 800; color: #a78bfa;">4 Levels</div>
            <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.3rem;">Risk Classes</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top: 2.25rem;">
        <strong>How the system works</strong><br><br>
        Input data flows through a complete scikit-learn Pipeline:
        <br>• <strong>StandardScaler</strong> for numerical features
        <br>• <strong>OneHotEncoder</strong> for categorical features (Gender, Smoker, Diabetes)
        <br>• Tuned <strong>DecisionTreeClassifier</strong> (criterion=entropy, max_depth=15, class_weight=balanced)
        <br><br>
        Output classes: <strong>Low Risk</strong> · <strong>Moderate Risk</strong> · <strong>High Risk</strong> · <strong>Critical Risk</strong>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    PulseGuard AI · Educational Heart Risk Prediction System<br>
    This application is for demonstration purposes only and is not a substitute for professional medical advice
</div>
""", unsafe_allow_html=True)