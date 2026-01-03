import sys
import os
import time
import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from explainability.explainer import explain_prediction
from utils.recommender import recommend
from genai.insight_engine import generate_batch_insight

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="BatchMind AI", layout="wide")

# ---------------- GLOBAL STYLES ----------------
st.markdown(
    """
    <style>
        body {
            background-color: #f4f6f9;
        }
        .card {
            background-color: white;
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            margin-bottom: 15px;
            text-align: center;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
        }
        .metric-label {
            font-size: 14px;
            color: #7f8c8d;
        }
        .low { color: #27ae60; }
        .medium { color: #f39c12; }
        .high { color: #e74c3c; }
        .alert-box {
            background-color: #fff1f0;
            color: #c0392b;
            padding: 16px;
            border-radius: 12px;
            font-weight: 600;
            margin-bottom: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }
        .section-title {
            font-size: 22px;
            font-weight: 600;
            margin-top: 25px;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/deviation_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ---------------- TITLE ----------------
st.markdown(
    "<div class='section-title'>BatchMind AI – Explainable, Predictive & Generative Intelligence for Manufacturing</div>",
    unsafe_allow_html=True
)

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CONSTANTS ----------------
BATCH_VALUE = 500000
LOSS_RATE = 0.20
ALERT_THRESHOLD = 50000

# ---------------- SIDEBAR ----------------
st.sidebar.header("Batch Control Mode")
mode = st.sidebar.radio("Select Mode", ["Manual Mode", "Live Mode"])

st.sidebar.header("Simulate Batch Parameters")

if mode == "Manual Mode":
    temperature = st.sidebar.slider("Temperature", 40, 100, 70)
    pressure = st.sidebar.slider("Pressure", 20, 50, 30)
    process_duration = st.sidebar.slider("Process Duration", 30, 120, 60)
    material_quality = st.sidebar.slider("Material Quality", 0.6, 1.0, 0.9)
    machine_load = st.sidebar.slider("Machine Load", 30, 100, 60)
else:
    temperature = int(np.random.normal(70, 5))
    pressure = int(np.random.normal(30, 3))
    process_duration = int(np.random.normal(60, 10))
    material_quality = round(np.random.uniform(0.75, 0.95), 2)
    machine_load = int(np.random.normal(60, 8))

input_df = pd.DataFrame([{
    "temperature": temperature,
    "pressure": pressure,
    "process_duration": process_duration,
    "material_quality": material_quality,
    "machine_load": machine_load
}])

batch_input = input_df.iloc[0].to_dict()

run_simulation = st.button("Run Batch Simulation") or mode == "Live Mode"

# ---------------- MAIN LOGIC ----------------
if run_simulation:
    with st.spinner("Analyzing batch"):
        time.sleep(1)

    risk_score = model.predict_proba(scaler.transform(input_df))[0][1]
    expected_loss = risk_score * BATCH_VALUE * LOSS_RATE

    if risk_score < 0.30:
        risk_level, risk_class = "LOW", "low"
    elif risk_score < 0.60:
        risk_level, risk_class = "MEDIUM", "medium"
    else:
        risk_level, risk_class = "HIGH", "high"

    if expected_loss < 25000:
        severity, severity_class = "MONITOR", "low"
    elif expected_loss < 60000:
        severity, severity_class = "ACT SOON", "medium"
    else:
        severity, severity_class = "IMMEDIATE ACTION", "high"

    if expected_loss >= ALERT_THRESHOLD:
        st.markdown(
            f"""
            <div class="alert-box">
                ALERT: Expected financial loss exceeds ₹{ALERT_THRESHOLD:,}. Immediate attention required.
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- METRICS ----------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"<div class='card'><div class='metric-label'>Risk Score</div><div class='metric-value'>{risk_score:.2f}</div></div>",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"<div class='card'><div class='metric-label'>Risk Level</div><div class='metric-value {risk_class}'>{risk_level}</div></div>",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"<div class='card'><div class='metric-label'>Estimated Loss</div><div class='metric-value'>₹{int(expected_loss):,}</div></div>",
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"<div class='card'><div class='metric-label'>Severity</div><div class='metric-value {severity_class}'>{severity}</div></div>",
            unsafe_allow_html=True
        )

    # ---------------- EXPLAINABILITY ----------------
    explanations, importance_df, top_features = explain_prediction(model, input_df)
    recommendations = recommend(top_features)

    st.markdown("<div class='section-title'>Insights & Recommendations</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Why is this batch risky")
        for e in explanations:
            st.write("•", e)

    with col2:
        st.subheader("Recommended Actions")
        for r in recommendations:
            st.write("•", r)

    # ---------------- GENAI INSIGHTS ----------------
    st.markdown("<div class='section-title'>AI Generated Manufacturing Insights</div>", unsafe_allow_html=True)

    ai_insight = generate_batch_insight(
        prediction_label=risk_level,
        prediction_score=risk_score,
        top_features=[(f, "high") for f in top_features],
        batch_data=batch_input
    )

    st.markdown(f"<div class='card'>{ai_insight}</div>", unsafe_allow_html=True)

    # ---------------- VISUALS ----------------
    st.markdown("<div class='section-title'>Risk Analysis</div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)

    with g1:
        fig1, ax1 = plt.subplots(figsize=(3.2, 2.2), dpi=120)
        ax1.barh(importance_df["feature"], importance_df["importance"])
        ax1.invert_yaxis()
        st.pyplot(fig1)

    with g2:
        trend = np.clip(np.random.normal(risk_score, 0.05, 10), 0, 1)
        fig2, ax2 = plt.subplots(figsize=(3.2, 2.2), dpi=120)
        ax2.plot(trend, marker="o")
        ax2.set_ylim(0, 1)
        st.pyplot(fig2)

    # ---------------- HISTORY ----------------
    st.markdown("<div class='section-title'>Recent Batch History</div>", unsafe_allow_html=True)

    st.session_state.history.append({
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "risk": round(risk_score, 2),
        "level": risk_level,
        "severity": severity,
        "loss": int(expected_loss)
    })

    st.session_state.history = st.session_state.history[-5:]
    st.table(pd.DataFrame(st.session_state.history))

    if mode == "Live Mode":
        time.sleep(3)
        st.rerun()
