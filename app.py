import streamlit as st
import pandas as pd
import numpy as np
from model_utils import train_models
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

# Title
st.title("🩺 Diabetes Risk Predictor")
st.markdown("### AI-powered health risk analysis with explanation")

# Load data
df = pd.read_csv("diabetes.csv")

# Train model
model, X = train_models(df)

# =========================
# SAFE RANGES
# =========================
st.subheader("📌 Healthy Reference Ranges")

safe_ranges = {
    "Glucose": (70, 140),
    "BloodPressure": (60, 120),
    "BMI": (18.5, 24.9),
    "Age": (0, 45)
}

st.info("""
🟢 Glucose: 70 - 140  
🟢 Blood Pressure: 60 - 120  
🟢 BMI: 18.5 - 24.9  
🟢 Age: Below 45  
""")

# =========================
# INPUT SECTION
# =========================
st.subheader("🧾 Enter Your Health Details")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin Level", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 1, 100)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Check Risk"):

    input_data = np.array([[pregnancies, glucose, bp, skin,
                            insulin, bmi, dpf, age]])

    prob = model.predict_proba(input_data)[0]

    low_risk = prob[0] * 100
    high_risk = prob[1] * 100

    st.subheader("📊 Risk Analysis")

    # =========================
    # 🎯 COLOR RISK METER
    # =========================
    st.markdown("### 🎯 Risk Meter")

    if high_risk < 30:
        color = "green"
        label = "LOW RISK"
    elif high_risk < 60:
        color = "orange"
        label = "MODERATE RISK"
    else:
        color = "red"
        label = "HIGH RISK"

    st.markdown(
        f"""
        <div style="
            background-color:{color};
            padding:20px;
            border-radius:10px;
            text-align:center;
            color:white;
            font-size:24px;">
            {label} : {high_risk:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(f"🟢 Low Risk: {low_risk:.2f}%")
    st.error(f"🔴 High Risk: {high_risk:.2f}%")

    # =========================
    # 🧠 EXPLANATION
    # =========================
    st.subheader("🧠 Why is your risk this level?")

    reasons = []

    if glucose > 140:
        reasons.append("High Glucose level increases diabetes risk")

    if bp > 120:
        reasons.append("High Blood Pressure can contribute to risk")

    if bmi > 25:
        reasons.append("High BMI indicates overweight condition")

    if age > 45:
        reasons.append("Age above 45 increases risk")

    if insulin > 200:
        reasons.append("Abnormal insulin levels detected")

    if len(reasons) == 0:
        st.success("✅ All your values are within safe range. Good health condition!")
    else:
        for r in reasons:
            st.warning("⚠️ " + r)

    # =========================
    # FINAL INTERPRETATION
    # =========================
    st.subheader("📢 Final Advice")

    if high_risk < 30:
        st.success("✅ You are in a SAFE zone. Maintain healthy lifestyle.")
    elif high_risk < 60:
        st.warning("⚠️ Moderate risk. Improve diet and exercise regularly.")
    else:
        st.error("🚨 High risk. Please consult a doctor soon.")