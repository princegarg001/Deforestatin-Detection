import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from streamlit_lottie import st_lottie
import json

# Load model and scaler
model = joblib.load("best_fire_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set page config
st.set_page_config(page_title="🔥 Fire Type Classifier", layout="centered")

# --- Optional: Load Lottie Animation ---
def load_lottie(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None

# Load Lottie animation
lottie_fire = load_lottie("fire_animation.json")

# Page Header
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0.2em;'>🔥 Fire Detection and Classification</h1>
    <p style='text-align: center; font-size: 17px; color: #ccc;'>
        Powered by MODIS satellite data and machine learning.<br>
        Identify if it's a 🌿 vegetation fire, 🌊 offshore event, or a static source using remote sensing parameters.
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("<hr style='margin-top: 1em; margin-bottom: 1em;'>", unsafe_allow_html=True)


# Show Lottie Animation if available
if lottie_fire:
    st_lottie(lottie_fire, speed=1, height=300, key="fire_anim")

# --- Input Fields Layout ---
st.header("  🛰️Enter Satellite Fire Parameters")
col1, col2 = st.columns(2)

with col1:
    brightness = st.number_input("🔆 Brightness", value=300.0)
    bright_t31 = st.number_input("🌡️ Brightness T31", value=290.0)
    frp = st.number_input("🔥 Fire Radiative Power (FRP)", value=15.0)

with col2:
    scan = st.number_input("📡 Scan", value=1.0)
    track = st.number_input("🎯 Track", value=1.0)
    confidence = st.selectbox("📈 Confidence Level", ["low", "nominal", "high"])

# --- Map confidence to numeric ---
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# --- Prediction ---
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
scaled_input = scaler.transform(input_data)

fire_types = {
    0: "Vegetation Fire",
    2: "Other Static Land Source",
    3: "Offshore Fire"
}

if st.button("🔍 Predict Fire Type"):
    prediction = model.predict(scaled_input)[0]
    result = fire_types.get(prediction, "Unknown")
    st.success(f"**Predicted Fire Type:** {result}")

    # Show prediction confidence if available
    try:
        prediction_proba = model.predict_proba(scaled_input)[0]
        st.info(f"Prediction Confidence: {np.max(prediction_proba) * 100:.2f}%")
    except:
        st.warning("Confidence score not available for this model.")

# --- 3D Visualization ---
st.markdown("---")
st.header("📊 3D Visualization of Fire Characteristics")

# Sample data for visualization
sample_data = pd.DataFrame({
    "brightness": [300, 310, 290, 305, brightness],
    "bright_t31": [290, 295, 285, 288, bright_t31],
    "frp": [10, 20, 15, 12, frp],
    "fire_type": ["Vegetation", "Offshore", "Other", "Vegetation", "Input"]
})

fig = px.scatter_3d(
    sample_data,
    x='brightness',
    y='bright_t31',
    z='frp',
    color='fire_type',
    title="3D Fire Feature Distribution",
    symbol='fire_type',
    opacity=0.8
)

st.plotly_chart(fig, use_container_width=True)
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px;">
        <p style="font-size: 16px;">Built with ❤️ by <strong>Prince Garg</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)

