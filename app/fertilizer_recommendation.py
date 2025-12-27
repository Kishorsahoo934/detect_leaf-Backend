import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import os
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "fertilizer_recommendation_model_latest.joblib"
METADATA_PATH = MODELS_DIR / "fertilizer_model_metadata_latest.json"

# ------------------------------------------------------------------
# Static data
# ------------------------------------------------------------------
SOIL_TYPES = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
CROP_TYPES = [
    'Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy',
    'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts'
]

FERTILIZER_MAP = {
    0: '14-35-14',
    1: '28-28',
    2: 'DAP',
    3: 'MOP',
    4: 'Potash',
    5: 'SSP',
    6: 'Urea'
}

# ------------------------------------------------------------------
# Model Loader
# ------------------------------------------------------------------

@st.cache_resource
def load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata

# ------------------------------------------------------------------
# Page Function (IMPORTABLE)
# ------------------------------------------------------------------

def fertilizer_recommendation_page():

    st.set_page_config(
        page_title="Fertilizer Recommendation System",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🌱 Fertilizer Recommendation System")
    st.markdown("Enter soil and crop details to get fertilizer recommendations.")

    # Load model
    try:
        model_pipeline, metadata = load_model_and_metadata()
        FEATURE_ORDER = metadata['feature_info']['feature_columns']
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    # -------------------- FORM --------------------

    with st.form("recommendation_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            temp = st.number_input("Temperature (°C)", 1.0, 60.0, 25.0)
            humidity = st.number_input("Humidity (%)", 1.0, 100.0, 60.0)
            moisture = st.number_input("Moisture (%)", 1.0, 100.0, 30.0)

        with col2:
            nitrogen = st.number_input("Nitrogen (N)", 0, 200, 80)
            phosphorous = st.number_input("Phosphorus (P)", 0, 200, 50)
            potassium = st.number_input("Potassium (K)", 0, 200, 40)
            ph = st.number_input("pH Value", 3.0, 10.0, 6.5)

        with col3:
            soil = st.selectbox("Soil Type", SOIL_TYPES)
            crop = st.selectbox("Crop Type", CROP_TYPES)

        submit = st.form_submit_button("Get Recommendation")

    # -------------------- PREDICTION --------------------

    if submit:
        input_data = {
            'Temparature': [temp],
            'Moisture': [moisture],
            'Soil Type': [SOIL_TYPES.index(soil)],
            'Crop Type': [CROP_TYPES.index(crop)],
            'Nitrogen': [nitrogen],
            'Phosphorous': [phosphorous],
            'Potassium': [potassium],
            'pH': [ph],
            'Humidity ': [humidity]
        }

        df = pd.DataFrame(input_data)[FEATURE_ORDER]

        try:
            prediction = model_pipeline.predict(df)[0]
            fertilizer = FERTILIZER_MAP.get(prediction, "Unknown")

            st.success(f"Recommended Fertilizer: **{fertilizer}**")
            st.balloons()

        except Exception as e:
            st.error(f"Prediction error: {e}")
