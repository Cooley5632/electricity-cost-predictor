import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Electricity Cost Predictor",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------------
# Load Models
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

reg_path = BASE_DIR / "Models" / "regression_model.pkl"
clf_path = BASE_DIR / "Models" / "classification_model.pkl"


def load_models():
    regression_model = joblib.load(reg_path)
    classification_model = joblib.load(clf_path)
    return regression_model, classification_model

regression_model, classification_model = load_models()

# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("⚙️ Building Features")

f1 = st.sidebar.slider("Site Area (sq ft)", min_value=500, max_value=5000, value=500)
f2 = st.sidebar.slider("Water Consumption (liters)", min_value=1000, max_value=11000, value=1000)
f3 = st.sidebar.slider("Resident Count", min_value=0, max_value=489, value=10)
f4 = st.sidebar.slider("Structure Type Mixed Use",min_value= 0, max_value=100, value=5)
f5 = st.sidebar.slider("Utilization Rate (%)", min_value=0, max_value=100, value=5)
f6 = st.sidebar.slider("Structure Type Industrial", min_value=0, max_value=100, value=5)
f7 = st.sidebar.slider("Structure Type Residential", min_value=0, max_value=100, value=5)

feature_names = [
    "site_area",
    "water_consumption",
    "resident_count",
    "structure_type_Mixed-Use",
    "utilization_rate",
    "structure_type_Industrial",
    "structure_type_Residential"
]

# Build DataFrame for current slider values
X = np.array([[f1, f2, f3, f4, f5, f6, f7]])
df = pd.DataFrame(X, columns=feature_names)

# Reorder BEFORE prediction
df = df[regression_model.feature_names_in_]

# Toggle for Prediction Mode

mode = st.radio(
    "Prediction Mode",
    ["Use Buttons", "Auto‑Update"],
    horizontal=True
)

# Compute predictions
if mode == "Use Buttons":
    reg_pred = None
    class_pred = None
else:
    reg_pred = regression_model.predict(df)[0]
    class_pred = classification_model.predict(df)[0]

# Update DataFrame with predicted target value
df["electricity_cost"] = reg_pred if reg_pred is not None else np.nan

weights = {"Site Area": .873392, "Water Consumption (liters)": .697782, "Resident Count": .363192, "Structure Type Mixed Use": .240326, "Utilization Rate (%)": .206496, "Structure Type Industrial": .172968, "Structure Type Residential": .110309}

electricity_cost = df[feature_names].multiply(list(weights.values()), axis=1).sum(axis=1)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("⚡ Electricity Cost Prediction Dashboard")

st.markdown(
"""
This application uses **Machine Learning models** to:

- Predict **monthly electricity cost**
- Classify **energy consumption category**
"""
)

# ---------------------------------------------------------
# Compute predictions (auto or button mode)
# ---------------------------------------------------------
if mode == "Use Buttons":
    reg_pred = None
    class_pred = None
else:
    reg_pred = regression_model.predict(df)[0]
    class_pred = classification_model.predict(df)[0]
    
df = df[regression_model.feature_names_in_]   

# ---------------------------------------------------------
# Prediction Section
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Regression Prediction")

    if mode == "Use Buttons":
        if st.button("Predict Electricity Cost"):
            reg_pred = regression_model.predict(df)[0]

    if reg_pred is not None:
        st.metric("Estimated Monthly Electricity Cost", f"${reg_pred:.2f}")
        
  
with col2:
    st.subheader("📊 Classification Prediction")

    if mode == "Use Buttons":
        if st.button("Predict Energy Category"):
            class_pred = classification_model.predict(df)[0]

# ---------------------------------------------------------
# Update DataFrame with predicted target value
# ---------------------------------------------------------
if reg_pred is not None:
    df["electricity_cost"] = reg_pred
else:
    df["electricity_cost"] = np.nan

# ---------------------------------------------------------
# Input Summary Table
# ---------------------------------------------------------
st.divider()
st.subheader("📋 Input Feature Summary")
st.dataframe(df)

# ---------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------
st.subheader("📊 Feature Importance")

try:
    importance = regression_model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(feature_names, importance)
    st.pyplot(fig)
except:
    st.info("Feature importance not available for this model.")

# ---------------------------------------------------------
# Input Visualization
# ---------------------------------------------------------
st.subheader("📈 Input Feature Visualization")

fig2, ax2 = plt.subplots()
ax2.bar(feature_names, X[0])
plt.xticks(rotation=45)
st.pyplot(fig2)

