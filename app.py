import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Electricity Cost Predictor",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("⚙️ Input Features")

site_area = st.sidebar.slider("Site Area (sq ft)", 500, 5000, 500)
water_consumption = st.sidebar.slider("Water Consumption (liters)", 1000, 11000, 1000)
resident_count = st.sidebar.slider("Resident Count", 0, 489, 10)
mixed_use = st.sidebar.slider("Structure Type Mixed Use", 0, 100, 5)
utilization = st.sidebar.slider("Utilization Rate (%)", 0, 100, 5)
industrial = st.sidebar.slider("Structure Type Industrial", 0, 100, 5)
residential = st.sidebar.slider("Structure Type Residential", 0, 100, 5)

feature_names = [
    "site_area",
    "water_consumption",
    "resident_count",
    "structure_type_Mixed-Use",
    "utilization_rate",
    "structure_type_Industrial",
    "structure_type_Residential"
]

X = np.array([[site_area, water_consumption, resident_count,
               mixed_use, utilization, industrial, residential]])

df = pd.DataFrame(X, columns=feature_names)

# ---------------------------------------------------------
# Define Weights
# ---------------------------------------------------------
weights = {
    "site_area": 0.8734,
    "water_consumption": 0.6977,
    "resident_count": 0.3632,
    "structure_type_Mixed-Use": 0.2403,
    "utilization_rate": 0.2064,
    "structure_type_Industrial": 0.1729,
    "structure_type_Residential": 0.1103
}

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("⚡ Electricity Cost Prediction Dashboard")

st.markdown("""
This dashboard calculates **Electricity Cost** based on key input features.
""")

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "classification" not in st.session_state:
    st.session_state.classification = None

if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

# ---------------------------------------------------------
# Reset prediction + classification when sliders change
# ---------------------------------------------------------
current_inputs = tuple(X[0])

if st.session_state.last_inputs != current_inputs:
    st.session_state.prediction = None
    st.session_state.classification = None

st.session_state.last_inputs = current_inputs

# ---------------------------------------------------------
# Prediction Button
# ---------------------------------------------------------
st.divider()
st.subheader("🔄 Update Prediction")

if st.button("Electricity Cost"):
    weighted_sum = sum(df[col].iloc[0] * weights[col] for col in feature_names)
    st.session_state.prediction = weighted_sum

prediction = st.session_state.prediction

# ---------------------------------------------------------
# Classification Button
# ---------------------------------------------------------
if st.button("Classify Electricity Cost"):
    if prediction is None:
        st.session_state.classification = "— (calculate cost first)"
    else:
        if prediction < 1955:
            st.session_state.classification = "Low"
        elif 1955 <= prediction < 3632:
            st.session_state.classification = "Medium"
        else:
            st.session_state.classification = "High"

classification = st.session_state.classification

# ---------------------------------------------------------
# Prediction Display
# ---------------------------------------------------------
st.subheader("📈 Electricity Cost")

if prediction is not None:
    st.metric("Electricity Cost", f"{prediction:.2f}")
else:
    st.metric("Electricity Cost", "—")

# ---------------------------------------------------------
# Classification Display
# ---------------------------------------------------------
st.subheader("📊 Cost Category")
st.text_input("Category", value=classification if classification else "—")

# ---------------------------------------------------------
# Weighted Feature Bar Chart
# ---------------------------------------------------------
st.divider()
st.subheader("📊 Feature Contribution (Weighted Values)")

weighted_values = {f: df[f].iloc[0] * weights[f] for f in feature_names}
w_df = pd.DataFrame({
    "Feature": list(weighted_values.keys()),
    "Weighted Value": list(weighted_values.values())
})

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.bar(w_df["Feature"], w_df["Weighted Value"], color="skyblue")
ax1.set_title("Weighted Feature Contributions")
ax1.set_ylabel("Weighted Value")
ax1.set_xticklabels(w_df["Feature"], rotation=45, ha="right")

st.pyplot(fig1)

# ---------------------------------------------------------
# Classification Chart
# ---------------------------------------------------------
st.subheader("📊 Electricity Cost Classification Chart")

bins = ["Low", "Medium", "High"]
values = [1 if classification == b else 0 for b in bins]

fig2, ax2 = plt.subplots(figsize=(6, 3))
bars = ax2.bar(bins, values, color=["green" if b == classification else "gray" for b in bins])
ax2.set_ylim(0, 1)
ax2.set_title("Cost Category Highlight")

st.pyplot(fig2)

# ---------------------------------------------------------
# Input Summary
# ---------------------------------------------------------
st.divider()
st.subheader("📋 Input Features")

display_df = df.copy()
display_df["electricity_cost"] = prediction if prediction is not None else "—"
display_df["category"] = classification if classification is not None else "—"

st.dataframe(display_df)

