import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# ======================================================
# Custom Transformer (Required for loading pickle model)
# ======================================================

class outliercapper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)

        self.lower = q1 - 1.5 * (q3 - q1)
        self.upper = q3 + 1.5 * (q3 - q1)

        return self

    def transform(self, X):
        return np.clip(X, self.lower, self.upper)

    def get_feature_names_out(self, input_features=None):
        return input_features


# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="Patient Adherence Prediction",
    page_icon="🏥",
    layout="wide"
)

# ======================================================
# Load Model
# ======================================================

with open("patient_Adherence_prediction.pkl", "rb") as file:
    model = pickle.load(file)

# ======================================================
# Title
# ======================================================

st.title("🏥 Patient Adherence Prediction System")

st.markdown("""
This application predicts whether a patient is likely to **adhere** to the prescribed treatment.

Fill in all patient details below and click **Predict**.
""")

st.divider()

# ======================================================
# Input Form
# ======================================================

with st.form("patient_form"):

    col1, col2 = st.columns(2)

    # ===========================
    # Left Column
    # ===========================

    with col1:

        st.subheader("👤 Patient Information")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

        education = st.selectbox(
            "Education Level",
            ["High School", "Graduate", "Postgraduate"]
        )

        income = st.number_input(
            "Income",
            min_value=0,
            value=50000
        )

        social = st.selectbox(
            "Social Support Level",
            ["Low", "Medium", "High"]
        )

        insurance = st.selectbox(
            "Insurance Coverage",
            [0, 1]
        )

    # ===========================
    # Right Column
    # ===========================

    with col2:

        st.subheader("💊 Medical Information")

        medication = st.selectbox(
            "Medication Type",
            ["TypeA", "TypeB", "TypeC"]
        )

        dosage = st.number_input(
            "Dosage (mg)",
            min_value=0.0,
            value=100.0
        )

        previous = st.slider(
            "Previous Adherence (%)",
            0,
            100,
            80
        )

        severity = st.selectbox(
            "Condition Severity",
            ["Mild", "Moderate", "Severe"]
        )

        healthcare = st.selectbox(
            "Healthcare Access",
            ["Poor", "Average", "Good"]
        )

        mental = st.selectbox(
            "Mental Health Status",
            ["Poor", "Moderate", "Good"]
        )

        comorbidity = st.number_input(
            "Comorbidities Count",
            min_value=0,
            value=1
        )

    submitted = st.form_submit_button(
        "🔍 Predict Adherence",
        use_container_width=True
    )

# ======================================================
# Prediction
# ======================================================

if submitted:

    input_data = pd.DataFrame({

        "Age": [age],
        "Gender": [gender],
        "Medication_Type": [medication],
        "Dosage_mg": [dosage],
        "Previous_Adherence": [previous],
        "Education_Level": [education],
        "Income": [income],
        "Social_Support_Level": [social],
        "Condition_Severity": [severity],
        "Comorbidities_Count": [comorbidity],
        "Healthcare_Access": [healthcare],
        "Mental_Health_Status": [mental],
        "Insurance_Coverage": [insurance]

    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success("✅ Patient is likely to be Adherent")

    else:

        st.error("❌ Patient is likely to be Non-Adherent")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Adherent Probability",
            value=f"{probability[1]*100:.2f}%"
        )

    with col2:
        st.metric(
            label="Non-Adherent Probability",
            value=f"{probability[0]*100:.2f}%"
        )

    st.divider()

    st.subheader("Patient Details")

    st.dataframe(input_data, use_container_width=True)