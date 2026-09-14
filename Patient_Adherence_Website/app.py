import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Patient Adherence Prediction",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("patient_Adherence_prediction_final.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_error = None
except Exception as e:
    model = None
    model_error = e

st.title("🏥 Patient Adherence Prediction")
st.write(
    "Enter the patient's information below to estimate medication adherence "
    "using the trained machine-learning model."
)

st.info(
    "How to use: enter values that match the training dataset. "
    "For Income and Dosage, use the same units/scales used when the model was trained."
)

if model_error:
    st.error("The trained model could not be loaded.")
    st.code(str(model_error))
    st.warning(
        "This model was saved with scikit-learn 1.6.1. "
        "Make sure the environment uses the versions listed in requirements.txt."
    )
    st.stop()

st.divider()

st.header("1. Patient Demographics")
st.caption("Enter basic information about the patient.")

c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input(
        "Age",
        min_value=0, max_value=120, value=40,
        help="Enter the patient's age in years."
    )
with c2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"],
        help="Select the patient's gender category used in the training data."
    )
with c3:
    education = st.selectbox(
        "Education Level",
        ["High School", "Graduate", "Postgraduate"],
        help="Select the patient's highest education level."
    )

st.header("2. Treatment Information")
st.caption("Enter information about the patient's medication and previous adherence.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    medication = st.selectbox(
        "Medication Type",
        ["TypeA", "TypeB", "TypeC"],
        help="Select the medication type. Use the category names from the dataset."
    )
with c2:
    dosage = st.number_input(
        "Dosage (mg)",
        min_value=0.0, value=100.0, step=1.0,
        help="Enter the medication dosage in mg, using the same scale as the training dataset."
    )
with c3:
    previous_adherence = st.selectbox(
        "Previous Adherence",
        ["Yes", "No"],
        help="Select Yes if the patient had previous adherence (1 in the dataset), otherwise No (0)."
    )
with c4:
    insurance = st.selectbox(
        "Insurance Coverage",
        ["Yes", "No"],
        help="Select Yes if the patient has insurance coverage (1 in the dataset), otherwise No (0)."
    )

st.header("3. Clinical & Support Factors")
st.caption("Select the patient's current clinical, healthcare-access, mental-health and social-support levels.")

c1, c2, c3 = st.columns(3)
with c1:
    social_support = st.selectbox(
        "Social Support Level",
        ["Low", "Medium", "High"],
        help="Choose the patient's level of social support."
    )
with c2:
    condition = st.selectbox(
        "Condition Severity",
        ["Mild", "Moderate", "Severe"],
        help="Choose the severity of the patient's condition. The model uses the logical order Mild < Moderate < Severe."
    )
with c3:
    healthcare = st.selectbox(
        "Healthcare Access",
        ["Poor", "Average", "Good"],
        help="Choose the patient's level of access to healthcare."
    )

c1, c2, c3 = st.columns(3)
with c1:
    mental_health = st.selectbox(
        "Mental Health Status",
        ["Poor", "Moderate", "Good"],
        help="Choose the patient's current mental-health status."
    )
with c2:
    comorbidities = st.number_input(
        "Number of Comorbidities",
        min_value=0, max_value=50, value=0, step=1,
        help="Enter the number of additional medical conditions the patient has."
    )
with c3:
    income = st.number_input(
        "Income",
        min_value=0.0, value=50000.0, step=1000.0,
        help="Enter income using the same unit and scale as the training dataset."
    )

st.divider()

if st.button("🔍 Predict Adherence", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Medication_Type": medication,
        "Dosage_mg": dosage,
        "Previous_Adherence": 1 if previous_adherence == "Yes" else 0,
        "Education_Level": education,
        "Income": income,
        "Social_Support_Level": social_support,
        "Condition_Severity": condition,
        "Comorbidities_Count": comorbidities,
        "Healthcare_Access": healthcare,
        "Mental_Health_Status": mental_health,
        "Insurance_Coverage": 1 if insurance == "Yes" else 0
    }])

    try:
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        probability = float(probabilities[prediction])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ Likely Adherent")
        else:
            st.warning("⚠️ Likely Non-Adherent")

        st.metric(
            "Predicted-Class Probability",
            f"{probability * 100:.1f}%"
        )
        st.progress(probability)

        with st.expander("View submitted patient information"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))

st.divider()
st.caption(
    "⚠️ Project demonstration only. This model is not a medical diagnosis or a substitute "
    "for advice from a qualified healthcare professional."
)
