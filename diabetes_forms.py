import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the trained model and feature columns
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Title
st.title("🩺 Diabetes Diagnosis App")
st.markdown("Enter patient information below to predict whether they are diabetic.")

# --- Input Form Sections ---
st.markdown("### 🧍 Patient Information")

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    help="Number of times the patient has been pregnant"
)

age = st.number_input(
    "Age (years)",
    min_value=0,
    help="Age of the patient in years"
)

st.markdown("### 🧪 Lab Measurements")

glucose = st.number_input(
    "Glucose Level",
    min_value=0.0,
    help="Plasma glucose concentration 2 hours after oral glucose tolerance test"
)

blood_pressure = st.number_input(
    "Blood Pressure (Diastolic)",
    min_value=0.0,
    help="Diastolic blood pressure in mm Hg"
)

skin_thickness = st.number_input(
    "Skin Thickness (Triceps)",
    min_value=0.0,
    help="Triceps skin fold thickness in mm"
)

insulin = st.number_input(
    "Insulin (2-Hour Serum)",
    min_value=0.0,
    help="2-hour serum insulin level (mu U/ml)"
)

bmi = st.number_input(
    "BMI (Body Mass Index)",
    min_value=0.0,
    help="BMI = weight in kg / (height in m)^2"
)

st.markdown("### 🧬 Genetic Risk")

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    help="Score based on family history of diabetes"
)

# Collect all inputs into a list matching feature order
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]])

input_df = pd.DataFrame(user_input, columns=feature_columns)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.markdown("### 🔎 Prediction Result")

    if prediction == 1:
        st.error(f"🔴 Likely Diabetic (Confidence: {round(prob[1]*100, 2)}%)")
    else:
        st.success(f"🟢 Not Diabetic (Confidence: {round(prob[0]*100, 2)}%)")
