import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()

st.title("Heart Failure Prediction App")
st.write("Enter the patient details below to predict the clinical outcome.")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=60)
    anaemia = st.selectbox("Anaemia (0: No, 1: Yes)", [0, 1])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", value=500)
    diabetes = st.selectbox("Diabetes (0: No, 1: Yes)", [0, 1])
    ejection_fraction = st.number_input("Ejection Fraction (%)", value=38)
    high_blood_pressure = st.selectbox("High Blood Pressure (0: No, 1: Yes)", [0, 1])

with col2:
    platelets = st.number_input("Platelets (kiloplatelets/mL)", value=260000.0)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", value=1.0)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", value=137)
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
    smoking = st.selectbox("Smoking (0: No, 1: Yes)", [0, 1])
    time = st.number_input("Follow-up Period (days)", value=100)

# Prediction Logic
if st.button("Predict"):
    # Arrange features in the exact order the model expects 
    features = np.array([[
        age, anaemia, creatinine_phosphokinase, diabetes, 
        ejection_fraction, high_blood_pressure, platelets, 
        serum_creatinine, serum_sodium, sex, smoking, time
    ]])
    
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.error("Result: High risk of heart failure event.")
    else:
        st.success("Result: Low risk of heart failure event.")

st.info("Note: This is a machine learning demonstration based on a KNN model.")
