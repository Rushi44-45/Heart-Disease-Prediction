import streamlit as st
import pickle
import numpy as np

# Optimization 1: Cache the model loading
# This prevents the app from re-reading the .pkl file on every interaction.
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

st.title("Fast Heart Failure Predictor")

# Optimization 2: Use a form to prevent the app from re-running 
# until the user is completely finished entering data.
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 1, 120, 60)
        anaemia = st.selectbox("Anaemia", [0, 1])
        cpk = st.number_input("CPK Enzyme", value=500)
        diabetes = st.selectbox("Diabetes", [0, 1])
        ef = st.number_input("Ejection Fraction %", value=38)
        hbp = st.selectbox("High Blood Pressure", [0, 1])

    with col2:
        platelets = st.number_input("Platelets", value=260000.0)
        sc = st.number_input("Serum Creatinine", value=1.0)
        ss = st.number_input("Serum Sodium", value=137)
        sex = st.selectbox("Sex (0:F, 1:M)", [0, 1])
        smoking = st.selectbox("Smoking", [0, 1])
        time = st.number_input("Follow-up Time (days)", value=100)

    submit = st.form_submit_button("Predict Now")

if submit:
    # Feature names from your model: age, anaemia, creatinine_phosphokinase, etc. 
    features = np.array([[age, anaemia, cpk, diabetes, ef, hbp, platelets, sc, ss, sex, smoking, time]])
    
    # KNN effective metric is 'euclidean' [cite: 21]
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.error("High risk of mortality.")
    else:
        st.success("Low risk of mortality.")
