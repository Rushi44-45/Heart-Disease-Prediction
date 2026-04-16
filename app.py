import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Model Load & Cache
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Page Configuration
st.set_page_config(page_title="Heart Health AI", layout="centered")

# App Header
st.title("❤️ Heart Failure Prediction AI")
st.write("Patient chi mahiti bhara ani clinical outcome check kara.")
st.divider()

# 2. User Friendly Input Form
with st.form("main_form"):
    st.subheader("📝 Patient Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age (Vay)", 1, 110, 55)
        sex = st.radio("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", horizontal=True)
        diabetes = st.selectbox("Diabetes?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        anaemia = st.selectbox("Anaemia (Raktakshay)?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        hbp = st.selectbox("High Blood Pressure?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        smoking = st.selectbox("Smoking Savay aahe?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        ef = st.number_input("Ejection Fraction (%)", 0, 100, 38)
        sc = st.number_input("Serum Creatinine (mg/dL)", 0.0, 15.0, 1.1)
        ss = st.number_input("Serum Sodium (mEq/L)", 100, 150, 137)
        cpk = st.number_input("CPK Enzyme (mcg/L)", 0, 10000, 500)
        platelets = st.number_input("Platelets Count", 10000.0, 1000000.0, 250000.0)
        time = st.number_input("Follow-up Period (Days)", 1, 300, 100)

    st.divider()
    submit = st.form_submit_button("Generate Prediction Result")

# 3. Prediction & Visual Feedback
if submit:
    # Feature array in correct order 
    features = np.array([[age, anaemia, cpk, diabetes, ef, hbp, platelets, sc, ss, sex, smoking, time]])
    
    # Prediction logic
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1] # Risk percentage

    st.subheader("📊 Analysis Result")
    
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk: Heart failure chi shakyata jasta aahe.")
        st.warning(f"Risk Score: {probability*100:.1f}%")
    else:
        st.success(f"✅ Low Risk: Patient surakshit distoy.")
        st.info(f"Risk Score: {probability*100:.1f}%")

    # Visual Representation (Simple Gauge Chart substitute)
    st.progress(probability)
    
st.caption("Tip: He prediction Machine Learning var adharit aahe. Doctor cha salla naki ghya.")
