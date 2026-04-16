import streamlit as st
import pickle
import numpy as np

# 1. Model Loading with Cache
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# 2. Page Configuration & Custom CSS for Design
st.set_page_config(page_title="HeartCare AI", page_icon="❤️", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: none;
    }
    div[data-testid="stForm"] {
        padding: 30px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_index=True)

# 3. UI Header
st.title("❤️ Heart Failure Analysis")
st.markdown("##### Machine Learning Prediction System")
st.write("Patient che clinical metrics khaali bhara jyamule risk level samajel.")

# 4. Input Form with Cards Layout
with st.form("prediction_form"):
    st.subheader("📋 Patient Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**General Information**")
        age = st.slider("Age (Vay)", 1, 110, 60)
        sex = st.radio("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", horizontal=True)
        smoking = st.selectbox("Smoking Habits", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        diabetes = st.selectbox("Diabetes History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        anaemia = st.selectbox("Anaemia (Blood Iron)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        hbp = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        st.info("**Lab Reports**")
        ef = st.number_input("Ejection Fraction (%)", 0, 100, 38, help="Pumping capacity of heart")
        sc = st.number_input("Serum Creatinine (mg/dL)", 0.1, 15.0, 1.1)
        ss = st.number_input("Serum Sodium (mEq/L)", 100, 150, 137)
        cpk = st.number_input("CPK Enzyme (mcg/L)", 0, 10000, 550)
        platelets = st.number_input("Platelet Count", 10000.0, 900000.0, 260000.0)
        time = st.number_input("Follow-up Period (Days)", 1, 300, 100)

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("Predict Clinical Outcome")

# 5. Result Display
if submit:
    # Feature ordering for model
    features = np.array([[age, anaemia, cpk, diabetes, ef, hbp, platelets, sc, ss, sex, smoking, time]])
    
    # Prediction
    prediction = model.predict(features)
    
    # Result Visuals
    st.divider()
    if prediction[0] == 1:
        st.error("### 🚨 Prediction: High Risk")
        st.write("Patient cha clinical report heart failure chi shakyata dakhvtoy. तातडीने डॉक्टरांशी संपर्क साधा.")
    else:
        st.success("### ✅ Prediction: Low Risk")
        st.write("Patient che metrics sadhya surakshit range madhe distoyet.")

st.caption("Disclaimer: This tool is for educational purposes only. Always consult a professional doctor.")
