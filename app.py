import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

# Custom CSS styles and fonts
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Roboto&display=swap');
    body, .stApp {
        background-color: #f9f9f9;
        color: #1e1e1e;
        font-family: 'Roboto', sans-serif;
    }

    /* Title */
    .title {
        font-family: 'Poppins', sans-serif;
        color: #F67011;
        font-weight: 700;
        font-size: 40px;
        margin-bottom: 1rem;
    }

    /* Section headings & labels */
    .stSelectbox label, .stNumberInput label, .stSidebar label {
        font-weight: 600;
        font-size: 16px;
        color: #1e1e1e;
    }

    /* Input fields */
    input, select {
        font-size: 16px !important;
    }

    /* Button styling */
    div.stButton > button {
        background-color: #F67011;
        color: white;
        font-weight: 700;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 10px;
        border: none;
        margin-top: 10px;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #873800;
    }

    /* Sidebar style */
    .css-1d391kg {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="title">❤️ Njechelani Heart Disease Predictor</h1>', unsafe_allow_html=True)

# Input form layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, step=1)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    trtbps = st.number_input("Resting Blood Pressure (mm Hg)", step=1)
    chol = st.number_input("Cholesterol (mg/dl)", step=1)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

with col2:
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"])
    thalachh = st.number_input("Max Heart Rate", step=1)
    exng = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("Oldpeak (ST depression)", format="%.2f", step=0.1)
    slp = st.selectbox("Slope of ST", ["Up", "Flat", "Down"])

# Categorical value conversion
sex = 1 if sex == "Male" else 0
cp = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)
fbs = 1 if fbs == "Yes" else 0
restecg = ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"].index(restecg)
exng = 1 if exng == "Yes" else 0
slp = ["Up", "Flat", "Down"].index(slp)

# Sidebar
with st.sidebar:
    st.markdown("## 🩺 Get Prediction")
    if st.button("Predict"):
        input_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg,
                                thalachh, exng, oldpeak, slp]])
        input_scaled = scaler.transform(input_data)
        result = model.predict(input_scaled)

        if result[0] == 1:
            st.error("⚠️ Likely to have Heart Disease please visit doctor for more treatment")
        else:
            st.success("✅congratulation Unlikely to have Heart Disease ")

    st.markdown("---")
    st.caption("Created with mussa njechelani❤️ using Streamlit")

