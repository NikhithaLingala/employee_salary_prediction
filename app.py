import streamlit as st
import pandas as pd
import joblib

# =========================
# 📦 Load Model (XGBoost)
# =========================
model = joblib.load("salary_prediction.pkl")

st.set_page_config(page_title="Salary Predictor", layout="centered")

# =========================
# 🎨 UI
# =========================
st.title("💼  Salary Prediction App")
st.markdown("Predict salary using Machine Learning (XGBoost Model)")

st.divider()

# =========================
# 🎯 Inputs
# =========================

col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox("Job Title", [
        "AI Engineer", "Data Analyst", "Business Analyst", "Software Engineer"
    ])

    experience_years = st.slider("Experience (Years)", 0, 20, 2)

    education_level = st.selectbox("Education Level", [
        "Bachelor", "Master", "PhD", "High School", "Diploma"
    ])

    skills_count = st.slider("Skills Count", 1, 20, 5)

with col2:
    industry = st.selectbox("Industry", [
        "Finance", "Consulting", "Media", "Manufacturing",
        "Technology", "Government", "Healthcare",
        "Education", "Telecom", "Retail"
    ])

    company_size = st.selectbox("Company Size", [
        "Small", "Medium", "Large", "Enterprise", "Startup"
    ])

    location = st.selectbox("Location", [
        "Australia", "Canada", "Sweden", "Remote", "Singapore",
        "USA", "UK", "India", "Netherlands", "Germany"
    ])

    remote_work = st.selectbox("Remote Work", [
        "No", "Hybrid", "Yes"
    ])

    certifications = st.slider("Certifications", 0, 5, 1)

# =========================
# 🔧 Feature Engineering
# =========================

def experience_level(x):
    if x <= 2:
        return 'Fresher'
    elif x <= 7:
        return 'Mid'
    elif x <= 15:
        return 'Senior'
    else:
        return 'Expert'

exp_level = experience_level(experience_years)

skill_density = skills_count / (experience_years + 1)
strength_score = experience_years * skills_count * certifications

# =========================
# 📦 Input Data
# =========================

input_data = pd.DataFrame({
    'job_title': [job_title],
    'experience_years': [experience_years],
    'education_level': [education_level],
    'skills_count': [skills_count],
    'industry': [industry],
    'company_size': [company_size],
    'location': [location],
    'remote_work': [remote_work],
    'certifications': [certifications],
    'experience_level': [exp_level],
    'skill_density': [skill_density],
    'strength_score': [strength_score]
})

# =========================
# 🚀 Prediction
# =========================

if st.button("Predict Salary 🚀"):
    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Salary: ₹ {int(prediction):,}")

    st.info("Prediction based on optimized XGBoost model (R² ≈ 0.98)")