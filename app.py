from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

logo = Image.open("logo.png")
st.image(logo, width=150)

# Load model and encoders
model = joblib.load("best_salary_model.pkl")
le_dept = joblib.load("label_encoder_dept.pkl")
le_country = joblib.load("label_encoder_country.pkl")

# Title
st.title("💰 Employee Salary Prediction App")
st.markdown("Predict annual salary based on department, country, experience, and more.")

# Inputs
years = st.number_input("📅 Years at Company", min_value=0, max_value=40, step=1)
job_rate = st.slider("⭐ Job Rate (1 to 5)", min_value=1.0, max_value=5.0, step=0.5)
sick_leaves = st.number_input("🤒 Sick Leaves", min_value=0, step=1)
unpaid_leaves = st.number_input("🚫 Unpaid Leaves", min_value=0, step=1)
overtime_hours = st.number_input("⏱️ Overtime Hours", min_value=0, step=1)

# Get department and country options from encoders
dept_options = list(le_dept.classes_)
country_options = list(le_country.classes_)

department = st.selectbox("🏢 Department", dept_options)
country = st.selectbox("🌍 Country", country_options)

# Encode categorical inputs
department_enc = le_dept.transform([department])[0]
country_enc = le_country.transform([country])[0]

# Prepare feature array
input_data = pd.DataFrame([[
    years, job_rate, department_enc, country_enc, sick_leaves, unpaid_leaves, overtime_hours
]], columns=['Years', 'Job Rate', 'Department_enc', 'Country_enc', 'Sick Leaves', 'Unpaid Leaves', 'Overtime Hours'])

# Predict
if st.button("🚀 Predict Salary"):
    prediction = model.predict(input_data)
    st.success(f"💸 Predicted Annual Salary: ${prediction[0]:,.2f}")
else:
    st.subheader("📋 Input Summary")
    st.dataframe(input_data)
    st.info("🖱️ Enter values and press predict.")




# Feature importance (only works for tree-based models like RF/XGB)
try:
    importances = model.feature_importances_
    features = input_data.columns
    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=features, ax=ax, palette='viridis')
    ax.set_title("Feature Importance")
    st.pyplot(fig)
except:
    st.warning("Feature importance not available for this model.")

