import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model and features
with open("models/xgb_churn.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("Customer Churn Predictor")
st.markdown("Enter customer details below to predict if they will churn.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 0, 120, 65)
    total_charges = st.number_input("Total Charges ($)", 
                                     value=float(tenure * monthly_charges))
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male"])

with col2:
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

st.divider()

# Predict button
if st.button("🔍 Predict Churn", use_container_width=True):

    # Build input
    input_dict = {
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender": 1 if gender == "Male" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"

    # Show result
    st.subheader("Prediction Result")

    if risk == "High":
        st.error(f"🔴 HIGH RISK — {prob:.1%} chance of churning")
        st.markdown("**Action:** Contact this customer immediately with a retention offer.")
    elif risk == "Medium":
        st.warning(f"🟡 MEDIUM RISK — {prob:.1%} chance of churning")
        st.markdown("**Action:** Send this customer a check-in email or special discount.")
    else:
        st.success(f"🟢 LOW RISK — {prob:.1%} chance of churning")
        st.markdown("**Action:** This customer is happy. Keep up the good work!")

    # Progress bar
    st.progress(float(prob))
    st.caption(f"Churn probability: {prob:.1%}")