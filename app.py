# ==========================================
# app.py
# ELECTRICITY BILL PREDICTION APP
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# LOAD MODEL AND SCALER
# ==========================================

model = joblib.load("electricity_bill_model.pkl")

scaler = joblib.load("scaler.pkl")

# ==========================================
# TITLE
# ==========================================

st.title("Electricity Bill Prediction")

st.write(
    "Predict Electricity Bill using Machine Learning Regression"
)

# ==========================================
# USER INPUTS
# ==========================================

units_consumed = st.number_input(
    "Units Consumed",
    min_value=0.0
)

ac_hours = st.number_input(
    "AC Usage Hours",
    min_value=0.0
)

family_members = st.number_input(
    "Family Members",
    min_value=1
)

appliance_count = st.number_input(
    "Appliance Count",
    min_value=0
)

# ==========================================
# CREATE INPUT ARRAY
# ==========================================

input_data = np.array([
    [
        units_consumed,
        ac_hours,
        family_members,
        appliance_count
    ]
])

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("Predict Electricity Bill"):

    # SCALE INPUT

    input_data = scaler.transform(input_data)

    # PREDICT

    prediction = model.predict(input_data)

    # DISPLAY RESULT

    st.success(
        f"Predicted Electricity Bill = ₹ {prediction[0]:.2f}"
    )

# ==========================================
# SAMPLE DATA
# ==========================================

st.subheader("Sample Input")

sample_df = pd.DataFrame({
    "Units_Consumed": [250],
    "AC_Hours": [6],
    "Family_Members": [4],
    "Appliance_Count": [8]
})

st.table(sample_df)