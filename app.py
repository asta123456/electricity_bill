import streamlit as st
import numpy as np
import pandas as pd
import joblib

# PAGE CONFIG

st.set_page_config(
    page_title="Electricity Bill Prediction",
    page_icon="⚡"
)

# LOAD MODEL AND SCALER

model = joblib.load("electricity_bill_model.pkl")

scaler = joblib.load("scaler.pkl")

# GET FEATURE NAMES

feature_names = scaler.feature_names_in_

# TITLE

st.title("⚡ Electricity Bill Prediction")

st.write(
    "Predict Electricity Bill using Machine Learning"
)

# USER INPUTS

user_inputs = []

for feature in feature_names:

    value = st.number_input(
        f"Enter {feature}",
        value=0.0
    )

    user_inputs.append(value)

# PREDICT BUTTON

if st.button("Predict"):

    # CREATE DATAFRAME

    input_data = pd.DataFrame(
        [user_inputs],
        columns=feature_names
    )

    # SCALE INPUT

    input_scaled = scaler.transform(input_data)

    # PREDICT

    prediction = model.predict(input_scaled)

    # SHOW RESULT

    st.success(
        f"Predicted Electricity Bill = ₹ {prediction[0]:.2f}"
    )

    st.balloons()
