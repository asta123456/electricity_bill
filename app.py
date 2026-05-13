import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD MODEL AND SCALER

model = joblib.load("electricity_bill_model.pkl")

scaler = joblib.load("scaler.pkl")

# LOAD DATASET

df = pd.read_csv("electricity_bill.csv")

# KEEP ONLY NUMERIC COLUMNS

df = df.select_dtypes(include=np.number)

# TARGET COLUMN

target_column = df.columns[-1]

# FEATURE COLUMNS

feature_columns = df.drop(target_column, axis=1).columns

# TITLE

st.title("Electricity Bill Prediction")

st.write(
    "Predict Electricity Bill using Machine Learning"
)

# STORE USER INPUTS

user_inputs = []

# CREATE INPUT BOXES DYNAMICALLY

for col in feature_columns:

    value = st.number_input(
        f"Enter {col}",
        value=0.0
    )

    user_inputs.append(value)

# PREDICT BUTTON

if st.button("Predict"):

    # CONVERT TO ARRAY

    input_data = np.array([user_inputs])

    # SCALE DATA

    input_data = scaler.transform(input_data)

    # PREDICT

    prediction = model.predict(input_data)

    # SHOW RESULT

    st.success(
        f"Predicted Electricity Bill = ₹ {prediction[0]:.2f}"
    )
