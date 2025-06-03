# insurance_app.py

# -------------------
# Imports and Setup
# -------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# -------------------
# Load and Explore Data
# -------------------
df = pd.read_csv("insurance.csv")

# Optional: EDA (can be disabled in final app)
# st.write("### Sample Data")
# st.dataframe(df.head())

# -------------------
# Preprocessing
# -------------------
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------
# Train Model
# -------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------
# Streamlit App UI
# -------------------
st.title("💸 Medical Insurance Price Predictor")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
# User enters height and weight instead of BMI
height_cm = st.number_input("Height (in cm)", min_value=100, max_value=250, value=170)
weight_kg = st.number_input("Weight (in kg)", min_value=30, max_value=200, value=70)

# Calculate BMI
height_m = height_cm / 100
bmi = weight_kg / (height_m ** 2)
st.write(f"📏 Your calculated BMI: **{bmi:.2f}**")

#bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# -------------------
# Prepare Input for Prediction
# -------------------
input_data = {
    'age': age,
    'bmi': bmi,
    'children': children,
    'sex_male': 1 if sex == 'male' else 0,
    'smoker_yes': 1 if smoker == 'yes' else 0,
    'region_northwest': 1 if region == 'northwest' else 0,
    'region_southeast': 1 if region == 'southeast' else 0,
    'region_southwest': 1 if region == 'southwest' else 0
}

input_df = pd.DataFrame([input_data])

# -------------------
# Predict
# -------------------
if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_df)[0]

    # Currency conversion (USD to INR)
    usd_to_inr = 83  # Update as needed
    prediction_inr = prediction * usd_to_inr

    st.success(f"💰 Estimated Insurance Cost:")
    #st.write(f"- 💵 USD: **${prediction:,.2f}**")
    st.write(f"- 🇮🇳 INR: **₹{prediction_inr:,.2f}**")

