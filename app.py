# app.py
import streamlit as st
import numpy as np

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/elasticnet_tuned.pkl")

# Access OneHotEncoder inside the "cat" pipeline
encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
category_lists = encoder.categories_
gender_categories, country_categories, cancer_categories = category_lists

# App title
st.markdown("<h1 style='text-align: center;'>📉 Cancer Severity Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter patient info below to predict the cancer severity score.</p>", unsafe_allow_html=True)

# Input fields
age = st.number_input("Age", min_value=10, max_value=110, value=50)
gender = st.selectbox("Gender", gender_categories)
country = st.selectbox("Country/Region", country_categories)
year = st.slider("Year of Diagnosis", 1990, 2025, value=2015)

genetic_risk = st.slider("Genetic Risk", 0.0, 10.0, step=0.1)
air_pollution = st.slider("Air Pollution", 0.0, 10.0, step=0.1)
alcohol_use = st.slider("Alcohol Use", 0.0, 10.0, step=0.1)
smoking = st.slider("Smoking", 0.0, 10.0, step=0.1)
obesity = st.slider("Obesity Level", 0.0, 10.0, step=0.1)

cancer_type = st.selectbox("Cancer Type", cancer_categories)

# Predict button
if st.button("Predict"):
    # Construct input DataFrame
    input_dict = {
        "Age": age,
        "Gender": gender,
        "Country_Region": country,
        "Year": year,
        "Genetic_Risk": genetic_risk,
        "Air_Pollution": air_pollution,
        "Alcohol_Use": alcohol_use,
        "Smoking": smoking,
        "Obesity_Level": obesity,
        "Cancer_Type": cancer_type
    }
    input_df = pd.DataFrame([input_dict])

    # Preprocess & predict
    input_transformed = preprocessor.transform(input_df)
    prediction = model.predict(input_transformed)[0]

    st.success(f"🎯 Predicted Cancer Severity Score: **{prediction:.2f}**")

    # SHAP explanation
    st.subheader("🔍 SHAP Feature Contribution")

    # Use model and feature names for explainer
    explainer = shap.LinearExplainer(model, input_transformed, feature_names=preprocessor.get_feature_names_out())

    shap_values = explainer(input_transformed)

    # Plot waterfall and display in Streamlit
    shap.plots.waterfall(shap_values[0], max_display=6, show=False)
    fig = plt.gcf()  # Get current figure created by SHAP
    st.pyplot(fig)

