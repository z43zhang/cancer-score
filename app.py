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
st.markdown("<p style='text-align: center;'>Developed by Zhonghao Zhang</p>", unsafe_allow_html=True)

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

    # Get coefficients and feature names
    importances = model.coef_
    feature_names = preprocessor.get_feature_names_out()

    # Map raw feature names to human-readable labels
    name_map = {
        "num__Smoking": "Smoking",
        "num__Genetic_Risk": "Genetic Risk",
        "num__Air_Pollution": "Air Pollution",
        "num__Alcohol_Use": "Alcohol Use",
        "num__Obesity_Level": "Obesity Level",
        "num__Age": "Age",
        "num__Year": "Year of Diagnosis",
    }

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        "Factors": [name_map.get(f, f) for f in feature_names],
        "Weights": importances
    })

    # Sort and get top 5
    top_df = importance_df.reindex(importance_df["Weights"].abs().sort_values(ascending=False).index).head(5).reset_index(drop=True)
    top_df.index += 1  # Start index from 1
    top_df["Rank"] = top_df.index  # Create Rank column
    top_df["Weights"] = top_df["Weights"].map(lambda x: f"{x:.2f}".rstrip("0").rstrip("."))


    # Reorder columns: Rank, Factors, Weights
    top_df = top_df[["Rank", "Factors", "Weights"]]

    # Prepare styled HTML
    styled_table = (
        top_df.style
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}
        ])
        .hide(axis="index")
        .to_html()
    )

    # Centered subtitle
    st.markdown("<h3 style='text-align: center;'>🔍 Feature Importance</h3>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='display: flex; justify-content: center;'>{styled_table}</div>",
        unsafe_allow_html=True
    )


