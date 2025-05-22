# app.py
import streamlit as st
import numpy as np

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

@st.cache_data
def load_scores():
    df = pd.read_csv("data/global_cancer_patients_2015_2024.csv")
    return df["Target_Severity_Score"]

score_distribution = load_scores()

# Load model and preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/elasticnet_tuned.pkl")

# Access OneHotEncoder inside the "cat" pipeline
encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
category_lists = encoder.categories_
gender_categories, country_categories, cancer_categories = category_lists

# App title
st.markdown("<h1 style='text-align: center;'>📉 Cancer Severity Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: right;'>Developed by Zhonghao Zhang</p>", unsafe_allow_html=True)

with st.expander("📖 How this model works"):
    st.markdown("""
    - ✅ Predicts **cancer severity scores** using patient lifestyle and medical data.
    - 🔍 It was trained using ElasticNet regression after preprocessing with a full pipeline.
    - 📊 Top influencing factors include Smoking, Genetic Risk, and Air Pollution.
    - 📈 Interpretability is built-in using model coefficients.
    """)


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
    # Build input dictionary from UI
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

    # Preprocess input and predict
    input_transformed = preprocessor.transform(input_df)
    prediction = model.predict(input_transformed)[0]
    st.success(f"🎯 Predicted Cancer Severity Score: **{prediction:.2f}**")

    if prediction < 3:
        level = "🟢 Low"
    elif prediction < 5:
        level = "🟡 Moderate"
    elif prediction < 7:
        level = "🟠 High"
    else:
        level = "🔴 Very High"
    st.info(f"**Interpretation:** This score falls in the category: {level}")

    # Rank & percentile calculation
    percentile = (score_distribution < prediction).mean() * 100
    rank = (score_distribution < prediction).sum() + 1  # Rank starts from 1
    total = len(score_distribution)

    st.info(f"📊 This score ranks in the **{percentile:.2f}th percentile** of the entire database")

    # SHAP: Create background with correct dimensions (same shape as input)
    background = np.zeros((1, input_transformed.shape[1]))

    # Use LinearExplainer (optimized for linear models like ElasticNet)
    explainer = shap.LinearExplainer(model, background)

    # Compute SHAP values (Explanation object)
    raw_shap = explainer(input_transformed)

    # Attach feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_out()
    shap_values = shap.Explanation(
        values=raw_shap.values,
        base_values=raw_shap.base_values,
        data=raw_shap.data,
        feature_names=feature_names
    )

    # Waterfall plot for this instance
    st.markdown("### 📊 SHAP Waterfall Plot (Feature Impact Breakdown)")
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)




