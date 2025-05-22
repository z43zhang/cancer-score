# 🌐 **Live Demo**: [Click here to try it out](https://cancer-score.streamlit.app/)

![App Preview](https://github.com/z43zhang/cancer-score/blob/main/assets/demo_image.png)

### 🩺 A machine learning regression app for predicting cancer severity scores using patient lifestyle, environmental, and clinical data.

---

# 🚀 Features

* 🔢 **Multimodal Input Handling** — Accepts both numeric (e.g., Age, Risk Scores) and categorical (e.g., Gender, Country, Cancer Type) features via interactive widgets
* 🧰 **Full scikit-learn Pipeline** — Modular ColumnTransformer handles imputation, encoding, and scaling
* 🤖 **Tuned ElasticNet Regression** — Balances L1 and L2 regularization with optimal hyperparameters (via Optuna)
* 📉 **Dimensionality Insights** — PCA applied to analyze intrinsic data structure and variance retention
* 📊 **Model Interpretability** — Displays top 5 contributing features dynamically based on model coefficients
* ⚙️ **Live Inference** — Real-time prediction with preprocessing + model inference pipeline

---

# 📂 Dataset

- Source: [Global Cancer Patients 2015–2024 (Kaggle)](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024/data)
- Records: 50,000 entries from 2015 to 2024
- Features include:
  - Demographics (Age, Gender, Country)
  - Lifestyle factors (Smoking, Alcohol Use, Obesity Level)
  - Environmental exposure (Air Pollution)
  - Medical history (Genetic Risk, Cancer Type)
- Target: `Target_Severity_Score` (continuous)

> 🚫 Features like `Treatment_Cost_USD`, `Survival_Years`, and `Cancer_Stage` were excluded during training to avoid data leakage.

---

# 🛠️ Tech Stack

| Component                | Description                           |
| ------------------------ | ------------------------------------- |
| `pandas`, `joblib`       | Data handling + model/pipeline saving |
| `scikit-learn`           | Preprocessing, Pipelines, ElasticNet  |
| `optuna`, `GridSearchCV` | Hyperparameter tuning                 |
| `matplotlib`, `shap`     | Interpretability & visualization      |
| `streamlit`              | Frontend and deployment engine        |

---

# 🧪 ML Pipeline Breakdown

This project follows a full ML lifecycle: from data exploration and preprocessing to model tuning and deployment. The work is structured into **three modular notebooks** and one deployment script.

## 📓 Notebook Walkthrough

| Notebook | Description |
|----------|-------------|
| [`1_Explore_Data_Analysis.ipynb`](https://github.com/z43zhang/cancer-score/blob/main/notebooks/1_Explore_Data_Analysis.ipynb) | EDA, visualizations, correlation checks |
| [`2_Data_Processing.ipynb`](https://github.com/z43zhang/cancer-score/blob/main/notebooks/2_Data_Processing.ipynb) | Feature engineering, pipeline construction, PCA |
| [`3_Severity_Score_Modeling.ipynb`](https://github.com/z43zhang/cancer-score/blob/main/notebooks/3_Severity_Score_Modeling.ipynb) | Model training, evaluation, hyperparameter tuning |

---

## 🔧 Pipeline & Feature Engineering

- Built a **`ColumnTransformer` pipeline** combining:
  - `StandardScaler` for numerical features
  - `OneHotEncoder` for categorical features
- Applied **PCA** to analyze feature redundancy
  - First 8 components explained ~80% variance

> ✅ All transformations were saved using `joblib` to enable plug-and-play deployment.

---

## 📋 Baseline Model Performance

| Model             | R²     | RMSE   | MAE    |
|------------------|--------|--------|--------|
| Bayesian Ridge    | 0.7866 | 0.5509 | 0.4785 |
| Linear Regression | 0.7866 | 0.5510 | 0.4785 |
| Ridge             | 0.7866 | 0.5510 | 0.4785 |
| Stacking Regressor| 0.7826 | 0.5561 | 0.4815 |
| CatBoost          | 0.7815 | 0.5575 | 0.4824 |
| LightGBM          | 0.7801 | 0.5593 | 0.4832 |
| ElasticNet        | 0.7720 | 0.5695 | 0.4880 |
| Lasso             | 0.7550 | 0.5903 | 0.5002 |
| KNN               | 0.6862 | 0.6681 | 0.5525 |

---

## 🔧 Tuned Model Performance

| Model                       | R²     | RMSE   | MAE    | ΔR²    | ΔRMSE   | ΔMAE   |
|----------------------------|--------|--------|--------|--------|---------|--------|
| ElasticNet (Tuned)         | 0.7868 | 0.5507 | 0.4783 | 🟢 +0.0148 | 🟢 −0.0188 | 🟢 −0.0097 |
| Lasso (Tuned)              | 0.7868 | 0.5508 | 0.4784 | 🟢 +0.0318 | 🟢 −0.0395 | 🟢 −0.0218 |
| Stacking Regressor (Tuned) | 0.7867 | 0.5508 | 0.4783 | 🟢 +0.0041 | 🟢 −0.0053 | 🟢 −0.0032 |
| Ridge (Tuned)              | 0.7866 | 0.5509 | 0.4785 | ⚪ ±0.0000 | 🟢 −0.0001 | ⚪ ±0.0000 |
| CatBoost (Tuned)           | 0.7859 | 0.5519 | 0.4789 | 🟢 +0.0044 | 🟢 −0.0056 | 🟢 −0.0035 |
| LightGBM (Tuned)           | 0.7847 | 0.5535 | 0.4798 | 🟢 +0.0046 | 🟢 −0.0058 | 🟢 −0.0034 |

---

## 📉 PCA-Based Model Evaluation

PCA-transformed inputs (8–15 components) were tested to evaluate the impact of dimensionality reduction on model performance.

- PCA did **not significantly improve** predictive accuracy compared to the original feature space.
- Models showed stable performance across 8-15 PCA components, confirming dimensionality reduction can be reduced without performance loss.
- Final models were trained without PCA to maintain full interpretability.

> PCA visualizations and explained variance analysis are available in the [3_Severity_Score_Modeling.ipynb](https://github.com/z43zhang/cancer-score/blob/main/notebooks/3_Severity_Score_Modeling.ipynb).

---

## 🗳️ Final Model Choice

The **ElasticNet (Tuned)** model was selected for its:
- 🧩 Balance between bias and variance via combined L1 and L2 regularization
- 🧠 Interpretability through feature coefficients — aligned with clinical expectations
- 🔁 Consistently stable results across cross-validation folds
- ⚡ Lightweight and efficient — suitable for real-time inference
- 📈 Achieves top-tier performance in R², RMSE, and MAE

> 🧾 Evaluation metrics (on test set):  
> **R²** = 0.7868 | **RMSE** = 0.5507 | **MAE** = 0.4783

---

## 🚀 Deployment (app.py)

- Built with `Streamlit` for real-time predictions
- Loads `elasticnet_tuned.pkl` and `preprocessor.pkl`
- Interactive inputs + top feature breakdown
- Fully hosted via [Streamlit Cloud](https://cancer-score.streamlit.app/)

```python
# app.py handles:
# - Form input
# - Joblib-based model loading
# - Top 5 feature weights display using HTML-styled tables
```

> 🎯 Try it live: [Click here to test the app](https://cancer-score.streamlit.app/)

---

# 🧪 Example Prediction

### 🧾 Input:

```
Age = 50
Gender = Male
Country = Germany
Year of Diagnosis = 2012
Cancer Type = Liver
Smoking = 2.2
Genetic Risk = 5.8
Air Pollution = 4.5
Obesity Level = 7.2
Alcohol Use = 6.5
```

### 📋 Output:

```
🎯 Predicted Cancer Severity Score: 4.91

ℹ️ Interpretation: This score falls in the category: 🟡 Moderate

📊 This score ranks in the 48.42th percentile of the entire database
```

### 🔍 Feature Impact Breakdown (SHAP Waterfall Plot):

![App Preview](https://github.com/z43zhang/cancer-score/blob/main/assets/example_prediction.png)

---


# 🛠️ Installation

```bash
git clone https://github.com/z43zhang/cancer-score-predictor.git
cd cancer-score-predictor
pip install -r requirements.txt
streamlit run app.py
```

---


