# Streamlit App for Salary Prediction
# Run using: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

# ==============================
# 1. Load and Train Model (same logic as your code)
# ==============================

@st.cache_resource
def train_model():
    data = pd.read_csv("ai_market.csv")

    target = "Salary_USD"

    ordinal_cols = ["Experience_Level"]
    ordinal_order = [["Entry", "Mid", "Senior", "Executive"]]

    categorical_cols = ["Country", "Company_Type", "Job_Title", "Top_Skill", "Remote"]

    numeric_cols = [col for col in data.columns if col not in ordinal_cols + categorical_cols + [target]]

    X = data.drop(target, axis=1)
    y = data[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", OrdinalEncoder(categories=ordinal_order), ordinal_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    return model, data

model, data = train_model()

# ==============================
# 2. Streamlit UI
# ==============================

st.title("💼 AI Salary Predictor")
st.write("Predict salaries based on job details using Machine Learning")

# Dropdown options from dataset
experience = st.selectbox("Experience Level", ["Entry", "Mid", "Senior", "Executive"])
country = st.selectbox("Country", sorted(data["Country"].unique()))
company = st.selectbox("Company Type", sorted(data["Company_Type"].unique()))
job = st.selectbox("Job Title", sorted(data["Job_Title"].unique()))
skill = st.selectbox("Top Skill", sorted(data["Top_Skill"].unique()))
remote = st.selectbox("Remote Work", sorted(data["Remote"].unique()))

# If numeric columns exist, take input
numeric_inputs = {}
for col in data.columns:
    if col not in ["Salary_USD", "Experience_Level", "Country", "Company_Type", "Job_Title", "Top_Skill", "Remote"]:
        numeric_inputs[col] = st.number_input(f"{col}", value=float(data[col].mean()))

# ==============================
# 3. Prediction
# ==============================

if st.button("Predict Salary"):
    input_dict = {
        "Experience_Level": experience,
        "Country": country,
        "Company_Type": company,
        "Job_Title": job,
        "Top_Skill": skill,
        "Remote": remote
    }

    input_dict.update(numeric_inputs)

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Salary: ${prediction:,.2f}")

# ==============================
# 4. Extra Info
# ==============================

st.markdown("---")
st.write("Built with ❤️ by Md. Mahfuz Karim Nasim")
