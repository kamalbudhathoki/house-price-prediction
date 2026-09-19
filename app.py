import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", layout="wide")

MODEL_PATH = Path("models/house_price_model.joblib")


def get_top_neighborhoods():
    df = pd.read_csv("data/train.csv")
    return df["Neighborhood"].value_counts().head(12).index.tolist()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("House Price Predictor")
st.caption("Ames housing dataset (Kaggle). Tuned XGBoost + preprocessing pipeline "
           "trained on the 2006-2010 Iowa data.")

predictor = load_model()
NEIGHBORHOODS = get_top_neighborhoods()

QUALITY_CODES = ["Po", "Fa", "TA", "Gd", "Ex"]
BASEMENT_CODES = ["NA", "Po", "Fa", "TA", "Gd", "Ex"]
DRIVE_CODES = ["N", "P", "Y"]
SLOPE_CODES = ["Gtl", "Mod", "Sev"]

with st.form("house_form"):
    st.subheader("Size & layout")
    col1, col2, col3 = st.columns(3)
    inputs = {
        "OverallQual": col1.slider("Overall quality (1-10)", 1, 10, 7),
        "OverallCond": col1.slider("Overall condition (1-10)", 1, 10, 5),
        "LotArea": col1.number_input("Lot area (sq ft)", min_value=500, max_value=215000, value=9500, step=100),
        "TotalBsmtSF": col2.number_input("Basement area (sq ft)", min_value=0, max_value=6110, value=1000, step=10),
        "GrLivArea": col2.number_input("Above-grade living area (sq ft)", min_value=300, max_value=5640, value=1500, step=10),
        "1stFlrSF": col2.number_input("1st floor (sq ft)", min_value=300, max_value=5090, value=1100, step=10),
        "2ndFlrSF": col3.number_input("2nd floor (sq ft)", min_value=0, max_value=2065, value=0, step=10),
        "TotRmsAbvGrd": col3.select_slider("Rooms above grade", options=range(2, 15), value=7),
        "BedroomAbvGr": col3.select_slider("Bedrooms", options=range(0, 7), value=3),
        "KitchenAbvGr": col1.select_slider("Kitchens", options=range(0, 4), value=1),
    }
    col1, col2, col3 = st.columns(3)
    inputs.update({
        "FullBath": col1.select_slider("Full baths", options=range(0, 4), value=2),
        "HalfBath": col2.select_slider("Half baths", options=range(0, 3), value=1),
        "BsmtFullBath": col3.select_slider("Full baths (bsmt)", options=range(0, 4), value=0),
        "BsmtHalfBath": col1.select_slider("Half baths (bsmt)", options=range(0, 3), value=0),
        "Fireplaces": col2.select_slider("Fireplaces", options=range(0, 4), value=1),
        "GarageCars": col3.select_slider("Garage cars", options=range(0, 5), value=2),
        "GarageArea": col1.number_input("Garage area (sq ft)", min_value=0, max_value=1488, value=500, step=10),
    })

    st.subheader("Age & sale info")
    col1, col2, col3 = st.columns(3)
    inputs.update({
        "YearBuilt": col1.slider("Year built", 1870, 2010, 1980),
        "YearRemodAdd": col2.slider("Year remodeled/added", 1950, 2010, 2000),
        "YrSold": col3.select_slider("Year sold", options=range(2006, 2011), value=2008),
        "MoSold": col1.select_slider("Month sold", options=range(1, 13), value=7),
        "Neighborhood": col2.selectbox("Neighborhood", NEIGHBORHOODS),
        "MSZoning": col3.selectbox("Zoning", ["RL", "RM", "C (all)", "FV", "RH"]),
        "CentralAir": col1.selectbox("Central air", ["Y", "N"]),
        "PavedDrive": col2.selectbox("Paved driveway", DRIVE_CODES),
        "LandSlope": col3.selectbox("Land slope", SLOPE_CODES),
    })

    st.subheader("Quality ratings")
    col1, col2, col3 = st.columns(3)
    inputs.update({
        "ExterQual": col1.selectbox("Exterior quality", QUALITY_CODES, index=3),
        "ExterCond": col2.selectbox("Exterior condition", QUALITY_CODES, index=2),
        "KitchenQual": col3.selectbox("Kitchen quality", QUALITY_CODES, index=3),
        "HeatingQC": col1.selectbox("Heating quality", QUALITY_CODES, index=2),
        "BsmtQual": col2.selectbox("Basement quality", BASEMENT_CODES),
        "FireplaceQu": col3.selectbox("Fireplace quality", BASEMENT_CODES),
        "MasVnrType": col1.selectbox("Masonry veneer", ["None", "BrkFace", "BrkCmn", "Stone"]),
    })
    inputs["MasVnrType"] = "None" if inputs["MasVnrType"] == "None" else inputs["MasVnrType"]
    inputs["MasVnrArea"] = 0 if inputs["MasVnrType"] == "None" else 100

    submitted = st.form_submit_button("Predict sale price", type="primary")

if submitted:
    with st.spinner("Predicting..."):
        price = predictor.predict(inputs)
    st.success(f"Estimated sale price")
    st.markdown(f"## $ {price:,.0f}")
else:
    st.info("Fill in the house features and click **Predict sale price**.")