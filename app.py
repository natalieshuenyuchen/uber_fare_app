## Step 00 - Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Uber Fare Predictor 🚕",
    layout="centered",
    page_icon="🚕",
)

## Step 01 - Sidebar navigation
st.sidebar.title("Uber NYC – Fare Analysis 🚕")
page = st.sidebar.selectbox(
    "Select Page",
    ["Introduction 📘", "Visualization 📊", "Prediction 🔮"]
)

# st.image("nyc.png")   # uncomment once you upload an image

## Step 02 - Load the dataset
df = pd.read_csv("uber.csv")
df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")  # drop the two ID columns

## Step 03 - Pages
if page == "Introduction 📘":
    st.title("🚕 Predicting Uber Fares in New York City")
    st.subheader("01 Introduction 📘")

    st.markdown("### The Business Problem")
    st.markdown("""
    *(write your business case here — see my draft below)*
    """)

    # Quick headline numbers
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Average fare", f"${df['fare_amount'].mean():.2f}")

    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)
    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ You have missing values")

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

elif page == "Visualization 📊":
    st.subheader("02 Data Visualization 📊")
    st.write("Coming soon — we'll build this next.")

elif page == "Prediction 🔮":
    st.subheader("03 Prediction 🔮")
    st.write("Coming soon.")
