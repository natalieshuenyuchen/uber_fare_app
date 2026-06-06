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

## Step 02 - Load and prepare the dataset
df = pd.read_csv("uber.csv")
df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")
df = df.dropna()

# Create a DISTANCE column from the pickup/dropoff coordinates
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 3959 * 2 * np.arcsin(np.sqrt(a))   # distance in miles

df["distance_miles"] = haversine(
    df["pickup_longitude"], df["pickup_latitude"],
    df["dropoff_longitude"], df["dropoff_latitude"]
)

# Pull the HOUR and DAY out of the pickup time 
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df["hour"] = df["pickup_datetime"].dt.hour
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek   # Monday = 0

# Remove bad rows / outliers so the charts aren't ruined 
df = df[df["fare_amount"].between(2.5, 100)]
df = df[df["passenger_count"].between(1, 6)]
df = df[df["distance_miles"].between(0.1, 60)]
df = df.dropna(subset=["hour"])

## Step 03 - Pages
if page == "Introduction 📘":
    st.title("🚕 Predicting Uber Fares in New York City")
    st.subheader("01 Introduction 📘")

    st.markdown("### The Business Problem")
    st.markdown("**Goal:** predict the price of an Uber ride in NYC from trip distance, time of day, and passengers.")

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

    # Use a smaller sample so the charts draw quickly
    plot_df = df.sample(min(5000, len(df)), random_state=1)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Fare Distribution 💵", "Fare vs Distance 📏",
        "Fare by Hour 🕐", "Correlation 🔥"
    ])

    with tab1:
        st.markdown("#### How much do rides cost?")
        fig, ax = plt.subplots()
        sns.histplot(df["fare_amount"], bins=40, kde=True, ax=ax)
        ax.set_xlabel("Fare (USD)")
        st.pyplot(fig)

    with tab2:
        st.markdown("#### Does distance drive the fare?")
        fig, ax = plt.subplots()
        sns.scatterplot(data=plot_df, x="distance_miles", y="fare_amount",
                        alpha=0.3, ax=ax)
        ax.set_xlabel("Trip distance (miles)")
        ax.set_ylabel("Fare (USD)")
        st.pyplot(fig)

    with tab3:
        st.markdown("#### When are rides most expensive?")
        by_hour = df.groupby("hour")["fare_amount"].mean()
        fig, ax = plt.subplots()
        sns.lineplot(x=by_hour.index, y=by_hour.values, marker="o", ax=ax)
        ax.set_xlabel("Hour of day (0–23)")
        ax.set_ylabel("Average fare (USD)")
        st.pyplot(fig)

    with tab4:
        st.markdown("#### Correlation between numeric columns")
        df_numeric = df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

elif page == "Prediction 🔮":
    st.subheader("03 Prediction 🔮")
    st.write("")
