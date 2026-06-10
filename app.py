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

# Shared color scheme
UBER_BLUE   = "#276EF1"
UBER_GREEN  = "#06C167"
UBER_DARK   = "#142328"
BLUE_SEQ    = "Blues"   # for the heatmap

# Make every matplotlib/seaborn chart use the same look
sns.set_theme(style="whitegrid")
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["figure.autolayout"] = True   # stops labels getting cut off

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

# Distance column from coordinates
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 3959 * 2 * np.arcsin(np.sqrt(a))

df["distance_miles"] = haversine(
    df["pickup_longitude"], df["pickup_latitude"],
    df["dropoff_longitude"], df["dropoff_latitude"]
)

# Time columns from pickup_datetime
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df["hour"] = df["pickup_datetime"].dt.hour
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
df["year"] = df["pickup_datetime"].dt.year      # <-- this is what your caption needs

# Remove outliers
df = df[df["fare_amount"].between(2.5, 100)]
df = df[df["passenger_count"].between(1, 6)]
df = df[df["distance_miles"].between(0.1, 60)]
df = df.dropna(subset=["hour"])

## Step 03 - Pages
if page == "Introduction 📘":
    st.title("🚕 Predicting Uber Fares in New York City")
    st.subheader("01 Introduction 📘")
    st.caption(f"Data spans {df['year'].min()}–{df['year'].max()} "
           f"(about {2026 - int(df['year'].max())} years old).")
    
    st.markdown("### The Business Problem")
    st.markdown("**Goal:** predict the price of an Uber ride in NYC from trip distance, time of day, and passengers.")

    # Quick headline numbers
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Average fare", f"${df['fare_amount'].mean():.2f}")

    st.markdown("##### Data Preview")
    view = st.radio("Show from:", ["Head (top)", "Tail (bottom)"], horizontal=True)
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    if view == "Head (top)":
        st.dataframe(df.head(rows))
    else:
        st.dataframe(df.tail(rows))

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)
    if missing.sum() == 0:
        st.success("No missing values found")
    else:
        st.warning("You have missing values")

    st.markdown("##### 📈 Summary Statistics")
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
        sns.histplot(df["fare_amount"], bins=40, kde=True, color=UBER_BLUE, ax=ax)
        ax.set_xlabel("Fare (USD)")
        ax.set_title("Most rides cost $5-15")
        st.pyplot(fig)
        st.caption("Typical rides are cheap and short. The small bumps near $50-57 are airport trips.")

    with tab2:
        st.markdown("#### Does distance drive the fare?")
        fig, ax = plt.subplots()
        sns.scatterplot(data=plot_df, x="distance_miles", y="fare_amount",
                        alpha=0.3, color=UBER_BLUE, ax=ax)
        ax.set_xlabel("Trip distance (miles)")
        ax.set_ylabel("Fare (USD)")
        ax.set_title("Fare rises almost linearly with distance")
        st.pyplot(fig)
        st.caption("This near-straight-line relationship is exactly why linear regression fits this problem.")

    with tab3:
        st.markdown("#### When are rides most expensive?")
        by_hour = df.groupby("hour")["fare_amount"].mean()
        fig, ax = plt.subplots()
        sns.lineplot(x=by_hour.index, y=by_hour.values, marker="o",
                     color=UBER_GREEN, linewidth=2.5, ax=ax)
        ax.set_xlabel("Hour of day (0-23)")
        ax.set_ylabel("Average fare (USD)")
        ax.set_title("Early-morning rides cost the most")
        st.pyplot(fig)
        st.caption("The 5 AM spike is likely long airport runs before flights; evening rides are short and cheap.")

    with tab4:
        st.markdown("#### Correlation between numeric columns")
        df_numeric = df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f",
                    cmap=BLUE_SEQ, ax=ax)
        ax.set_title("Distance is the strongest predictor of fare (0.89)")
        st.pyplot(fig)
        st.caption("Distance correlates 0.89 with fare far higher than anything else.")

elif page == "Prediction 🔮":
    st.subheader("03 Prediction 🔮")
    st.markdown("We train a **Linear Regression** model to predict the fare of a ride "
                "from things we know *before* the trip starts.")

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics

    # The features (X) we predict FROM, and the target (y) we predict
    features = ["distance_miles", "passenger_count", "hour", "day_of_week"]
    X = df[features]
    y = df["fare_amount"]

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # --- How accurate is it? ---
    st.markdown("#### How accurate is the model?")
    r2   = metrics.r2_score(y_test, predictions)
    mae  = metrics.mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

    c1, c2, c3 = st.columns(3)
    c1.metric("R² (variance explained)", f"{r2:.3f}")
    c2.metric("Average error (MAE)", f"${mae:.2f}")
    c3.metric("Typical error (RMSE)", f"${rmse:.2f}")
    st.caption(f"On average the prediction is off by about ${mae:.2f}. "
               f"An R² of {r2:.2f} means the model explains about {r2*100:.0f}% "
               "of why fares differ.")

    # The driving variables
    st.markdown("#### What drives the prediction?")
    coef_df = pd.DataFrame({
        "Feature": features,
        "Dollars added per unit": model.coef_
    }).sort_values("Dollars added per unit", key=lambda s: s.abs(), ascending=False)
    st.dataframe(coef_df, hide_index=True, use_container_width=True)
    st.caption(f"Each extra mile adds about ${model.coef_[0]:.2f} to the fare. "
               "Distance is by far the biggest driver — exactly what the data showed us.")

    # Predicted vs actual
    st.markdown("#### Predicted vs actual fares")
    sample = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}) \
                .sample(min(2000, len(y_test)), random_state=1)
    fig, ax = plt.subplots()
    ax.scatter(sample["Actual"], sample["Predicted"], alpha=0.3, color=UBER_BLUE)
    ax.plot([sample["Actual"].min(), sample["Actual"].max()],
            [sample["Actual"].min(), sample["Actual"].max()],
            "--", color=UBER_GREEN, linewidth=2)
    ax.set_xlabel("Actual fare (dollars)")
    ax.set_ylabel("Predicted fare (dollars)")
    ax.set_title("Points near the green line are good predictions")
    st.pyplot(fig)
    st.caption("The model tracks normal rides well. The spread at higher fares is surge "
               "pricing and airport flat-rates, which a straight line cannot fully capture.")

    # Try it yourself 
    st.markdown("#### Try it yourself")
    DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    c1, c2 = st.columns(2)
    with c1:
        distance   = st.slider("Trip distance (miles)", 0.5, 30.0, 3.0, 0.5)
        passengers = st.slider("Passengers", 1, 6, 1)
    with c2:
        hour      = st.slider("Hour of day", 0, 23, 18)
        day_label = st.selectbox("Day of week", DAY_NAMES, index=4)
    day_of_week = DAY_NAMES.index(day_label)

    new_trip = pd.DataFrame([[distance, passengers, hour, day_of_week]], columns=features)
    predicted_fare = max(model.predict(new_trip)[0], 2.5)

    st.success(f"Estimated fare: ${predicted_fare:.2f}")
    st.caption(f"For a {distance:.1f}-mile trip with {passengers} passenger(s) "
               f"on a {day_label} at {hour:02d}:00.")
