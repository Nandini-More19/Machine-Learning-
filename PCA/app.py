import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Air Pollution PCA",
    page_icon="🌍",
    layout="wide"
)


# -------------------------------
# Title
# -------------------------------

st.title("🌍 Air Pollution Analysis using PCA")
st.subheader("🤖 AQI Category Prediction using PC1 & PC2")


# -------------------------------
# Load Dataset
# -------------------------------

csv_path = r"C:\Users\inter\OneDrive\Desktop\PCA\air_pollution_1500.csv"

df = pd.read_csv(csv_path)


# -------------------------------
# Load Model
# -------------------------------

model_path = r"C:\Users\inter\OneDrive\Desktop\PCA\model.pkl"

model = joblib.load(model_path)



# -------------------------------
# Dataset Preview
# -------------------------------

st.header("📂 Dataset Preview")

st.dataframe(df.head())


col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Total Rows",
        df.shape[0]
    )

with col2:
    st.metric(
        "Total Columns",
        df.shape[1]
    )


# -------------------------------
# Visualization
# -------------------------------

st.header("📊 Data Visualization")


option = st.selectbox(
    "Choose Graph",
    [
        "Correlation Heatmap",
        "AQI Distribution",
        "Temperature Boxplot"
    ]
)


if option == "Correlation Heatmap":

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)



elif option == "AQI Distribution":

    fig, ax = plt.subplots()

    sns.histplot(
        df["AQI"],
        bins=30,
        kde=True,
        color="green",
        ax=ax
    )

    ax.set_title("AQI Distribution")

    st.pyplot(fig)



elif option == "Temperature Boxplot":

    fig, ax = plt.subplots()

    sns.boxplot(
        y=df["Temperature"],
        color="skyblue",
        ax=ax
    )

    ax.set_title(
        "Temperature Distribution"
    )

    st.pyplot(fig)



# -------------------------------
# PCA Input
# -------------------------------

st.header("🧠 Enter PCA Components")


st.info(
    "Model is trained using only two PCA features: PC1 and PC2"
)


PC1 = st.number_input(
    "🔵 PC1 Value",
    value=0.0
)


PC2 = st.number_input(
    "🟢 PC2 Value",
    value=0.0
)



# -------------------------------
# Prediction
# -------------------------------

if st.button("🚀 Predict AQI Category"):


    input_data = np.array(
        [
            [
                PC1,
                PC2
            ]
        ]
    )


    prediction = model.predict(input_data)


    result = prediction[0]


    categories = {
        0:"🟢 Good Air Quality",
        1:"🟡 Moderate Air Quality",
        2:"🟠 Poor Air Quality",
        3:"🔴 Very Poor Air Quality"
    }


    st.success(
        f"Prediction Result : {categories.get(result,'Unknown')}"
    )



# -------------------------------
# Footer
# -------------------------------

st.markdown(
"""
---
### 🌱 Project : Air Pollution Analysis using PCA
### 📌 Technique : Principal Component Analysis
### 🐍 Python + Streamlit
"""
)