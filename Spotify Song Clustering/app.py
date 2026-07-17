import streamlit as st
import pandas as pd
import numpy as np
import pickle


# ---------------------------
# File Paths
# ---------------------------
model_path = r"C:\Users\inter\OneDrive\Desktop\Spotify Song Clustering\Spotify_Cluster_model.pkl"

scaler_path = r"C:\Users\inter\OneDrive\Desktop\Spotify Song Clustering\Spotify_scaler.pkl"

dataset_path = r"C:\Users\inter\OneDrive\Desktop\Spotify Song Clustering\Spotify_Song_Clustering_1500.csv"


# ---------------------------
# Load Model and Scaler
# ---------------------------

model = pickle.load(open(model_path, "rb"))

scaler = pickle.load(open(scaler_path, "rb"))

df = pd.read_csv(dataset_path)


# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🎵 Spotify Song Clustering")

st.write(
    "Predict song cluster based on audio features"
)


st.sidebar.header("Enter Song Features")


# Input Features

danceability = st.sidebar.number_input(
    "Danceability",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

energy = st.sidebar.number_input(
    "Energy",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

loudness = st.sidebar.number_input(
    "Loudness",
    value=-5.0
)

speechiness = st.sidebar.number_input(
    "Speechiness",
    min_value=0.0,
    max_value=1.0,
    value=0.05
)

acousticness = st.sidebar.number_input(
    "Acousticness",
    min_value=0.0,
    max_value=1.0,
    value=0.2
)

instrumentalness = st.sidebar.number_input(
    "Instrumentalness",
    min_value=0.0,
    max_value=1.0,
    value=0.0
)

valence = st.sidebar.number_input(
    "Valence",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

popularity = st.sidebar.number_input(
    "Popularity",
    min_value=0,
    max_value=100,
    value=50
)

tempo = st.sidebar.number_input(
    "Tempo",
    value=120
)


# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Cluster"):


    input_data = np.array(
        [
            danceability,
            energy,
            loudness,
            speechiness,
            acousticness,
            instrumentalness,
            valence,
            popularity,
            tempo
        ]
    ).reshape(1,-1)


    scaled_data = scaler.transform(input_data)


    cluster = model.predict(scaled_data)


    st.success(
        f"Predicted Song Cluster : {cluster[0]}"
    )


# ---------------------------
# Dataset Preview
# ---------------------------

st.subheader("Dataset Preview")

st.dataframe(df.head())