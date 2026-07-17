import streamlit as st
import pickle
import numpy as np

model = pickle.load(open(r"C:\Users\inter\OneDrive\Desktop\Flight Ticket Price Prediction\Flight_Price_Model.pkl", "rb"))

st.set_page_config(page_title="Flight Price Predictor", page_icon="✈")

st.title("✈ Flight Price Prediction App")

# ---------------- Inputs with icons ----------------
airline = st.number_input("✈ Airline (encoded)", 0)
source = st.number_input("📍 Source (encoded)", 0)
destination = st.number_input("📍 Destination (encoded)", 0)
stops = st.number_input("🔁 Stops", 0)
duration = st.number_input("⏱ Duration (minutes)", 0)
days_left = st.number_input("📅 Days Left for Travel", 0)
travel_class = st.number_input("💺 Travel Class (0=Economy, 1=Business)", 0, 1)

# ---------------- Prediction ----------------
if st.button("💰 Predict Flight Price"):
    features = np.array([[airline, source, destination, stops, duration, days_left, travel_class]])
    
    prediction = model.predict(features)

    st.success(f"✈ Estimated Price: ₹ {prediction[0]:.2f}")