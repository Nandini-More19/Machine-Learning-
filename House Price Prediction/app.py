import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# App Title
st.title("🏠 House Price Prediction")

st.write("Enter the house details below:")

# User Inputs
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
living_area = st.number_input("Living Area (sq.ft)", min_value=500, max_value=5000, value=1500)

# Predict Button
if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "BedroomAbvGr": [bedrooms],
        "FullBath": [bathrooms],
        "GrLivArea": [living_area]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted House Price: ₹ {prediction[0]:,.2f}")