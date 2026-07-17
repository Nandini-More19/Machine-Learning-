import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# TITLE
# -----------------------------
st.title("💃 Dance Performance Rating Prediction App")

# -----------------------------
# LOAD CSV (PATH 2)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\inter\OneDrive\Desktop\Dance Performance Rating Prediction\dance_performance_1000.csv")
    return df

df = load_data()

st.subheader("📊 Dataset Preview")
st.write(df.head())

# -----------------------------
# FEATURES & TARGET
# -----------------------------
target_column = "Rating"

X = df.drop(target_column, axis=1)
y = df[target_column]

# -----------------------------
# TRAIN MODEL (ONLY IF model.pkl NOT FOUND)
# -----------------------------
try:
    # -----------------------------
    # LOAD PICKLE MODEL (PATH 3)
    # -----------------------------
    model = pickle.load(open(r"C:\Users\inter\OneDrive\Desktop\Dance Performance Rating Prediction\model.pkl", "rb"))
    st.success("✅ Loaded model from model.pkl")

except:
    st.warning("⚠️ model.pkl not found, training new model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    pickle.dump(model, open(r"C:\Users\inter\OneDrive\Desktop\Dance Performance Rating Prediction\model.pkl", "wb"))

    st.success("✅ Model trained and saved as model.pkl")

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("🎯 Enter Dance Performance Details")

input_data = []

for col in X.columns:
    val = st.number_input(f"{col}", value=0.0)
    input_data.append(val)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Rating"):
    prediction = model.predict([input_data])
    st.success(f"⭐ Predicted Rating: {prediction[0]:.2f}")