import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(
    page_title="HousePricePro",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 HousePricePro")
st.subheader("AI Powered Property Valuation")

st.write("Estimate property prices using our machine learning model.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", 500, 10000, 2000)
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)

with col2:
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)
    year = st.number_input("Year Built", 1900, 2025, 2010)

location = st.selectbox(
    "Location",
    ["Downtown", "Suburban", "Rural", "Urban"]
)

st.markdown("---")

predict = st.button("Estimate Property Value")

if predict:

    loc_downtown = 1 if location == "Downtown" else 0
    loc_suburban = 1 if location == "Suburban" else 0
    loc_rural = 1 if location == "Rural" else 0
    loc_urban = 1 if location == "Urban" else 0

    sample = np.array([[
        area,
        bedrooms,
        bathrooms,
        year,
        loc_downtown,
        loc_suburban,
        loc_rural,
        loc_urban
    ]])

    sample = scaler.transform(sample)

    price = model.predict(sample)

    st.success(f"Estimated Property Value: ${price[0]:,.2f}")

st.markdown("---")
st.write("HousePricePro Realty © 2026")