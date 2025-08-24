import streamlit as st
import plotly.express as px
from prediction_helper import PredictionHelper

# Initialize helper
helper = PredictionHelper()

st.set_page_config(page_title="🌾 Crop Recommendation System", layout="wide")

st.title("🌾 Smart Crop Recommendation & Financial Analysis")

# Two-column layout
col1, col2 = st.columns(2)

# Farmer info (col1)
with col1:
    st.header("👨‍🌾 Farmer Information")
    farmer_name = st.text_input("Farmer Name")
    region = st.text_input("Region")
    land_acres = st.number_input("Total Acres of Land", min_value=1, step=1)

# Soil & Environmental Parameters (col2)
with col2:
    st.header("🌍 Soil & Environmental Parameters")
    n = st.number_input("Nitrogen (N)", min_value=0)
    p = st.number_input("Phosphorus (P)", min_value=0)
    k = st.number_input("Potassium (K)", min_value=0)
    temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.5)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.5)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)

# Recommendation button
if st.button("🔍 Recommend Crop"):
    if land_acres <= 0:
        st.error("⚠️ Please enter valid land acres")
    else:
        crop = helper.recommend_crop(n, p, k, temp, humidity, ph, region, land_acres, farmer_name)
        st.success(f"✅ Recommended Crop for {farmer_name} ({region}): **{crop.capitalize()}**")

        # Financial analysis
        st.subheader("💰 Financial Analysis (Recommended Crop)")
        finance = helper.financial_analysis(crop, land_acres)
        st.json(finance)

        # All crops comparison
        st.subheader("📊 Financial Breakdown (All Crops)")
        df = helper.all_crops_analysis(land_acres)

        # Pie chart (medium size)
        pie = px.pie(df, names="Crop", values="Net Profit", title="Net Profit Share by Crop", width=600, height=400)
        st.plotly_chart(pie)

        # Bar chart (medium size)
        bar = px.bar(df, x="Crop", y="Net Profit", title="Net Profit by Crop", width=600, height=400)
        st.plotly_chart(bar)