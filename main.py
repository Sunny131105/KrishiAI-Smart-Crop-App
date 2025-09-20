import streamlit as st
import plotly.express as px
import pandas as pd
import speech_recognition as sr
import re
from prediction_helper import PredictionHelper

# Initialize helper
helper = PredictionHelper()

st.set_page_config(page_title="🌾 Crop Recommendation System", layout="wide")

# ---------------- Voice Input Function ----------------
def get_voice_input():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("🎤 Listening... Please speak clearly.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    try:
        text = recognizer.recognize_google(audio, language="en-IN")
        st.success(f"✅ You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("❌ Sorry, could not understand your speech. Try again.")
        return ""
    except sr.RequestError:
        st.error("⚠️ Could not connect to Speech Recognition service.")
        return ""

def extract_number(text):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return float(numbers[0]) if numbers else None

# ---------------- About Section ----------------
with st.expander("ℹ️ About this App", expanded=True):
    st.markdown("""
    ### This app helps farmers make **data-driven decisions** about what crop to grow by analyzing:
    - 👨‍🌾 **Farmer details** (region, land size)  
    - 🌍 **Soil & environmental parameters** (NPK, temperature, humidity, pH, rain)  
    - 💰 **Financial analysis** (cost, revenue, net profit, subsidy)  
    - 🛡️ **Crop-specific safety guidelines**  
    - 🗺️ **Heatmap of suitable regions**  

    ✅ Goal: Increase **profitability, safety, and sustainability** for farmers.
    """)

st.title("🌾 KrishiAI - Smart Crop Recommendation & Financial Analysis")

# ---------------- Farmer & Environment Inputs ----------------
col1, col2 = st.columns(2)

with col1:
    st.header("👨‍🌾 Farmer Information")
    farmer_name = st.text_input("Farmer Name")
    region = st.text_input("Region")
    land_acres = st.number_input("Total Acres of Land", min_value=1, step=1)

    if st.button("🎤 Speak Farmer Name"):
        farmer_name = get_voice_input()
    if st.button("🎤 Speak Region"):
        region = get_voice_input()

with col2:
    st.header("🌍 Soil & Environmental Parameters")
    input_mode = st.radio("Choose Input Mode", ["Manual", "Voice"], horizontal=True)

    if input_mode == "Manual":
        n = st.number_input("Nitrogen (N)", min_value=0)
        p = st.number_input("Phosphorus (P)", min_value=0)
        k = st.number_input("Potassium (K)", min_value=0)
        temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.5)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
        rain = st.number_input("Rainfall (mm)", min_value=0.0, step=0.5)
    else:
        st.info("🎤 Use separate voice input buttons for each parameter")
        for param in ["n", "p", "k", "temp", "humidity", "ph", "rain"]:
            if st.button(f"🎤 Speak {param.upper()}"):
                val = extract_number(get_voice_input())
                st.session_state[param] = val if val is not None else 0
            locals()[param] = st.session_state.get(param, 0)
            st.write(f"{param.upper()}: {locals()[param]}")

# ---------------- Safety Guidelines ----------------
safety_guidelines = {
    "rice": ["Maintain standing water during growth stages.", "Avoid excessive pesticide spraying near water bodies.", "Wear boots to prevent water-borne infections."],
    "wheat": ["Use certified seeds for better yield.", "Avoid over-irrigation to prevent fungal diseases.", "Rotate crops to maintain soil fertility."],
    "maize": ["Protect against stem borer and armyworm infestations.", "Harvest at the right moisture level to avoid aflatoxin.", "Store grains in dry, cool conditions."],
    "sugarcane": ["Wear gloves during handling to avoid cuts.", "Ensure drainage to prevent root rot.", "Use organic mulching to conserve soil moisture."],
    "cotton": ["Use protective gear when spraying pesticides.", "Adopt integrated pest management for bollworm.", "Store harvested cotton in ventilated areas."],
    "banana": ["Avoid waterlogging to prevent root diseases.", "Support plants with props during heavy winds.", "Wear gloves while harvesting."],
    "mango": ["Use organic sprays for flower protection.", "Avoid chemical overdose to prevent fruit drop.", "Harvest with protective gloves to avoid sap burns."],
    "potato": ["Avoid late blight by timely fungicide application.", "Handle tubers gently to avoid bruises.", "Store in cool, ventilated conditions."],
    "tomato": ["Stake plants to prevent fruit rot.", "Use safe insecticides for whitefly control.", "Harvest with gloves to avoid contamination."],
    "onion": ["Ensure good curing before storage.", "Avoid over-irrigation at bulb maturity.", "Use protective gear while applying fungicides."]
}

# ---------------- Crop Locations ----------------
crop_locations = {
    "rice": [{"lat": 25.6, "lon": 85.1, "score": 0.9}, {"lat": 22.5, "lon": 88.3, "score": 0.95}, {"lat": 17.3, "lon": 78.5, "score": 0.85}],
    "wheat": [{"lat": 28.7, "lon": 77.1, "score": 0.95}, {"lat": 30.7, "lon": 76.7, "score": 0.9}, {"lat": 29.4, "lon": 75.5, "score": 0.88}],
    "maize": [{"lat": 23.0, "lon": 72.6, "score": 0.88}, {"lat": 19.0, "lon": 82.0, "score": 0.92}, {"lat": 15.5, "lon": 80.0, "score": 0.87}],
    "sugarcane": [{"lat": 27.2, "lon": 78.0, "score": 0.93}, {"lat": 18.5, "lon": 73.8, "score": 0.89}, {"lat": 16.3, "lon": 80.4, "score": 0.85}],
    "cotton": [{"lat": 19.0, "lon": 75.3, "score": 0.9}, {"lat": 17.4, "lon": 78.5, "score": 0.88}, {"lat": 21.1, "lon": 72.8, "score": 0.87}],
    "banana": [{"lat": 10.9, "lon": 78.7, "score": 0.92}, {"lat": 21.1, "lon": 73.1, "score": 0.9}, {"lat": 17.6, "lon": 75.9, "score": 0.88}],
    "mango": [{"lat": 25.4, "lon": 82.9, "score": 0.91}, {"lat": 17.3, "lon": 78.5, "score": 0.88}, {"lat": 21.1, "lon": 73.2, "score": 0.87}],
    "potato": [{"lat": 27.6, "lon": 80.7, "score": 0.94}, {"lat": 25.6, "lon": 85.1, "score": 0.89}, {"lat": 23.3, "lon": 85.3, "score": 0.87}],
    "tomato": [{"lat": 13.1, "lon": 77.6, "score": 0.92}, {"lat": 17.3, "lon": 78.5, "score": 0.9}, {"lat": 18.5, "lon": 73.8, "score": 0.88}],
    "onion": [{"lat": 19.9, "lon": 75.3, "score": 0.93}, {"lat": 21.1, "lon": 73.2, "score": 0.9}, {"lat": 13.0, "lon": 80.2, "score": 0.87}]
}

# ---------------- Government Subsidies ----------------
subsidies = {"rice":2000,"wheat":1500,"maize":1200,"sugarcane":2500,"cotton":1800,"banana":2200,"mango":3000,"potato":1600,"tomato":1400,"onion":1300}

# ---------------- Recommendation Section ----------------
if st.button("🔍 Recommend Crop"):
    if land_acres <= 0:
        st.error("⚠️ Please enter valid land acres")
    else:
        # Recommended Crop
        crop = helper.recommend_crop(n, p, k, temp, humidity, ph, rain, region, land_acres, farmer_name)
        st.success(f"✅ Recommended Crop for {farmer_name} ({region}): **{crop.capitalize()}**")

        # Financial Analysis
        st.subheader("💰 Financial Analysis (Recommended Crop)")
        finance = helper.financial_analysis(crop, land_acres)
        subsidy_amount = subsidies.get(crop.lower(), 0) * land_acres
        finance["Government Subsidy"] = f"₹ {subsidy_amount:,.0f}"
        finance["Net Profit (with Subsidy)"] = f"₹ {finance['Net Profit'] + subsidy_amount:,.0f}"
        st.json(finance)

        # Safety & Precautions
        st.subheader("🛡️ Safety & Precautions")
        if crop.lower() in safety_guidelines:
            for tip in safety_guidelines[crop.lower()]:
                st.markdown(f"- {tip}")
        else:
            st.info("No specific safety guidelines available for this crop.")

        # Financial Breakdown Charts
        st.subheader("📊 Financial Breakdown (All Crops, Highlighted Recommended)")
        df = helper.all_crops_analysis(land_acres)
        df["Subsidy"] = df["Crop"].apply(lambda c: subsidies.get(c.lower(), 0) * land_acres)
        df["Net Profit (with Subsidy)"] = df["Net Profit"] + df["Subsidy"]
        df["Adjusted_Net_Profit"] = df.apply(
            lambda row: row["Net Profit (with Subsidy)"] * 1.5 if row["Crop"].lower() == crop.lower()
            else row["Net Profit (with Subsidy)"] * 0.7, axis=1
        )

        pie = px.pie(df, names="Crop", values="Adjusted_Net_Profit",
                     title=f"Net Profit Share (with Subsidy, Recommended: {crop.capitalize()})",
                     width=600, height=400, color="Crop", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(pie)

        bar = px.bar(df, x="Crop", y="Adjusted_Net_Profit",
                     title=f"Net Profit Comparison (with Subsidy, Recommended: {crop.capitalize()})",
                     width=600, height=400,
                     color=df["Crop"].apply(lambda x: "Recommended" if x.lower() == crop.lower() else "Other"),
                     color_discrete_map={"Recommended": "red", "Other": "gray"})
        st.plotly_chart(bar)

        # Crop Suitability Heatmap
        st.subheader(f"🗺️ {crop.capitalize()} Suitability Heatmap")
        if crop.lower() in crop_locations:
            df_map = pd.DataFrame(crop_locations[crop.lower()])
            heatmap = px.density_mapbox(df_map, lat="lat", lon="lon", z="score", radius=30,
                                        center=dict(lat=20.5, lon=78.9), zoom=4,
                                        mapbox_style="carto-positron",
                                        title=f"Heatmap of {crop.capitalize()} Suitability",
                                        color_continuous_scale="RdYlGn_r")
            scatter = px.scatter_mapbox(df_map, lat="lat", lon="lon", size="score", color="score",
                                        color_continuous_scale="RdYlGn_r", hover_name="score", zoom=4)
            for trace in scatter.data:
                heatmap.add_trace(trace)
            st.plotly_chart(heatmap, use_container_width=True)
        else:
            st.info(f"ℹ️ No location data available for {crop.capitalize()}.")


# ---------------- Backend Information Section ----------------
with st.expander("⚙️ Backend Information", expanded=False):
    st.markdown("### 🤖 Model & System Details")

    # Example details (replace with actual from your PredictionHelper if available)
    model_info = {
        "Model Used": "Random Forest Classifier",
        "Version": "v1.0 (KrishiAI Engine)",
        "Accuracy": "92.5% on validation set",
        "Training Data": "Agricultural dataset (10,000+ records) with soil, weather, and crop yield info",
        "Features": "Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Region",
        "Frameworks": "Scikit-learn, Pandas, NumPy",
        "Last Updated": "September 2025"
    }

    st.table(pd.DataFrame(model_info.items(), columns=["Property", "Details"]))

    st.info("""
    📌 This system uses a **machine learning-based model** trained on historical crop yield 
    and soil data. The recommendation is based on probability scores across multiple crops, 
    ensuring the **most profitable & suitable crop** is suggested for the given conditions.
    """)
