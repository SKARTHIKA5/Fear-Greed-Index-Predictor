# app.py

import streamlit as st
from main_logic import calculate_fgi

st.set_page_config(page_title="Fear & Greed Index Predictor", page_icon="📊")

st.title("📈 Fear & Greed Index Predictor")
st.markdown("Predict market emotions using market data and news sentiment!")

st.header("🧮 Input Market Data")
volatility = st.number_input("Market Volatility (0–50)", min_value=0.0, max_value=50.0, value=20.0)
price_change = st.number_input("Stock Price Change (%) (-10 to +10)", min_value=-10.0, max_value=10.0, value=0.0)
volume = st.number_input("Trading Volume (millions)", min_value=0.0, max_value=100.0, value=10.0)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=10.0, value=5.0)
sentiment_text = st.text_area("Enter Market News or Headline", "Investors expect a rise in profits.")

if st.button("🔍 Predict Market Emotion"):
    fgi_value, sentiment_score, emotion = calculate_fgi(volatility, price_change, volume, interest_rate, sentiment_text)
    st.success(f"Predicted Fear & Greed Index: {fgi_value:.2f}")
    st.info(f"Sentiment Score: {sentiment_score:.2f}")
    st.markdown(f"### Market Emotion: {emotion}")
    st.progress(int(fgi_value))
