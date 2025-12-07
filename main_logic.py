# main_logic.py

import numpy as np

def calculate_fgi(volatility, price_change, volume, interest_rate, sentiment_text):
    """
    Simple Fear & Greed Index calculator using numeric inputs and sentiment keywords.
    """

    # Normalize numerical indicators
    vol_score = np.clip((50 - volatility) * 2, 0, 100)      # High volatility → fear
    price_score = np.clip((price_change + 10) * 5, 0, 100)  # -10% → 0, +10% → 100
    volume_score = np.clip(volume * 2, 0, 100)              # High volume → greed
    rate_score = np.clip((10 - interest_rate) * 10, 0, 100) # High interest → fear

    # Basic text-based sentiment analysis (no external libraries)
    sentiment_text = sentiment_text.lower()
    positive_words = ["gain", "growth", "optimistic", "profit", "bullish", "rise", "strong"]
    negative_words = ["loss", "fall", "drop", "bearish", "fear", "inflation", "crash"]

    sentiment_score = 0
    for word in positive_words:
        if word in sentiment_text:
            sentiment_score += 1
    for word in negative_words:
        if word in sentiment_text:
            sentiment_score -= 1

    # Scale sentiment score to -1 to +1 range
    sentiment_score = np.clip(sentiment_score / 5, -1, 1)

    # Combine all features to compute FGI
    fgi_value = (
        (vol_score * 0.3)
        + (price_score * 0.3)
        + (volume_score * 0.2)
        + (rate_score * 0.2)
        + (sentiment_score * 50)
    )

    fgi_value = np.clip(fgi_value, 0, 100)

    # Determine market emotion
    if fgi_value < 25:
        emotion = "🧊 Extreme Fear"
    elif fgi_value < 50:
        emotion = "😨 Fear"
    elif fgi_value < 75:
        emotion = "💰 Greedy"
    else:
        emotion = "🔥 Extreme Greed"

    return fgi_value, sentiment_score, emotion
