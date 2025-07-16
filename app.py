import streamlit as st
import joblib
import re

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("📰 Fake News Detector")
text = st.text_area("Enter news text or headline")

if st.button("Predict"):
    vec = tfidf.transform([clean(text)])
    result = model.predict(vec)[0]
    st.success("✅ REAL News" if result == 1 else "❌ FAKE News")
