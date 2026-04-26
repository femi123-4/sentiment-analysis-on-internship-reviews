import streamlit as st
import pandas as pd
from transformers import pipeline

# ---------------------------
# Load model (cached for speed)
# ---------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

# ---------------------------
# App UI
# ---------------------------
st.title("📊 Internship Experience Sentiment Analyzer")

st.write("""
Analyze how people feel about their internship experiences using NLP.
Enter a review or upload a CSV file to get started.
""")

# ---------------------------
# Single text input
# ---------------------------
st.header("✍️ Single Review Analysis")

text = st.text_area("Enter an internship experience:")

if st.button("Analyze Text"):
    if text.strip() != "":
        result = model(text)[0]

        label = result["label"]
        score = result["score"]

        if label == "POSITIVE":
            st.success(f"Positive 😊 ({score:.2f})")
        else:
            st.error(f"Negative 😠 ({score:.2f})")
    else:
        st.warning("Please enter some text.")

# ---------------------------
# Batch CSV upload
# ---------------------------
st.header("📁 Batch Analysis (CSV Upload)")

uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("CSV must contain a 'review' column.")
    else:
        st.write("Preview:")
        st.dataframe(df.head())

        results = model(df["review"].tolist())

        df["sentiment"] = [r["label"] for r in results]
        df["confidence"] = [r["score"] for r in results]

        st.write("Results:")
        st.dataframe(df)

        # Summary
        st.subheader("📊 Summary")
        st.write(df["sentiment"].value_counts())