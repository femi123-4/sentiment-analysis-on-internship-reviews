# 📊 Internship Experience Sentiment Analyzer

## Overview
This project is a machine learning-powered web application that analyzes internship experience reviews and classifies them as positive or negative using Natural Language Processing (NLP).

It provides both single-text analysis and batch processing through CSV uploads using a trained machine learning model deployed with Streamlit.

---

## 🚀 Features

- 🧠 Sentiment classification using Machine Learning (TF-IDF + Logistic Regression)
- ✍️ Real-time sentiment analysis for single text input
- 📁 Batch analysis via CSV upload
- 😊 Sentiment classification (Positive / Negative)
- 📊 Summary statistics of results
- 📉 Confidence scoring (if supported by model)
- 🖥 Interactive Streamlit web interface

---

## 🧠 How It Works

1. User enters a review or uploads a CSV file
2. Text is converted into numerical features using TF-IDF vectorization
3. A trained Logistic Regression model predicts sentiment
4. Results are displayed in an interactive dashboard

---

## 🏗 Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Pickle (for model storage)
- TF-IDF Vectorization

---

## 📂 Project Structure
sentiment-analysis-project/
│
├── app.py # Streamlit web application
├── model.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── train_model.py # Model training script (if included)
├── requirements.txt # Dependencies
└── README.md # Project documentation



---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sentiment-analysis-project.git
cd sentiment-analysis-project
pip install -r requirements.txt
streamlit run app.py 
```