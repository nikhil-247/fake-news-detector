# Fake News Detection using Machine Learning

Detect whether a news article is **REAL** or **FAKE** using a trained ML model, Flask, openrouter api and an interactive web frontend.

---

## 🚀 Demo

Try it locally:  
`http://127.0.0.1:5000`

---

## 📌 Features

- ✅ Machine Learning Model with 94%+ Accuracy
- ✅ Real vs Fake News Classification
- ✅ Clean and Modern Web Interface
- ✅ Flask Backend with REST API
- ✅ Fully Functional Chatbot (Optional Add-on)

---

## 🧠 How It Works

1. Dataset: Merged `true.csv` & `fake.csv` from Kaggle  
2. Preprocessing: Cleaned text, removed stopwords, used TF-IDF  
3. Model: Trained with PassiveAggressiveClassifier  
4. Deployment: Model saved using `joblib`, served via Flask  
5. Frontend: HTML, CSS & JS to interact with backend API

---

## 🧰 Tech Stack

- Python 3.12
- Flask
- scikit-learn
- pandas / numpy
- HTML5, CSS3, JavaScript
- Joblib
- Optional: OpenAI or Gemini API for Chatbot

---

## 💻 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector
