# # from flask import Flask, request, jsonify
# # import joblib
# # import numpy as np

# # # Initialize Flask app
# # app = Flask(__name__)

# # # Load the trained model and vectorizer
# # model = joblib.load('model.pkl')
# # vectorizer = joblib.load('vectorizer.pkl')

# # # Route to check if the model is working
# # @app.route('/')
# # def home():
# #     return "Fake News Detection Model is Running!"

# # # Route to make predictions
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     # Get the article from the request
# #     article = request.json['article']
    
# #     # Preprocess the text (same cleaning as in training)
# #     article_cleaned = clean_text(article)
    
# #     # Vectorize the input article
# #     article_vectorized = vectorizer.transform([article_cleaned])
    
# #     # Get model prediction
# #     prediction = model.predict(article_vectorized)
# #     prediction_proba = model.predict_proba(article_vectorized)  # Get confidence score
    
# #     # Confidence score in percentage
# #     confidence = np.max(prediction_proba) * 100
    
# #     # Response
# #     result = {
# #         'prediction': 'Real' if prediction[0] == 'REAL' else 'Fake',
# #         'confidence': round(confidence, 2)
# #     }
    
# #     return jsonify(result)

# # # Run the app
# # if __name__ == '__main__':
# #     app.run(debug=True)
# from flask import Flask, request, jsonify
# import joblib
# import re
# import string

# app = Flask(__name__)

# model = joblib.load('model.pkl')
# vectorizer = joblib.load('vectorizer.pkl')

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     return text

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     news = data.get("text", "")
#     clean = clean_text(news)
#     vec = vectorizer.transform([clean])
#     prediction = model.predict(vec)[0]
#     prob = model.decision_function(vec)[0]
#     confidence = min(max(abs(prob) / 3, 0), 1) * 100  # Approximation

#     return jsonify({
#         "prediction": prediction,
#         "confidence": round(confidence, 2)
#     })

# if __name__ == "__main__":
#     app.run(debug=True)
# from flask import Flask, request, jsonify
# import joblib
# import re
# import string

# app = Flask(__name__)

# model = joblib.load('model.pkl')
# vectorizer = joblib.load('vectorizer.pkl')

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     return text

# @app.route("/")
# def home():
#     return "Fake News Detection API is running!"

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     news = data.get("text", "")
#     clean = clean_text(news)
#     vec = vectorizer.transform([clean])
#     prediction = model.predict(vec)[0]
#     prob = model.decision_function(vec)[0]
#     confidence = min(max(abs(prob) / 3, 0), 1) * 100  # Approximation

#     return jsonify({
#         "prediction": prediction,
#         "confidence": round(confidence, 2)
#     })

# if __name__ == "__main__":
#     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import joblib, re, string

# app = Flask(__name__)
# CORS(app)

# model = joblib.load('model.pkl')
# vectorizer = joblib.load('vectorizer.pkl')

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     return text

# @app.route("/")
# def home():
#     return render_template("fake-news-detector\frontend\index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     news = data.get("text", "")
#     clean = clean_text(news)
#     vec = vectorizer.transform([clean])
#     prediction = model.predict(vec)[0]
#     prob = model.decision_function(vec)[0]
#     confidence = min(max(abs(prob) / 3, 0), 1) * 100

#     return jsonify({
#         "prediction": prediction,
#         "confidence": round(confidence, 2)
#     })

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route("/")
def home():
    return render_template("sejal.frontend\index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    news = data["news"]
    vec_news = vectorizer.transform([news])
    prediction = model.predict(vec_news)[0]
    return jsonify({"result": prediction})

if __name__ == "__main__":
    app.run(debug=True)

