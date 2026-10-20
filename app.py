from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
MAX_ARTICLE_CHARS = 20_000


@lru_cache(maxsize=1)
def get_model() -> Any:
    """Load the trained vectorizer/classifier pipeline on first use."""
    model_path = Path(os.getenv("MODEL_PATH", "artifacts/news_classifier.joblib"))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run train_model.py first."
        )
    return joblib.load(model_path)


@app.get("/api/health")
def health():
    try:
        get_model()
    except FileNotFoundError:
        return jsonify({"status": "not_ready", "model_loaded": False}), 503
    except (OSError, ValueError, EOFError):
        app.logger.exception("Could not load the configured classifier artifact")
        return jsonify({"status": "error", "model_loaded": False}), 500
    return jsonify({"status": "ok", "model_loaded": True})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON object with a 'text' field is required."}), 400

    article = payload.get("text")
    if not isinstance(article, str) or not article.strip():
        return jsonify({"error": "'text' must be a non-empty string."}), 400
    article = article.strip()
    if len(article) > MAX_ARTICLE_CHARS:
        return jsonify({"error": f"'text' must be at most {MAX_ARTICLE_CHARS} characters."}), 413

    try:
        model = get_model()
    except FileNotFoundError:
        return jsonify({"error": "Model is not trained yet. Run train_model.py first."}), 503
    except (OSError, ValueError, EOFError):
        app.logger.exception("Could not load the configured classifier artifact")
        return jsonify({"error": "The configured model could not be loaded."}), 500

    label = str(model.predict([article])[0]).upper()
    if label not in {"REAL", "FAKE"}:
        app.logger.error("Classifier returned unexpected label: %s", label)
        return jsonify({"error": "The model returned an unsupported label."}), 500

    return jsonify(
        {
            "label": label,
            "note": "Model classification only; independently verify the claim and sources.",
        }
    )


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
