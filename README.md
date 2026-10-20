# Fake News Text Classifier

An educational machine-learning prototype that classifies a news article's text as `REAL` or `FAKE` using a TF-IDF + SGDClassifier using a passive-aggressive learning schedule.

**Important:** This is a text-classification demo, not a fact-checking service. It can learn misleading patterns from its training data and may be wrong about new or out-of-distribution articles. A predicted label is not evidence that a claim is true or false.

## What is included

- A reproducible training script with a stratified hold-out split.
- A saved scikit-learn pipeline (vectorizer + classifier).
- A small Flask API with input validation and a model-readiness health check.
- Automated API tests.

The datasets and trained model are not committed. You provide the CSV files locally, train a model, then run the API. The older scripts in this repository are kept as legacy experiments; use the root-level flow below for the documented path.

## Requirements

- Python 3.10+
- CSV files from a dataset you are permitted to use, with a `text` column. The common Fake/True CSV format works.

## Setup

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Place your source files at `data/raw/Fake.csv` and `data/raw/True.csv` (or pass other paths to the script). These files are intentionally not included in this repository.

## Train and evaluate

```bash
python train_model.py --fake-csv data/raw/Fake.csv --real-csv data/raw/True.csv --model-path artifacts/news_classifier.joblib
```

The script removes empty article text, trains on a stratified training split, and prints held-out accuracy and a classification report before saving the complete vectorizer/classifier pipeline. Report those measured results rather than assuming a fixed accuracy.

## Run the API

Set the path to the trained model and start Flask.

PowerShell:

```powershell
$env:MODEL_PATH = "artifacts/news_classifier.joblib"
python app.py
```

macOS/Linux:

```bash
MODEL_PATH=artifacts/news_classifier.joblib python app.py
```

The server listens on `http://127.0.0.1:5000` by default.

### Endpoints

- `GET /api/health` — reports whether the trained model is available.
- `POST /api/predict` — accepts a JSON object containing `text`.

Example request:

```json
{
  "text": "Paste a news article here."
}
```

Example response:

```json
{
  "label": "REAL",
  "note": "Model classification only; independently verify the claim and sources."
}
```

A prediction request returns `400` for invalid or empty input, `413` for text over 20,000 characters, and `503` when no trained model is available.

## Tests

```bash
pytest -q
```

The tests cover the API contract and missing-model behavior. They do not validate model quality; the training script's held-out evaluation is a separate check.

## Responsible use

- Do not use a model label as the sole basis for deciding whether a story is true.
- Evaluate on data from sources and time periods not represented in training.
- Inspect class balance, precision, recall, confusion matrix, and data leakage.
- Do not report an accuracy number unless it comes from a reproducible held-out evaluation.
