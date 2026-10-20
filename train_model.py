from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_articles(fake_csv: Path, real_csv: Path) -> pd.DataFrame:
    """Load labeled CSVs and validate the expected article-text column."""
    fake = pd.read_csv(fake_csv)
    real = pd.read_csv(real_csv)
    for name, frame in (("fake", fake), ("real", real)):
        if "text" not in frame.columns:
            raise ValueError(f"The {name} CSV must contain a 'text' column.")

    fake_rows = fake.loc[:, ["text"]].copy()
    fake_rows["label"] = "FAKE"
    real_rows = real.loc[:, ["text"]].copy()
    real_rows["label"] = "REAL"

    data = pd.concat([fake_rows, real_rows], ignore_index=True)
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data = data.loc[data["text"].ne("")].copy()

    # The same article must not silently be labeled both FAKE and REAL.
    label_counts = data.groupby("text", sort=False)["label"].nunique()
    ambiguous_texts = set(label_counts[label_counts > 1].index)
    if ambiguous_texts:
        data = data.loc[~data["text"].isin(ambiguous_texts)]
    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if data["label"].nunique() != 2:
        raise ValueError("Both REAL and FAKE examples are required.")
    if data["label"].value_counts().min() < 2:
        raise ValueError("At least two non-empty examples per class are required.")
    return data


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.95)),
            (
                "classifier",
                SGDClassifier(
                    loss="hinge",
                    penalty=None,
                    learning_rate="pa1",
                    eta0=1.0,
                    max_iter=1000,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TF-IDF fake/real text classifier.")
    parser.add_argument("--fake-csv", type=Path, default=Path("data/raw/Fake.csv"))
    parser.add_argument("--real-csv", type=Path, default=Path("data/raw/True.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/news_classifier.joblib"))
    args = parser.parse_args()

    data = load_articles(args.fake_csv, args.real_csv)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["text"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)

    print(f"Examples after cleaning/deduplication: {len(data):,}")
    print(f"Train examples: {len(train_texts):,} | Held-out examples: {len(test_texts):,}")
    print(f"Held-out accuracy: {accuracy_score(test_labels, predictions):.4f}")
    print("Held-out classification report:")
    print(classification_report(test_labels, predictions, zero_division=0))

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_path)
    print(f"Saved trained pipeline to: {args.model_path}")


if __name__ == "__main__":
    main()
