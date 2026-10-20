from __future__ import annotations

import pandas as pd

import app as app_module
from train_model import build_pipeline, load_articles


def test_health_is_unavailable_when_model_is_missing(monkeypatch):
    def missing_model():
        raise FileNotFoundError("not trained")

    monkeypatch.setattr(app_module, "get_model", missing_model)
    response = app_module.app.test_client().get("/api/health")

    assert response.status_code == 503
    assert response.get_json()["model_loaded"] is False


def test_predict_rejects_empty_text():
    response = app_module.app.test_client().post("/api/predict", json={"text": "  "})

    assert response.status_code == 400
    assert "non-empty" in response.get_json()["error"]


def test_predict_rejects_non_json_body():
    response = app_module.app.test_client().post(
        "/api/predict", data="not json", content_type="text/plain"
    )

    assert response.status_code == 400


def test_predict_returns_label_and_scope_note(monkeypatch):
    class FakeModel:
        def predict(self, texts):
            assert texts == ["A sample news article."]
            return ["REAL"]

    monkeypatch.setattr(app_module, "get_model", lambda: FakeModel())
    response = app_module.app.test_client().post(
        "/api/predict", json={"text": "A sample news article."}
    )

    assert response.status_code == 200
    assert response.get_json()["label"] == "REAL"
    assert "independently verify" in response.get_json()["note"]


def test_load_articles_cleans_empty_rows_and_removes_duplicate_text(tmp_path):
    fake_path = tmp_path / "Fake.csv"
    real_path = tmp_path / "True.csv"
    pd.DataFrame({"text": ["fake article alpha", "fake article delta", "fake article delta", "", None]}).to_csv(fake_path, index=False)
    pd.DataFrame({"text": ["real article beta", "real article gamma"]}).to_csv(real_path, index=False)

    data = load_articles(fake_path, real_path)

    assert len(data) == 4
    assert set(data["label"]) == {"FAKE", "REAL"}
    assert data["text"].is_unique
    assert not data["text"].str.strip().eq("").any()


def test_training_pipeline_fits_and_predicts_distinct_text():
    pipeline = build_pipeline()
    texts = [
        "sports team wins championship match",
        "football players score winning goal",
        "coach discusses league season",
        "match result announced after tournament",
        "central bank adjusts monetary policy",
        "government publishes economic statistics",
        "finance ministry reports budget figures",
        "official data shows inflation trend",
    ]
    labels = ["REAL", "REAL", "REAL", "REAL", "FAKE", "FAKE", "FAKE", "FAKE"]

    pipeline.fit(texts, labels)
    predictions = pipeline.predict(["sports team wins match", "central bank publishes inflation data"])

    assert len(predictions) == 2
    assert set(predictions).issubset({"REAL", "FAKE"})


def test_load_articles_drops_text_with_conflicting_labels(tmp_path):
    fake_path = tmp_path / "Fake.csv"
    real_path = tmp_path / "True.csv"
    pd.DataFrame({"text": ["shared article", "fake article alpha", "fake article beta"]}).to_csv(fake_path, index=False)
    pd.DataFrame({"text": ["shared article", "real article alpha", "real article beta"]}).to_csv(real_path, index=False)

    data = load_articles(fake_path, real_path)

    assert "shared article" not in set(data["text"])
    assert set(data["label"]) == {"FAKE", "REAL"}
    assert data["text"].is_unique
